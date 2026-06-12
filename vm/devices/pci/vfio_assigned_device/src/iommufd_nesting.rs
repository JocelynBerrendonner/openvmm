// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! iommufd nested translation for VFIO devices behind an accel-capable SMMU.
//!
//! This module implements HW-accelerated nested stage 1 translation using
//! iommufd. The guest programs the emulated SMMU's stream table entries (STEs)
//! and page tables. The VMM intercepts CMDQ commands via the
//! [`smmu::AcceleratedStreamBackend`] trait and forwards raw STE bytes and
//! invalidation commands to iommufd, which programs the host IOMMU hardware.
//!
//! # Architecture
//!
//! ```text
//! Guest programs emulated SMMU ──► CMDQ commands
//!        │
//!        ▼
//! SmmuDevice dispatches to registered AcceleratedStreamBackend
//!        │
//!        ▼
//! IommufdStreamBackend (per VFIO device)
//!   ├─ on_cfgi_ste: parse STE.Config, allocate/switch nested HWPT
//!   └─ on_tlbi: forward raw command to iommufd HWPT_INVALIDATE
//!        │
//!        ▼
//! Host IOMMU HW walks guest S1 tables ──► physical DMA
//! ```
//!
//! # Object Lifecycle
//!
//! - [`SmmuAccelState`]: per-SMMU iommufd objects (vIOMMU). Created lazily on
//!   first VFIO device attachment. Shared across all devices behind the same
//!   SMMU.
//! - [`IommufdStreamBackend`]: per-device stream backend. Created during VFIO
//!   cdev device resolution. Registered with [`smmu::SmmuSharedState`] by
//!   stream ID.

use anyhow::Context as _;
use parking_lot::Mutex;
use std::fs::File;
use std::os::unix::prelude::*;
use std::sync::Arc;
use vfio_sys::iommufd::IommufdCtx;

/// Query the physical SMMUv3's capabilities for a device bound to iommufd.
///
/// Issues a single `IOMMU_GET_HW_INFO` and decodes the fields the vSMMU
/// finalizes against the host (currently IDR5.OAS). Handed to
/// [`smmu::SmmuSharedState::resolve_host_caps`].
pub fn query_host_caps(ctx: &IommufdCtx, dev_id: u32) -> anyhow::Result<smmu::HostSmmuCaps> {
    let mut info = vfio_sys::iommufd::IommuHwInfoArmSmmuv3 {
        flags: 0,
        __reserved: 0,
        idr: [0; 6],
        iidr: 0,
        aidr: 0,
    };
    let (data_type, _caps) = ctx
        .get_hw_info(
            dev_id,
            std::ptr::from_mut(&mut info) as u64,
            size_of::<vfio_sys::iommufd::IommuHwInfoArmSmmuv3>() as u32,
        )
        .context("IOMMU_GET_HW_INFO failed")?;
    if data_type != vfio_sys::iommufd::IOMMU_HW_INFO_TYPE_ARM_SMMUV3 {
        anyhow::bail!("unexpected host IOMMU hw info type {data_type} (expected ARM SMMUv3)");
    }
    // IDR5.OAS is the low 3 bits of IDR5 (idr[5]).
    let oas_enc = (info.idr[5] & 0x7) as u8;
    let oas_bits = smmu::oas_bits_from_encoding(oas_enc)
        .with_context(|| format!("host SMMUv3 reported unknown OAS encoding {oas_enc}"))?;
    Ok(smmu::HostSmmuCaps { oas_bits })
}

/// STE Config field values (bits `[3:1]` of DW0).
///
/// Duplicated from smmu::spec to avoid re-exporting internal spec types.
mod ste_config {
    pub const ABORT: u8 = 0b000;
    pub const BYPASS: u8 = 0b100;
    pub const S1_TRANS: u8 = 0b101;
}

/// Per-SMMU iommufd objects for HW-accelerated nested translation.
///
/// Created lazily on first VFIO device attachment for an accel-capable SMMU.
/// Shared (via `Arc`) across all [`IommufdStreamBackend`] instances behind
/// the same SMMU.
///
/// The vIOMMU represents the emulated SMMU in the iommufd object model.
/// Nested HWPTs (per-device S1 translation contexts) and vDevices are
/// allocated under this vIOMMU.
pub struct SmmuAccelState {
    /// The iommufd context (shared with IoasManager).
    ctx: Arc<IommufdCtx>,
    /// Virtual IOMMU ID (one per emulated SMMU instance).
    viommu_id: u32,
    /// S2 parent HWPT ID (nesting parent, linked to IOAS).
    ///
    /// This HWPT provides GPA→HPA translation for all nested devices.
    /// Devices in BYPASS mode are attached directly to this HWPT.
    s2_parent_hwpt_id: u32,
}

impl SmmuAccelState {
    /// Create per-SMMU iommufd objects.
    ///
    /// `dev_id` is any device bound to this IOMMU. The iommufd kernel
    /// requires a device reference to determine which physical IOMMU
    /// backs the vIOMMU.
    ///
    /// `s2_parent_hwpt_id` is the S2 parent HWPT, previously allocated
    /// via `IOMMU_HWPT_ALLOC` with `NEST_PARENT`.
    pub fn new(ctx: Arc<IommufdCtx>, dev_id: u32, s2_parent_hwpt_id: u32) -> anyhow::Result<Self> {
        let viommu_id = ctx
            .viommu_alloc(
                vfio_sys::iommufd::IOMMU_VIOMMU_TYPE_ARM_SMMUV3,
                dev_id,
                s2_parent_hwpt_id,
            )
            .context("failed to allocate vIOMMU for accel SMMU")?;

        tracing::info!(
            viommu_id,
            s2_parent_hwpt_id,
            "created SMMU accel state (vIOMMU)"
        );

        Ok(Self {
            ctx,
            viommu_id,
            s2_parent_hwpt_id,
        })
    }

    /// Returns the vIOMMU ID.
    pub fn viommu_id(&self) -> u32 {
        self.viommu_id
    }

    /// Returns the S2 parent HWPT ID (used for BYPASS mode attachment).
    pub fn s2_parent_hwpt_id(&self) -> u32 {
        self.s2_parent_hwpt_id
    }

    /// Returns the iommufd context.
    pub fn ctx(&self) -> &Arc<IommufdCtx> {
        &self.ctx
    }
}

/// Per-device iommufd stream backend for HW-accelerated nested S1.
///
/// Implements [`smmu::AcceleratedStreamBackend`], bridging SMMU CMDQ
/// commands to iommufd nested HWPT operations. One instance per VFIO
/// device behind an accel-capable SMMU.
///
/// # STE Config Handling
///
/// | STE.Config | Action |
/// |------------|--------|
/// | ABORT (0)  | Detach device — DMA faults |
/// | BYPASS (4) | Attach to S2 parent HWPT — identity GPA→HPA |
/// | S1_TRANS (5) | Allocate nested HWPT with STE DW0-1, attach |
///
/// # vDevice Allocation
///
/// The iommufd vDevice (virtual device within the vIOMMU) is allocated
/// lazily on first `on_cfgi_ste` with `Config=S1_TRANS`. The vDevice's
/// virtual stream ID is the guest-assigned BDF, which is not known at
/// device construction time (the guest assigns bus numbers after PCIe
/// enumeration).
pub struct IommufdStreamBackend {
    /// Per-SMMU shared state (vIOMMU, S2 parent HWPT).
    accel: Arc<SmmuAccelState>,
    /// iommufd device ID (from cdev bind).
    dev_id: u32,
    /// Dup'd VFIO cdev device fd for attach/detach ioctls.
    ///
    /// The original cdev fd is consumed by `CdevDevice::into_device()`.
    /// This dup'd fd retains the ability to issue
    /// `VFIO_DEVICE_ATTACH_IOMMUFD_PT` / `VFIO_DEVICE_DETACH_IOMMUFD_PT`.
    device_fd: File,
    /// Per-device mutable state (nested HWPT, vDevice).
    state: Mutex<StreamBackendState>,
}

/// Per-device mutable state for an [`IommufdStreamBackend`].
struct StreamBackendState {
    /// Current nested HWPT ID, if S1 translation is active.
    /// `None` when in ABORT (detached) or BYPASS (attached to S2 parent).
    current_nested_hwpt: Option<u32>,
    /// vDevice ID, lazily allocated on first `CFGI_STE` with `S1_TRANS`.
    vdevice_id: Option<u32>,
}

impl IommufdStreamBackend {
    /// Create a new stream backend.
    ///
    /// `device_fd` must be a dup'd VFIO cdev fd (still bound to iommufd).
    /// The device should already be attached to the S2 parent HWPT
    /// (BYPASS mode) as its initial state.
    pub fn new(accel: Arc<SmmuAccelState>, dev_id: u32, device_fd: File) -> Self {
        Self {
            accel,
            dev_id,
            device_fd,
            state: Mutex::new(StreamBackendState {
                current_nested_hwpt: None,
                vdevice_id: None,
            }),
        }
    }

    /// Extract the STE Config field (bits [3:1] of DW0) from raw STE bytes.
    fn parse_ste_config(ste_bytes: &[u8; 64]) -> (bool, u8) {
        let dw0 = u64::from_le_bytes(ste_bytes[0..8].try_into().unwrap());
        let valid = (dw0 & 1) != 0;
        let config = ((dw0 >> 1) & 0x7) as u8;
        (valid, config)
    }

    // Nesting-allowed masks from the kernel's arm-smmu-v3.h.
    // Only these STE fields are accepted by IOMMU_HWPT_ALLOC for nested
    // domains; all other bits must be zero.
    //
    // DW0: V | CFG | S1FMT | S1CTXPTR | S1CDMAX
    const STE0_NESTING_ALLOWED: u64 = 0xFFFF_FFFF_FFFF_FFFF; // all of DW0 is covered by allowed fields
    // DW1: S1DSS | S1CIR | S1COR | S1CSH | S1STALLD | EATS
    const STE1_NESTING_ALLOWED: u64 = {
        let s1dss = 0x3; // bits [1:0]
        let s1cir = 0x3 << 2; // bits [3:2]
        let s1cor = 0x3 << 4; // bits [5:4]
        let s1csh = 0x3 << 6; // bits [7:6]
        let s1stalld = 1 << 27; // bit 27
        let eats = 0x3 << 28; // bits [29:28]
        s1dss | s1cir | s1cor | s1csh | s1stalld | eats
    };

    /// Extract STE DW0 and DW1 (first 16 bytes) for nested HWPT allocation.
    ///
    /// Masks off bits that the kernel does not allow for nested domains
    /// (e.g., SHCFG, STRW) to prevent EIO from `IOMMU_HWPT_ALLOC`.
    fn extract_ste_dwords(ste_bytes: &[u8; 64]) -> [u64; 2] {
        let dw0 = u64::from_le_bytes(ste_bytes[0..8].try_into().unwrap());
        let dw1 = u64::from_le_bytes(ste_bytes[8..16].try_into().unwrap());
        [
            dw0 & Self::STE0_NESTING_ALLOWED,
            dw1 & Self::STE1_NESTING_ALLOWED,
        ]
    }

    /// Handle STE Config=ABORT: detach from any HWPT.
    fn handle_abort(&self, state: &mut StreamBackendState) -> anyhow::Result<()> {
        // Detach from current attachment (if any).
        if state.current_nested_hwpt.is_some() {
            vfio_sys::cdev::detach_pt(self.device_fd.as_fd())
                .context("failed to detach device for ABORT")?;
        }

        // Destroy old nested HWPT.
        if let Some(old_hwpt) = state.current_nested_hwpt.take() {
            // Best-effort destroy — log but don't fail.
            if let Err(e) = self.accel.ctx.destroy(old_hwpt) {
                tracing::warn!(
                    old_hwpt,
                    error = %e,
                    "failed to destroy old nested HWPT on ABORT"
                );
            }
        }

        tracing::debug!(dev_id = self.dev_id, "SMMU accel: STE → ABORT (detached)");
        Ok(())
    }

    /// Handle STE Config=BYPASS: attach to S2 parent HWPT.
    fn handle_bypass(&self, state: &mut StreamBackendState) -> anyhow::Result<()> {
        // Detach from current nested HWPT (if any).
        if state.current_nested_hwpt.is_some() {
            vfio_sys::cdev::detach_pt(self.device_fd.as_fd())
                .context("failed to detach device for BYPASS switch")?;
        }

        // Attach to S2 parent HWPT (identity GPA→HPA).
        vfio_sys::cdev::attach_pt(self.device_fd.as_fd(), self.accel.s2_parent_hwpt_id)
            .context("failed to attach device to S2 parent HWPT for BYPASS")?;

        // Destroy old nested HWPT.
        if let Some(old_hwpt) = state.current_nested_hwpt.take() {
            if let Err(e) = self.accel.ctx.destroy(old_hwpt) {
                tracing::warn!(
                    old_hwpt,
                    error = %e,
                    "failed to destroy old nested HWPT on BYPASS"
                );
            }
        }

        tracing::debug!(dev_id = self.dev_id, "SMMU accel: STE → BYPASS (S2 parent)");
        Ok(())
    }

    /// Handle STE Config=S1_TRANS: allocate nested HWPT, attach device.
    fn handle_s1_translate(
        &self,
        state: &mut StreamBackendState,
        ste_bytes: &[u8; 64],
        stream_id: Option<u32>,
    ) -> anyhow::Result<()> {
        // Lazy vDevice allocation — the virtual stream ID comes from the
        // CFGI_STE command's SID, which is the guest-assigned BDF.
        if state.vdevice_id.is_none() {
            if let Some(sid) = stream_id {
                let vdev_id = self
                    .accel
                    .ctx
                    .vdevice_alloc(self.accel.viommu_id, self.dev_id, sid as u64)
                    .with_context(|| {
                        format!(
                            "failed to allocate vDevice for dev_id={}, vsid={}",
                            self.dev_id, sid
                        )
                    })?;
                tracing::info!(
                    dev_id = self.dev_id,
                    vdevice_id = vdev_id,
                    virtual_sid = sid,
                    "allocated iommufd vDevice"
                );
                state.vdevice_id = Some(vdev_id);
            }
        }

        // Extract STE DW0-1 for the nested HWPT allocation.
        let ste_dwords = Self::extract_ste_dwords(ste_bytes);
        let ste_data = vfio_sys::iommufd::IommuHwptArmSmmuv3 { ste: ste_dwords };

        tracing::info!(
            dev_id = self.dev_id,
            ste_dw0 = format_args!("{:#018x}", ste_dwords[0]),
            ste_dw1 = format_args!("{:#018x}", ste_dwords[1]),
            "SMMU accel: allocating nested HWPT with STE data"
        );

        // Allocate a new nested HWPT under the vIOMMU.
        let new_hwpt = self
            .accel
            .ctx
            .hwpt_alloc(
                0, // flags: not a nest parent
                self.dev_id,
                self.accel.viommu_id, // parent is the vIOMMU
                vfio_sys::iommufd::IOMMU_HWPT_DATA_ARM_SMMUV3,
                std::ptr::from_ref(&ste_data) as u64,
                size_of::<vfio_sys::iommufd::IommuHwptArmSmmuv3>() as u32,
            )
            .context("failed to allocate nested HWPT for S1_TRANS")?;

        // Detach from current attachment.
        // The device may be attached to S2 parent (BYPASS) or an old
        // nested HWPT (previous S1_TRANS).
        vfio_sys::cdev::detach_pt(self.device_fd.as_fd())
            .context("failed to detach device for S1_TRANS switch")?;

        // Attach to the new nested HWPT.
        vfio_sys::cdev::attach_pt(self.device_fd.as_fd(), new_hwpt)
            .context("failed to attach device to nested HWPT")?;

        // Destroy old nested HWPT (if any).
        if let Some(old_hwpt) = state.current_nested_hwpt.replace(new_hwpt) {
            if let Err(e) = self.accel.ctx.destroy(old_hwpt) {
                tracing::warn!(
                    old_hwpt,
                    error = %e,
                    "failed to destroy old nested HWPT on S1_TRANS"
                );
            }
        }

        tracing::debug!(
            dev_id = self.dev_id,
            nested_hwpt = new_hwpt,
            "SMMU accel: STE → S1_TRANS (nested HWPT)"
        );
        Ok(())
    }
}

impl smmu::AcceleratedStreamBackend for IommufdStreamBackend {
    fn on_cfgi_ste(&self, sid: u32, ste_bytes: &[u8; 64]) -> anyhow::Result<()> {
        let (valid, config) = Self::parse_ste_config(ste_bytes);

        // Invalid STE (V=0) is treated as ABORT.
        if !valid {
            let mut state = self.state.lock();
            return self.handle_abort(&mut state);
        }

        let mut state = self.state.lock();
        match config {
            ste_config::ABORT => self.handle_abort(&mut state),
            ste_config::BYPASS => self.handle_bypass(&mut state),
            ste_config::S1_TRANS => self.handle_s1_translate(&mut state, ste_bytes, Some(sid)),
            other => {
                tracelimit::warn_ratelimited!(
                    dev_id = self.dev_id,
                    config = other,
                    "SMMU accel: unsupported STE config for nested S1"
                );
                // Treat unsupported configs as ABORT.
                self.handle_abort(&mut state)
            }
        }
    }

    fn on_tlbi(&self, cmd_bytes: &[u8; 16]) -> anyhow::Result<()> {
        // Forward the raw 16-byte CMDQ entry to iommufd via
        // IOMMU_HWPT_INVALIDATE on the vIOMMU.
        let invalidate_entry = vfio_sys::iommufd::IommuViommuArmSmmuv3Invalidate {
            cmd: [
                u64::from_le_bytes(cmd_bytes[0..8].try_into().unwrap()),
                u64::from_le_bytes(cmd_bytes[8..16].try_into().unwrap()),
            ],
        };

        self.accel
            .ctx
            .hwpt_invalidate(
                self.accel.viommu_id,
                vfio_sys::iommufd::IOMMU_VIOMMU_INVALIDATE_DATA_ARM_SMMUV3,
                std::ptr::from_ref(&invalidate_entry) as u64,
                size_of::<vfio_sys::iommufd::IommuViommuArmSmmuv3Invalidate>() as u32,
                1,
            )
            .context("iommufd HWPT_INVALIDATE (TLBI) failed")?;

        Ok(())
    }
}

impl Drop for IommufdStreamBackend {
    fn drop(&mut self) {
        let state = self.state.get_mut();

        // Detach the device (best-effort).
        let _ = vfio_sys::cdev::detach_pt(self.device_fd.as_fd());

        // Destroy the nested HWPT (best-effort).
        if let Some(hwpt_id) = state.current_nested_hwpt.take() {
            let _ = self.accel.ctx.destroy(hwpt_id);
        }

        // Destroy the vDevice (best-effort).
        if let Some(vdev_id) = state.vdevice_id.take() {
            let _ = self.accel.ctx.destroy(vdev_id);
        }
    }
}
