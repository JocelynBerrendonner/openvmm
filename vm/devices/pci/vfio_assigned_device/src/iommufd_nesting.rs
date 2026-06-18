// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! iommufd nested translation for VFIO devices behind an accel-capable SMMU.
//!
//! This module implements HW-accelerated nested stage 1 translation using
//! iommufd. The guest programs the emulated SMMU's stream table entries (STEs)
//! and page tables. The SMMU emulator decodes the guest's CMDQ commands and
//! dispatches a [`smmu::StreamConfig`] (and raw invalidation commands) to this
//! module via the [`smmu::AcceleratedStreamBackend`] trait, which programs the
//! host IOMMU hardware.
//!
//! # Architecture
//!
//! ```text
//! Guest programs emulated SMMU ──► CMDQ commands
//!        │
//!        ▼
//! SmmuDevice decodes STE/CMDQ and dispatches to AcceleratedStreamBackend
//!        │
//!        ▼
//! IommufdStreamBackend (per VFIO device)
//!   ├─ set_stream_config: map StreamConfig → allocate/switch nested HWPT
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
/// Issues a single `IOMMU_GET_HW_INFO` and hands the host's raw IDR registers
/// to [`smmu::HostSmmuCaps::from_idr`], which decodes the fields the vSMMU
/// finalizes against and validates compatibility with (OAS, TTF, TTENDIAN,
/// GRAN4K).
pub fn query_host_caps(ctx: &IommufdCtx, dev_id: u32) -> anyhow::Result<smmu::HostSmmuCaps> {
    let mut info = vfio_sys::iommufd::IommuHwInfoArmSmmuv3 {
        flags: 0,
        __reserved: 0,
        idr: [0; 6],
        iidr: 0,
        aidr: 0,
    };
    let (data_type, _caps) = ctx
        .get_hw_info(dev_id, &mut info)
        .context("IOMMU_GET_HW_INFO failed")?;
    if data_type != vfio_sys::iommufd::IOMMU_HW_INFO_TYPE_ARM_SMMUV3 {
        anyhow::bail!("unexpected host IOMMU hw info type {data_type} (expected ARM SMMUv3)");
    }
    Ok(smmu::HostSmmuCaps::from_idr(info.idr))
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
    /// Shared ABORT HWPT ID (nested HWPT under the vIOMMU with an abort STE).
    ///
    /// A nested HWPT binds to the vIOMMU, not to any single device, so one
    /// abort HWPT is reusable by every device behind this vSMMU. Attaching to
    /// it terminates all DMA (fail-closed), and the kernel validates the
    /// device's physical SMMU against the vIOMMU's during attach — so it
    /// doubles as a fail-closed probe target that never momentarily opens
    /// BYPASS (which would expose guest memory to a misbehaving device).
    abort_hwpt_id: u32,
}

impl SmmuAccelState {
    /// Create per-SMMU iommufd objects.
    ///
    /// `dev_id` is the first device bound to this vSMMU. The S2 parent HWPT
    /// and the vIOMMU are both allocated against this device, which binds
    /// them to the *physical* SMMU that backs it. The iommufd kernel requires
    /// `s2_parent->smmu == device->smmu` for `IOMMU_VIOMMU_ALLOC`, and every
    /// device later attached to this vSMMU's S2 parent must be behind the same
    /// physical SMMU. The S2 parent is allocated per vSMMU (not per IOAS) so
    /// that distinct vSMMUs — each backed by a different physical SMMU — can
    /// share a single IOAS (one GPA→HPA identity map) while each owns the
    /// nest-parent HWPT for its own physical SMMU.
    ///
    /// `ioas_id` is the IOAS that provides the GPA→HPA identity map; the S2
    /// parent HWPT is allocated against it with `NEST_PARENT`.
    pub fn new(ctx: Arc<IommufdCtx>, dev_id: u32, ioas_id: u32) -> anyhow::Result<Self> {
        let s2_parent_hwpt_id = ctx
            .hwpt_alloc::<vfio_sys::iommufd::IommuHwptArmSmmuv3>(
                vfio_sys::iommufd::IOMMU_HWPT_ALLOC_NEST_PARENT,
                dev_id,
                ioas_id,
                vfio_sys::iommufd::IOMMU_HWPT_DATA_NONE,
                None,
            )
            .context("failed to allocate S2 parent HWPT for accel SMMU")?;

        let viommu_id = ctx
            .viommu_alloc(
                vfio_sys::iommufd::IOMMU_VIOMMU_TYPE_ARM_SMMUV3,
                dev_id,
                s2_parent_hwpt_id,
            )
            .context("failed to allocate vIOMMU for accel SMMU")?;

        // Allocate a shared abort HWPT under the vIOMMU. The STE is
        // `[V=1, Config=ABORT]` (`[0x1, 0x0]`), which terminates all
        // transactions; the kernel's `arm_smmu_validate_vste` accepts this
        // encoding. Used as the fail-closed probe target below.
        let abort_ste = vfio_sys::iommufd::IommuHwptArmSmmuv3 { ste: [0x1, 0x0] };
        let abort_hwpt_id = ctx
            .hwpt_alloc(
                0, // flags: nested child, not a nest parent
                dev_id,
                viommu_id, // parent is the vIOMMU
                vfio_sys::iommufd::IOMMU_HWPT_DATA_ARM_SMMUV3,
                Some(&abort_ste),
            )
            .context("failed to allocate abort HWPT for accel SMMU")?;

        tracing::info!(
            viommu_id,
            s2_parent_hwpt_id,
            abort_hwpt_id,
            ioas_id,
            "created SMMU accel state (S2 parent HWPT + vIOMMU + abort HWPT)"
        );

        Ok(Self {
            ctx,
            viommu_id,
            s2_parent_hwpt_id,
            abort_hwpt_id,
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

    /// Returns the shared abort HWPT ID (fail-closed; used for the attach
    /// validation probe).
    pub fn abort_hwpt_id(&self) -> u32 {
        self.abort_hwpt_id
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
/// | ABORT (0)  | Attach to shared abort HWPT — DMA faults |
/// | BYPASS (4) | Attach to S2 parent HWPT — identity GPA→HPA |
/// | S1_TRANS (5) | Allocate nested HWPT with STE DW0-1, attach |
///
/// Each transition is a single `VFIO_DEVICE_ATTACH_IOMMUFD_PT`: the kernel
/// replaces the device's current attachment atomically (no separate detach),
/// so the device is never momentarily unattached while switching.
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
    /// Current per-device nested HWPT ID, if S1 translation is active.
    /// `None` when in ABORT (shared abort HWPT) or BYPASS (shared S2 parent) —
    /// those HWPTs are owned by [`SmmuAccelState`] and are not destroyed on
    /// switch; only a per-device nested HWPT tracked here is.
    current_nested_hwpt: Option<u32>,
    /// vDevice ID, lazily allocated on first `CFGI_STE` with `S1_TRANS`.
    vdevice_id: Option<u32>,
}

impl IommufdStreamBackend {
    /// Create a new stream backend.
    ///
    /// `device_fd` must be a dup'd VFIO cdev fd (still bound to iommufd).
    ///
    /// The device is attached to the shared abort HWPT (fail-closed) by the
    /// validation probe at bind time; this is the backend's initial state
    /// (`current_nested_hwpt: None`). The SMMU emulator drives the device to
    /// its correct initial policy when it registers the backend (see
    /// `SmmuSharedState::register_accel_device`): bypass (attach to the S2
    /// parent HWPT) while the SMMU is disabled with `GBPA.ABORT=0`, or abort
    /// otherwise.
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

    /// Handle STE Config=ABORT: attach to the shared abort HWPT.
    fn handle_abort(&self, state: &mut StreamBackendState) -> anyhow::Result<()> {
        // Attach to the shared abort HWPT. The abort STE terminates all DMA,
        // so this is the fail-closed blocked state — equivalent to detaching,
        // but it keeps the device attached to a known object and the attach
        // atomically replaces any current attachment (no unattached window).
        vfio_sys::cdev::attach_pt(self.device_fd.as_fd(), self.accel.abort_hwpt_id)
            .context("failed to attach device to abort HWPT for ABORT")?;

        // Destroy the old per-device nested HWPT (if any). The replace above
        // already moved the device off it, so it has no attached devices.
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

        tracing::debug!(dev_id = self.dev_id, "SMMU accel: STE → ABORT (abort HWPT)");
        Ok(())
    }

    /// Handle STE Config=BYPASS: attach to S2 parent HWPT.
    fn handle_bypass(&self, state: &mut StreamBackendState) -> anyhow::Result<()> {
        // Attach to the S2 parent HWPT (identity GPA→HPA). The attach
        // atomically replaces any current attachment (a previous nested HWPT),
        // so no separate detach is needed.
        vfio_sys::cdev::attach_pt(self.device_fd.as_fd(), self.accel.s2_parent_hwpt_id)
            .context("failed to attach device to S2 parent HWPT for BYPASS")?;

        // Destroy the old per-device nested HWPT (if any).
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
        nested_ste: [u64; 2],
        stream_id: u32,
    ) -> anyhow::Result<()> {
        // Lazy vDevice allocation — the virtual stream ID is the guest-assigned
        // BDF from the CFGI_STE command's SID, not known at construction time.
        if state.vdevice_id.is_none() {
            let vdev_id = self
                .accel
                .ctx
                .vdevice_alloc(self.accel.viommu_id, self.dev_id, stream_id as u64)
                .with_context(|| {
                    format!(
                        "failed to allocate vDevice for dev_id={}, vsid={}",
                        self.dev_id, stream_id
                    )
                })?;
            tracing::info!(
                dev_id = self.dev_id,
                vdevice_id = vdev_id,
                virtual_sid = stream_id,
                "allocated iommufd vDevice"
            );
            state.vdevice_id = Some(vdev_id);
        }

        // The STE the kernel reads to program nested stage-1 translation.
        // `nested_ste` carries only the stage-1 fields (the emulator stripped
        // everything else): the kernel's arm-smmu-v3 nesting path validates
        // the STE and rejects (`-EIO`) any reserved or stage-2/override bits
        // it doesn't expect, so they must already be cleared here.
        let ste_data = vfio_sys::iommufd::IommuHwptArmSmmuv3 { ste: nested_ste };

        tracing::info!(
            dev_id = self.dev_id,
            ste_dw0 = format_args!("{:#018x}", nested_ste[0]),
            ste_dw1 = format_args!("{:#018x}", nested_ste[1]),
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
                Some(&ste_data),
            )
            .context("failed to allocate nested HWPT for S1_TRANS")?;

        // Attach to the new nested HWPT. The attach atomically replaces the
        // current attachment (the S2 parent from BYPASS, the abort HWPT, or an
        // old nested HWPT), so no separate detach is needed.
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
    fn set_stream_config(&self, config: smmu::StreamConfig) -> anyhow::Result<()> {
        let mut state = self.state.lock();
        match config {
            smmu::StreamConfig::Abort => self.handle_abort(&mut state),
            smmu::StreamConfig::Bypass => self.handle_bypass(&mut state),
            smmu::StreamConfig::Translate { sid, nested_ste } => {
                self.handle_s1_translate(&mut state, nested_ste, sid)
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
                std::slice::from_ref(&invalidate_entry),
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
