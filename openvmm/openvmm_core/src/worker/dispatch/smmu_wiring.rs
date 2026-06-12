// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![cfg(guest_arch = "aarch64")]

//! SMMU resource resolution and wiring helpers for aarch64 VMs.
//!
//! This module handles combining SMMU MMIO ranges (from the memory layout
//! allocator) with SPI assignments (from the SPI allocator) into resolved
//! resources and instantiating SMMU chipset devices.

use chipset_device_resources::IRQ_LINE_SET;
use guestmem::GuestMemory;
use std::collections::HashMap;
use std::sync::Arc;
use vm_topology::pcie::PcieHostBridge;
use vmotherboard::ChipsetBuilder;

/// Resolved resources for a single SMMUv3 instance, combining MMIO and SPI
/// allocations.
pub(super) struct ResolvedSmmuResources {
    /// MMIO base address (from the memory layout allocator).
    pub base: u64,
    /// GIC INTID for the event queue interrupt (from the SPI allocator).
    pub evtq_intid: u32,
    /// GIC INTID for the global error interrupt (from the SPI allocator).
    pub gerr_intid: u32,
}

/// Combines SMMU MMIO ranges from the memory layout with SPI assignments from
/// the SPI layout into resolved resources.
pub(super) fn resolve_smmu_resources(
    smmu_ranges: &[memory_range::MemoryRange],
    spi_layout: &crate::worker::spi_layout::ResolvedSpiLayout,
) -> Vec<ResolvedSmmuResources> {
    smmu_ranges
        .iter()
        .zip(&spi_layout.smmu)
        .map(|(range, spis)| ResolvedSmmuResources {
            base: range.start(),
            evtq_intid: spis.evtq_intid,
            gerr_intid: spis.gerr_intid,
        })
        .collect()
}

/// Result of [`setup_smmu`].
pub(super) struct SmmuDevicesResult {
    /// Per-RC SMMU shared state, indexed parallel to `pcie_host_bridges`.
    /// `None` for root complexes without an SMMU.
    pub shared_states: Vec<Option<Arc<smmu::SmmuSharedState>>>,
    /// ACPI IORT configuration for each SMMU instance.
    pub configs: Vec<vmm_core::acpi_builder::AcpiSmmuConfig>,
}

/// Instantiate SMMU chipset devices for root complexes that have SMMU
/// configured.
///
/// This is the single entry point for all SMMU setup in dispatch. It
/// iterates root complex configs, creates one `SmmuDevice` per RC with
/// `iommu: Some(Smmu)`, and wires up interrupts.
pub(super) fn setup_smmu(
    root_complexes: &[openvmm_defs::config::PcieRootComplexConfig],
    resolved_smmu_resources: &[ResolvedSmmuResources],
    pcie_rc_name_to_idx: &HashMap<String, usize>,
    pcie_host_bridges: &[PcieHostBridge],
    chipset_builder: &ChipsetBuilder<'_>,
    gm: &GuestMemory,
    gpa_top: u64,
) -> anyhow::Result<SmmuDevicesResult> {
    // Instantiate SMMU chipset devices.
    let mut shared_states: Vec<Option<Arc<smmu::SmmuSharedState>>> =
        vec![None; pcie_host_bridges.len()];
    let mut configs = Vec::new();

    // Iterate RCs with SMMU enabled, zipping with resolved MMIO+SPI resources.
    let smmu_rcs = root_complexes.iter().filter_map(|rc| match &rc.iommu {
        Some(openvmm_defs::config::PcieIommuConfig::Smmu { accel, oas }) => {
            Some((rc, *accel, *oas))
        }
        _ => None,
    });

    for (idx, (rc, accel, oas)) in smmu_rcs.enumerate() {
        let rc_pos = pcie_rc_name_to_idx[rc.name.as_str()];

        let smmu = &resolved_smmu_resources[idx];
        let evtq_irq_vector = smmu.evtq_intid - *vmm_core::emuplat::gic::SPI_RANGE.start();
        let gerror_irq_vector = smmu.gerr_intid - *vmm_core::emuplat::gic::SPI_RANGE.start();
        let device_name = format!("smmu:{}", rc.name);

        // The smallest valid OAS that covers the guest's DMA address space:
        // RAM/layout top plus this root complex's high MMIO top (for P2P and
        // MSI doorbell targets).
        let dma_top = gpa_top.max(pcie_host_bridges[rc_pos].high_mmio.end());
        let gpa_oas = smmu::min_oas_bits_for(dma_top);

        // Resolve the OAS policy into an initial advertised value. For
        // non-accel SMMUs this is final; for accel SMMUs it is a provisional
        // floor that is finalized against the host SMMU at device attach.
        let (oas_bits, oas_policy) = match oas {
            openvmm_defs::config::SmmuOas::Auto => (gpa_oas, smmu::SmmuOasPolicy::Auto),
            openvmm_defs::config::SmmuOas::Fixed(bits) => {
                if !smmu::VALID_OAS_BITS.contains(&bits) {
                    anyhow::bail!(
                        "--smmu rc={}: oas={bits} is not a valid SMMUv3 output address \
                         size (expected one of {:?})",
                        rc.name,
                        smmu::VALID_OAS_BITS
                    );
                }
                if !accel && bits < gpa_oas {
                    anyhow::bail!(
                        "--smmu rc={}: oas={bits} is too small for the guest address \
                         space (needs at least {gpa_oas} bits); use oas=auto",
                        rc.name
                    );
                }
                (bits, smmu::SmmuOasPolicy::Fixed(bits))
            }
        };

        let smmu_config = smmu::SmmuConfig {
            sidsize: 16,
            oas: oas_bits,
            oas_policy,
            accel,
        };
        let smmu_device =
            chipset_builder
                .arc_mutex_device(device_name.as_str())
                .add(|services| {
                    let evtq_irq = services.new_line(IRQ_LINE_SET, "evtq", evtq_irq_vector);
                    let gerror_irq = services.new_line(IRQ_LINE_SET, "gerror", gerror_irq_vector);
                    smmu::SmmuDevice::new(
                        smmu.base,
                        gm.clone(),
                        &smmu_config,
                        Some(evtq_irq),
                        Some(gerror_irq),
                    )
                })?;

        shared_states[rc_pos] = Some(smmu_device.lock().shared_state().clone());
        // When the SMMU is in accel mode (iommufd nested), the L1
        // kernel's MSI reserved IOVA window must be identity-mapped in
        // the L2 guest's S1 page tables. The window is 128MB–129MB
        // (0x800_0000–0x80F_FFFF), which is the default ARM IOMMU MSI
        // reserved region.
        let reserved_iova_ranges = if accel {
            vec![(0x800_0000, 0x80F_FFFF)]
        } else {
            Vec::new()
        };

        configs.push(vmm_core::acpi_builder::AcpiSmmuConfig {
            rc_index: pcie_host_bridges[rc_pos].index,
            segment: pcie_host_bridges[rc_pos].segment,
            base: smmu.base,
            event_gsiv: smmu.evtq_intid,
            gerr_gsiv: smmu.gerr_intid,
            reserved_iova_ranges,
        });
    }

    Ok(SmmuDevicesResult {
        shared_states,
        configs,
    })
}
