// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Resource resolver for VFIO-assigned PCI devices.

use crate::VfioAssignedPciDevice;
use crate::manager::VfioContainerManager;
use crate::manager::VfioManagerClient;
use anyhow::Context as _;
use async_trait::async_trait;
use membacking::DmaMapperClient;
use pal_async::task::Spawn as _;
use parking_lot::RwLock;
use pci_core::bus_range::AssignedBusRange;
use pci_resources::ResolvePciDeviceHandleParams;
use pci_resources::ResolvedPciDevice;
use std::collections::HashMap;
use std::sync::Arc;
use vfio_assigned_device_resources::VfioCdevDeviceHandle;
use vfio_assigned_device_resources::VfioDeviceHandle;
use vm_resource::AsyncResolveResource;
use vm_resource::ResourceResolver;
use vm_resource::kind::PciDeviceHandleKind;

/// Resource resolver for [`VfioDeviceHandle`].
///
/// Spawns a `VfioContainerManager` task internally and communicates with it
/// via RPC to share VFIO containers across assigned devices.
pub struct VfioDeviceResolver {
    client: VfioManagerClient,
    _task: pal_async::task::Task<()>,
}

impl VfioDeviceResolver {
    /// Create a new resolver, spawning the container manager task.
    ///
    /// The manager registers each new VFIO container with the region manager
    /// so that DMA mappings are kept in sync with the VM's memory map.
    pub fn new(spawner: impl pal_async::task::Spawn, dma_mapper_client: DmaMapperClient) -> Self {
        let mut manager = VfioContainerManager::new(dma_mapper_client);
        let client = manager.client();
        let task = spawner.spawn("vfio-container-mgr", manager.run());
        Self {
            client,
            _task: task,
        }
    }

    /// Returns a handle that can be stored in the VM's inspect tree to
    /// expose the VFIO container/group topology.
    pub fn inspect_handle(&self) -> VfioManagerClient {
        self.client.clone()
    }
}

#[async_trait]
impl AsyncResolveResource<PciDeviceHandleKind, VfioDeviceHandle> for VfioDeviceResolver {
    type Output = ResolvedPciDevice;
    type Error = anyhow::Error;

    async fn resolve(
        &self,
        _resolver: &ResourceResolver,
        resource: VfioDeviceHandle,
        input: ResolvePciDeviceHandleParams<'_>,
    ) -> Result<Self::Output, Self::Error> {
        let VfioDeviceHandle {
            pci_id,
            group,
            bar_pt,
        } = resource;

        if input.software_iommu {
            anyhow::bail!(
                "VFIO device {pci_id} is behind a software IOMMU that cannot \
                 program the host IOMMU for passthrough DMA. Place the device \
                 on a root complex without a software IOMMU, or wait for \
                 iommufd nested translation support."
            );
        }

        tracing::info!(pci_id, "opening VFIO device");

        // Ask the container manager to prepare (or reuse) a container and
        // group for this device.
        let binding = self
            .client
            .prepare_device(pci_id.clone(), group)
            .await
            .context("VFIO container manager failed")?;

        let memory_mapper = input
            .shared_mem_mapper
            .context("memory mapper is required for VFIO device assignment")?;

        let device = VfioAssignedPciDevice::new(
            binding,
            pci_id,
            input.driver_source,
            input.register_mmio,
            input.msi_target,
            memory_mapper,
            bar_pt,
        )
        .await?;

        Ok(device.into())
    }
}

/// Resource resolver for [`VfioCdevDeviceHandle`] (cdev + iommufd path).
///
/// Spawns a `VfioCdevManager` task internally and communicates with it via RPC
/// to share IOAS contexts across devices referencing the same iommu ID.
///
/// When a nesting store is provided, devices whose `port_name` is registered
/// in the store get iommufd nested S1 translation: the resolver allocates the
/// S2 parent HWPT, creates the per-SMMU accel state and per-device stream
/// backend, and registers the backend with the SMMU shared state.
pub struct VfioCdevDeviceResolver {
    client: crate::manager::VfioCdevManagerClient,
    _task: pal_async::task::Task<()>,
    /// SMMU nesting context for each port behind an accel-capable SMMU.
    nesting_store: Arc<SmmuNestingStore>,
}

/// Shared store mapping port names to SMMU nesting context.
///
/// Populated by dispatch.rs after SMMU setup, before device resolution.
/// The resolver reads from this store during device resolution to determine
/// whether a VFIO cdev device should use iommufd nested translation, and to
/// find the emulated SMMU it must be wired into.
pub struct SmmuNestingStore {
    /// Per-port nesting entries.
    entries: RwLock<HashMap<String, SmmuNestingEntry>>,
}

/// Per-port SMMU nesting context.
struct SmmuNestingEntry {
    /// SMMU id (root complex index); identifies the emulated SMMU so the
    /// manager can share one vIOMMU across all of its ports.
    smmu_id: usize,
    /// SMMU shared state for the root complex this port belongs to.
    smmu_shared: Arc<smmu::SmmuSharedState>,
    /// The device's assigned bus range (shared with the PCIe port).
    bus_range: AssignedBusRange,
    /// Offset into the SMMU's stream table (0 for 1:1 SMMU-per-RC).
    stream_id_base: u32,
}

impl SmmuNestingStore {
    /// Create an empty nesting store.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            entries: RwLock::new(HashMap::new()),
        })
    }

    /// Register a port for SMMU nesting.
    ///
    /// `smmu_id` identifies the emulated SMMU (root complex index); all ports
    /// sharing it share a single vIOMMU (allocated by the manager).
    pub fn register(
        &self,
        port_name: String,
        smmu_id: usize,
        smmu_shared: Arc<smmu::SmmuSharedState>,
        bus_range: AssignedBusRange,
        stream_id_base: u32,
    ) {
        self.entries.write().insert(
            port_name,
            SmmuNestingEntry {
                smmu_id,
                smmu_shared,
                bus_range,
                stream_id_base,
            },
        );
    }

    /// Look up the nesting entry for a port.
    fn get(&self, port_name: &str) -> Option<SmmuNestingEntryRef> {
        let entries = self.entries.read();
        entries.get(port_name).map(|e| SmmuNestingEntryRef {
            smmu_id: e.smmu_id,
            smmu_shared: e.smmu_shared.clone(),
            bus_range: e.bus_range.clone(),
            stream_id_base: e.stream_id_base,
        })
    }
}

/// Cloned reference to a nesting entry (avoids holding the lock).
struct SmmuNestingEntryRef {
    smmu_id: usize,
    smmu_shared: Arc<smmu::SmmuSharedState>,
    bus_range: AssignedBusRange,
    stream_id_base: u32,
}

impl VfioCdevDeviceResolver {
    /// Create a new cdev resolver, spawning the cdev dispatcher task.
    pub fn new(
        spawner: impl pal_async::task::Spawn + 'static,
        dma_mapper_client: DmaMapperClient,
        nesting_store: Arc<SmmuNestingStore>,
    ) -> Self {
        // Arc the spawner so the dispatcher can spawn per-iommu manager tasks.
        let spawner: Arc<dyn pal_async::task::Spawn> = Arc::new(spawner);
        let mut manager = crate::manager::VfioCdevManager::new(spawner.clone(), dma_mapper_client);
        let client = manager.client();
        let task = spawner.spawn("vfio-cdev-dispatch", manager.run());
        Self {
            client,
            _task: task,
            nesting_store,
        }
    }

    /// Returns a handle for the VM's inspect tree.
    pub fn inspect_handle(&self) -> crate::manager::VfioCdevManagerClient {
        self.client.clone()
    }
}

#[async_trait]
impl AsyncResolveResource<PciDeviceHandleKind, VfioCdevDeviceHandle> for VfioCdevDeviceResolver {
    type Output = ResolvedPciDevice;
    type Error = anyhow::Error;

    async fn resolve(
        &self,
        _resolver: &ResourceResolver,
        resource: VfioCdevDeviceHandle,
        input: ResolvePciDeviceHandleParams<'_>,
    ) -> Result<Self::Output, Self::Error> {
        let VfioCdevDeviceHandle {
            pci_id,
            cdev,
            iommufd,
            iommu_id,
            bar_pt,
            port_name,
        } = resource;

        if input.software_iommu {
            anyhow::bail!(
                "VFIO device {pci_id} is behind a software IOMMU that cannot \
                 program the host IOMMU for passthrough DMA"
            );
        }

        // Check if this device is behind an accel-capable SMMU.
        let nesting_entry = self.nesting_store.get(&port_name);

        tracing::info!(
            pci_id,
            iommu_id,
            port_name,
            needs_nesting = nesting_entry.is_some(),
            "opening VFIO cdev device with iommufd"
        );

        let mut resp = self
            .client
            .prepare_device(crate::manager::CdevPrepareRequest {
                pci_id: pci_id.clone(),
                cdev,
                iommufd,
                iommu_id,
                smmu_id: nesting_entry.as_ref().map(|e| e.smmu_id),
            })
            .await
            .context("VFIO cdev manager failed")?;

        // If the device is nested, wire the manager's iommufd objects into
        // the emulated SMMU: finalize host-derived parameters and register
        // the per-device stream backend. The manager already created (or
        // reused) the shared vIOMMU and queried host capabilities.
        if let (Some(entry), Some(nesting)) = (nesting_entry, resp.nesting.take()) {
            // Finalize the vSMMU's host-derived parameters (OAS, ...) against
            // the physical SMMU backing this device. Runs once per vSMMU; a
            // later device on a different physical SMMU is rejected here.
            entry
                .smmu_shared
                .resolve_host_caps(nesting.host_caps)
                .with_context(|| format!("device {pci_id} is incompatible with the host SMMU"))?;

            let backend = Arc::new(crate::iommufd_nesting::IommufdStreamBackend::new(
                nesting.accel_state,
                resp.iommufd_devid,
                nesting.device_cdev_fd,
            ));

            entry.smmu_shared.register_accel_device(
                entry.bus_range.clone(),
                entry.stream_id_base,
                backend,
            );

            tracing::info!(
                pci_id,
                port_name,
                "registered iommufd nesting backend with SMMU"
            );
        }

        let cdev_binding = crate::manager::VfioCdevBinding::from_response(resp, pci_id.clone());

        let memory_mapper = input
            .shared_mem_mapper
            .context("memory mapper is required for VFIO device assignment")?;

        let device = VfioAssignedPciDevice::from_cdev(
            cdev_binding,
            pci_id,
            input.register_mmio,
            input.msi_target,
            memory_mapper,
            bar_pt,
        )
        .await?;

        Ok(device.into())
    }
}
