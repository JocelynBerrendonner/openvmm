// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Shared SMMU state and per-device translation wrappers.
//!
//! [`SmmuSharedState`] holds the SMMU configuration that per-device wrappers
//! need for translation: stream table base, CR0 state, and a reference to
//! guest memory for walking page tables.
//!
//! [`SmmuTranslator`] implements
//! [`IommuTranslator`](iommu_common::IommuTranslator), translating IOVAs to
//! GPAs via the SMMU page tables. The generic
//! [`TranslatingMemory`](iommu_common::TranslatingMemory) in `iommu_common`
//! provides the [`GuestMemoryAccess`] boilerplate.
//!
//! [`SmmuSignalMsi`] implements [`SignalMsi`], translating the MSI address
//! (which may be an IOVA) to a GPA before forwarding to the inner MSI
//! target.
//!
//! [`SmmuIrqFd`] implements [`IrqFd`](vmcore::irqfd::IrqFd), producing
//! [`SmmuIrqFdRoute`] instances that translate the MSI address on
//! [`enable`](vmcore::irqfd::IrqFdRoute::enable) before forwarding to the
//! inner irqfd route.

use crate::spec::events::EvtEntry;
use crate::spec::registers;
use crate::translate;
use guestmem::GuestMemory;
use pal_event::Event;
use parking_lot::Mutex;
use parking_lot::RwLock;
use pci_core::bus_range::AssignedBusRange;
use pci_core::msi::SignalMsi;
use std::sync::Arc;
use vmcore::irqfd::IrqFd;
use vmcore::irqfd::IrqFdRoute;
use vmcore::line_interrupt::LineInterrupt;
use zerocopy::IntoBytes;

/// Backend for a single VFIO device's stream, bridging SMMU CMDQ commands
/// to iommufd nested HWPT operations.
///
/// The SMMU emulator dispatches CMDQ commands to registered backends on a
/// per-stream-ID basis. Streams without a registered backend use the
/// software page table walk path (emulated devices). Streams with a backend
/// use hardware-accelerated translation via iommufd.
///
/// The SMMU emulator owns the SMMUv3 spec: it parses and validates the guest
/// STE and dispatches a decoded [`StreamConfig`] to the backend, which only
/// maps each variant onto host IOMMU operations. TLBI commands are forwarded
/// as raw bytes because the host kernel — not the VMM — parses them.
pub trait AcceleratedStreamBackend: Send + Sync {
    /// The guest reconfigured this stream's STE (via `CFGI_STE`), or the
    /// emulator recomputed the stream's policy (e.g. on a `GBPA` write or
    /// `SMMUEN` transition). The emulator has already parsed and validated
    /// the STE into `config`. Only [`StreamConfig::Translate`] carries a
    /// stream ID (for lazy vDevice allocation); the bypass and abort cases
    /// have no per-stream identity to act on.
    fn set_stream_config(&self, config: StreamConfig) -> anyhow::Result<()>;

    /// Guest issued a TLBI command. The raw 16-byte command entry is
    /// forwarded to iommufd via `IOMMU_HWPT_INVALIDATE`. The kernel
    /// parses the opcode and operands.
    fn on_tlbi(&self, cmd_bytes: &[u8; 16]) -> anyhow::Result<()>;
}

/// A decoded stream (STE) configuration the SMMU emulator dispatches to an
/// [`AcceleratedStreamBackend`].
///
/// The emulator decodes the guest's STE (validity and `STE.Config`) into one
/// of these variants so the backend never has to interpret raw STE bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamConfig {
    /// Abort all transactions. Produced for an invalid STE (`V=0`),
    /// `Config=ABORT`, or any config the emulator does not support in
    /// accelerated mode.
    Abort,
    /// Bypass translation (`Config=BYPASS`) — identity GPA→HPA via the
    /// nesting parent (S2) HWPT.
    Bypass,
    /// Stage-1 translation (`Config=S1_TRANS`). Carries the stream ID (for
    /// lazy vDevice allocation) and the STE double-words reduced to the
    /// fields the host nesting path accepts.
    Translate {
        /// Stream ID this configuration applies to. Used by the backend to
        /// allocate the iommufd vDevice (the virtual stream ID is not known
        /// at backend construction time).
        sid: u32,
        /// Masked `[DW0, DW1]` ready to hand to `IOMMU_HWPT_ALLOC`.
        nested_ste: [u64; 2],
    },
}

impl StreamConfig {
    /// Decode a guest STE for stream `sid` into the accelerated-stream action.
    ///
    /// Centralizes the SMMUv3 spec decisions (V-bit handling, `Config`
    /// dispatch, the nesting field selection) so backends consume only the
    /// resulting intent.
    pub(crate) fn from_ste(sid: u32, ste: &crate::spec::ste::Ste) -> Self {
        use crate::spec::ste::SteConfig;

        if !ste.valid() {
            return Self::Abort;
        }
        match ste.config() {
            SteConfig::ABORT => Self::Abort,
            SteConfig::BYPASS => Self::Bypass,
            SteConfig::S1_TRANS => Self::Translate {
                sid,
                nested_ste: ste.nesting_dwords(),
            },
            other => {
                tracelimit::warn_ratelimited!(
                    config = other.0,
                    "smmu: unsupported STE config for accelerated stream; treating as abort"
                );
                Self::Abort
            }
        }
    }
}

/// Registration entry for a VFIO device with iommufd-accelerated translation.
///
/// The SID is derived dynamically from the `bus_range` (which holds the
/// guest-assigned bus number) rather than being fixed at registration time,
/// because PCIe bus numbers are assigned by the guest during enumeration.
struct AccelDeviceRegistration {
    /// The device's assigned bus range (shared with the PCIe port).
    bus_range: AssignedBusRange,
    /// Offset into this SMMU's stream table for the device's root complex.
    stream_id_base: u32,
    /// The iommufd-backed stream handler.
    backend: Arc<dyn AcceleratedStreamBackend>,
}

/// Composes an SMMU-local stream ID from a bus range, a base offset,
/// and an optional per-device BDF.
///
/// The stream ID is `stream_id_base + (bdf & 0xFFFF)`. When `devid`
/// is `None`, the default BDF `(secondary_bus, dev 0, fn 0)` is used.
///
/// Returns `None` if the secondary bus has not been assigned yet
/// (still 0) or if the BDF's bus number falls outside the port's
/// assigned range.
fn compose_stream_id(
    bus_range: &AssignedBusRange,
    stream_id_base: u32,
    devid: Option<u32>,
) -> Option<u32> {
    let (secondary, subordinate) = bus_range.bus_range();
    if secondary == 0 {
        return None;
    }
    let bdf = devid.unwrap_or((secondary as u32) << 8);
    let bus = (bdf >> 8) as u8;
    if bus < secondary || bus > subordinate {
        tracelimit::warn_ratelimited!(bus, secondary, subordinate, "BDF out of port bus range");
        return None;
    }
    Some(stream_id_base + (bdf & 0xFFFF))
}

/// Result of an SMMU translation attempt.
#[derive(Debug)]
enum TranslateResult {
    /// SMMU disabled (with `GBPA.ABORT=0`) or bus not yet assigned — bypass
    /// (IOVA = GPA).
    Bypass,
    /// Translated GPA.
    Translated(u64),
    /// Global abort: the SMMU is disabled with `GBPA.ABORT=1`. Per SMMUv3,
    /// the transaction is terminated with an abort and **no** event record is
    /// generated (there is no stream context to fault against). Distinct from
    /// [`TranslateResult::Abort`], which is STE-driven and records an event.
    GlobalAbort,
    /// Abort — STE says to abort this stream's DMA. Records a `C_BAD_STE`
    /// event.
    Abort(EvtEntry),
    /// Translation fault — event to queue.
    Fault(EvtEntry),
}

/// Shared SMMU state accessed by per-device translation wrappers.
///
/// The SMMU device updates this state on register writes; per-device wrappers
/// read it during translation. The `RwLock` allows concurrent translations
/// (read path) while register writes (write path) are exclusive.
///
/// Queue and error state is behind a separate `Mutex` so that per-device
/// wrappers can write fault events and signal overflow without going through
/// the emulator.
pub struct SmmuSharedState {
    /// Translation configuration — RwLock for concurrent DMA reads.
    inner: RwLock<SharedStateInner>,
    /// Guest memory for reading page tables and stream table entries.
    guest_memory: GuestMemory,
    /// Event queue and global error state — single mutex covers both
    /// because the EVTQ overflow path needs to update GERROR atomically.
    queue_state: Mutex<QueueErrorState>,
    /// Wired SPI interrupt line for event queue signaling.
    evtq_irq: Option<LineInterrupt>,
    /// Wired SPI interrupt line for global error signaling.
    gerror_irq: Option<LineInterrupt>,
    /// Whether this SMMU is in accelerated mode (iommufd nested).
    ///
    /// When `true`, VFIO cdev devices behind this SMMU use hardware-
    /// accelerated S1 translation. When `false`, all devices use the
    /// software page table walk path.
    accel: bool,
    /// How the advertised OAS is resolved (see [`set_oas`](Self::set_oas)).
    oas_policy: crate::SmmuOasPolicy,
    /// Per-device accelerated backends (VFIO devices with iommufd nested).
    ///
    /// Devices not in this list use the software page table walk path.
    /// The SID is derived dynamically from each entry's `AssignedBusRange`
    /// because bus numbers are guest-assigned after device construction.
    accel_devices: RwLock<Vec<AccelDeviceRegistration>>,
    /// Serializes "compute current policy + apply to backend" for accelerated
    /// streams. Both device registration (resolver/manager thread) and
    /// CMDQ-driven re-config (vCPU thread) acquire this lock, so the two are
    /// totally ordered and the last-applied stream config always reflects the
    /// newest guest intent. Held across the backend ioctls (attach path, not
    /// the DMA hot path); never nested inside the translation `inner` lock.
    policy_lock: Mutex<()>,
}

struct SharedStateInner {
    /// Whether the SMMU is enabled (CR0.SMMUEN).
    enabled: bool,
    /// Mirror of `GBPA.ABORT`, kept in sync on GBPA writes. Selects the
    /// disabled-state policy: when the SMMU is disabled, `true` aborts all
    /// transactions and `false` bypasses (IOVA = GPA). Consulted by both the
    /// non-accel translate path and the accel policy computation.
    gbpa_abort: bool,
    /// Stream table base address.
    strtab_base: u64,
    /// Stream table log2 size (number of entries).
    strtab_log2size: u8,
    /// Advertised output address size in bits. Reflected in IDR5.OAS and
    /// used to derive `oas_mask`.
    oas_bits: u8,
    /// Host SMMU capabilities, once an accelerated VFIO device has bound and
    /// [`SmmuSharedState::resolve_host_caps`] has finalized the host-derived
    /// parameters. `None` until then (and always `None` for non-accel SMMUs).
    /// A second device reporting different host caps is rejected — a single
    /// vSMMU cannot be backed by two physical SMMUs.
    resolved_host_caps: Option<crate::HostSmmuCaps>,
    /// Output address mask: `(1 << oas_bits) - 1`. Computed addresses for
    /// STE/CD/PT fetches are masked with this per SMMUv3 §3.4.
    oas_mask: u64,
}

/// Event queue and global error state.
///
/// A single mutex serializes event writes from concurrent DMA fault
/// paths, GERROR updates from both the emulator and DMA overflow,
/// and interrupt line level changes.
struct QueueErrorState {
    // -- Event queue --
    /// EVTQ base GPA (parsed from EVTQ_BASE register).
    evtq_base_addr: u64,
    /// EVTQ log2 size (clamped to IDR1.EVENTQS).
    evtq_log2size: u8,
    /// Whether the event queue is enabled (CR0.EVENTQEN).
    evtq_enabled: bool,
    /// Whether the EVTQ interrupt is enabled (IRQ_CTRL.EVENTQ_IRQEN).
    evtq_irqen: bool,
    /// Producer index (advanced by the SMMU when writing events).
    evtq_prod: u32,
    /// Consumer index (advanced by the guest via MMIO).
    evtq_cons: u32,

    // -- Global error registers (toggle protocol) --
    /// GERROR register — individual error bits toggled by the SMMU.
    gerror: registers::Gerror,
    /// GERRORN register — written by the guest to acknowledge errors.
    gerrorn: registers::Gerror,
    /// Whether the GERROR interrupt is enabled (IRQ_CTRL.GERROR_IRQEN).
    gerror_irqen: bool,
}

/// Saved portion of [`QueueErrorState`] for state save/restore.
///
/// Only the producer/consumer indices and error toggle registers need
/// saving — the remaining fields (`evtq_base_addr`, `evtq_log2size`,
/// `evtq_enabled`, `evtq_irqen`, `gerror_irqen`) are derived from
/// SMMU register state and re-synced on restore.
pub(crate) struct SavedQueueState {
    pub evtq_prod: u32,
    pub evtq_cons: u32,
    pub gerror: u32,
    pub gerrorn: u32,
}

impl SmmuSharedState {
    /// Creates a new shared state with the SMMU disabled.
    ///
    /// `oas_bits` is the initial output address size in bits (e.g., 40 for a
    /// 40-bit physical address space). Computed addresses for STE/CD/PT
    /// fetches are truncated to this width, matching hardware behavior per
    /// SMMUv3 §3.4. `oas_policy` controls whether the value is finalized
    /// against the host SMMU at device-attach time (see
    /// [`Self::resolve_host_caps`]).
    pub fn new(
        guest_memory: GuestMemory,
        oas_bits: u8,
        oas_policy: crate::SmmuOasPolicy,
        accel: bool,
        evtq_irq: Option<LineInterrupt>,
        gerror_irq: Option<LineInterrupt>,
    ) -> Arc<Self> {
        let oas_mask = (1u64 << oas_bits) - 1;
        Arc::new(Self {
            inner: RwLock::new(SharedStateInner {
                enabled: false,
                gbpa_abort: false,
                strtab_base: 0,
                strtab_log2size: 0,
                oas_bits,
                resolved_host_caps: None,
                oas_mask,
            }),
            guest_memory,
            queue_state: Mutex::new(QueueErrorState {
                evtq_base_addr: 0,
                evtq_log2size: 0,
                evtq_enabled: false,
                evtq_irqen: false,
                evtq_prod: 0,
                evtq_cons: 0,
                gerror: registers::Gerror::new(),
                gerrorn: registers::Gerror::new(),
                gerror_irqen: false,
            }),
            evtq_irq,
            gerror_irq,
            accel,
            oas_policy,
            accel_devices: RwLock::new(Vec::new()),
            policy_lock: Mutex::new(()),
        })
    }

    /// Returns whether this SMMU is in accelerated mode (iommufd nested).
    pub fn is_accel(&self) -> bool {
        self.accel
    }

    /// Returns the currently advertised output address size in bits.
    pub fn oas_bits(&self) -> u8 {
        self.inner.read().oas_bits
    }

    /// Finalizes the host-derived vSMMU parameters against the physical SMMU
    /// backing an accelerated device, and validates host/guest compatibility.
    ///
    /// Called when an accelerated VFIO device binds to iommufd, at which
    /// point the backing physical SMMU is first known. Runs once per vSMMU:
    /// the first device validates compatibility (TTF, TTENDIAN, GRAN4K) and
    /// applies every host-derived parameter according to its configured
    /// policy (currently OAS — `auto` adopts the host value; `fixed` is
    /// validated as an upper bound). Subsequent devices must report identical
    /// host caps; a mismatch is rejected, since a single vSMMU cannot be
    /// backed by two different physical SMMUs.
    ///
    /// The compatibility checks cover only the features this emulator
    /// actually advertises that the host hardware must honor when walking the
    /// guest's page tables. Features the emulator does not advertise
    /// (SSIDSIZE, ATS, RIL, 16K/64K granules, 2-level stream tables) are
    /// intentionally not checked — see the TODOs at the IDR advertisement in
    /// `emulator.rs`. The host stream-ID size (IDR1.SIDSIZE) and stream-table
    /// format (IDR0.ST_LEVEL) are deliberately *not* validated: in the nested
    /// path the host never indexes or walks the guest's stream table (the VMM
    /// emulates it and registers each guest StreamID individually via
    /// `IOMMU_VDEVICE_ALLOC`), so the host and guest stream-table parameters
    /// are independent.
    pub fn resolve_host_caps(&self, caps: crate::HostSmmuCaps) -> anyhow::Result<()> {
        let mut inner = self.inner.write();

        if let Some(existing) = inner.resolved_host_caps {
            if existing != caps {
                anyhow::bail!(
                    "SMMU already bound to a physical SMMU ({existing:?}), but another \
                     device reports different host capabilities ({caps:?}); a single \
                     vSMMU cannot be backed by two physical SMMUs"
                );
            }
            return Ok(());
        }

        // TTF: the emulator builds AArch64 S1 page tables, so the host must be
        // able to walk them. TTF is a bitfield, not an ordered value — test
        // the AArch64 bit rather than comparing.
        if !caps.ttf.aarch64() {
            anyhow::bail!(
                "host SMMU does not support AArch64 translation tables \
                 (IDR0.TTF={:#05b})",
                u8::from(caps.ttf)
            );
        }

        // TTENDIAN: the emulator uses little-endian table walks. The encoding
        // is a set of distinct configurations, not an ordered range — test
        // membership rather than comparing.
        if !matches!(
            caps.ttendian,
            registers::Idr0TtEndian::MIXED | registers::Idr0TtEndian::LE
        ) {
            anyhow::bail!(
                "host SMMU does not support little-endian translation tables \
                 (IDR0.TTENDIAN={:#04b})",
                caps.ttendian.0
            );
        }

        // GRAN4K: the guest builds 4KB S1 page tables, so the host hardware
        // must support the 4KB granule.
        if !caps.gran4k {
            anyhow::bail!("host SMMU does not support the 4KB translation granule (IDR5.GRAN4K=0)");
        }

        // OAS: decode the host's IDR5.OAS encoding (may be a reserved value),
        // then `auto` adopts the host value while `fixed` must not exceed it.
        let host_oas_bits = caps.oas.bits().ok_or_else(|| {
            anyhow::anyhow!(
                "host SMMU reported an unknown OAS encoding ({})",
                caps.oas.0
            )
        })?;
        match self.oas_policy {
            crate::SmmuOasPolicy::Auto => {
                inner.oas_bits = host_oas_bits;
                inner.oas_mask = (1u64 << host_oas_bits) - 1;
            }
            crate::SmmuOasPolicy::Fixed(oas) => {
                if oas > host_oas_bits {
                    anyhow::bail!(
                        "configured SMMU oas={oas} exceeds host SMMU OAS {host_oas_bits}; \
                         lower the configured OAS or use oas=auto"
                    );
                }
            }
        }

        inner.resolved_host_caps = Some(caps);
        Ok(())
    }

    /// Updates the SMMU enable state (called by SmmuDevice on CR0 writes) and
    /// atomically re-drives accelerated backends to the new policy.
    ///
    /// The state write and the re-drive happen under a single `policy_lock`
    /// acquisition, so the transition is atomic with respect to device
    /// registration and other policy changes: a backend can never observe a
    /// half-updated view and apply a stale policy that then "wins".
    pub(crate) fn set_enabled(&self, enabled: bool) {
        let _policy = self.policy_lock.lock();
        self.inner.write().enabled = enabled;
        self.apply_all_locked();
    }

    /// Updates the mirrored `GBPA.ABORT` state (called by SmmuDevice on GBPA
    /// writes) and atomically re-drives accelerated backends to the new
    /// policy. Selects the disabled-state policy (abort vs bypass).
    ///
    /// Like [`set_enabled`](Self::set_enabled), the write and the re-drive are
    /// a single `policy_lock` critical section.
    pub(crate) fn set_gbpa_abort(&self, abort: bool) {
        let _policy = self.policy_lock.lock();
        self.inner.write().gbpa_abort = abort;
        self.apply_all_locked();
    }

    /// Atomically replaces all policy-relevant translation state (enable,
    /// `GBPA.ABORT`, stream table base/size) and re-drives accelerated
    /// backends to the resulting policy, in a single `policy_lock` critical
    /// section.
    ///
    /// Used on device reset and state restore, where several policy inputs
    /// change together: applying them as one atomic transition (rather than a
    /// sequence of single-field updates) avoids transient intermediate
    /// policies and any ordering fragility around when the final re-drive
    /// observes fully-consistent state.
    pub(crate) fn sync_translation_state(
        &self,
        enabled: bool,
        gbpa_abort: bool,
        strtab_base: u64,
        strtab_log2size: u8,
    ) {
        let _policy = self.policy_lock.lock();
        {
            let mut inner = self.inner.write();
            inner.enabled = enabled;
            inner.gbpa_abort = gbpa_abort;
            inner.strtab_base = strtab_base;
            inner.strtab_log2size = strtab_log2size;
        }
        self.apply_all_locked();
    }

    /// Updates the stream table configuration (called by SmmuDevice on
    /// STRTAB_BASE / STRTAB_BASE_CFG writes).
    pub fn set_strtab(&self, base: u64, log2size: u8) {
        let mut inner = self.inner.write();
        inner.strtab_base = base;
        inner.strtab_log2size = log2size;
    }

    /// Updates the event queue configuration (called by SmmuDevice on
    /// EVTQ_BASE writes).
    pub fn set_evtq_config(&self, base_addr: u64, log2size: u8) {
        let mut qs = self.queue_state.lock();
        qs.evtq_base_addr = base_addr;
        qs.evtq_log2size = log2size;
    }

    /// Updates the event queue enabled state (called on CR0 writes).
    pub fn set_evtq_enabled(&self, enabled: bool) {
        self.queue_state.lock().evtq_enabled = enabled;
    }

    /// Updates both interrupt enable flags from IRQ_CTRL (called on
    /// IRQ_CTRL writes). Also updates the GERROR interrupt line level.
    pub fn set_irq_ctrl(&self, evtq_irqen: bool, gerror_irqen: bool) {
        let mut qs = self.queue_state.lock();
        qs.evtq_irqen = evtq_irqen;
        qs.gerror_irqen = gerror_irqen;
        self.update_gerror_irq(&qs);
    }

    /// Reads the current GERROR register value.
    pub fn read_gerror(&self) -> registers::Gerror {
        self.queue_state.lock().gerror
    }

    /// Reads the current GERRORN register value.
    pub fn read_gerrorn(&self) -> registers::Gerror {
        self.queue_state.lock().gerrorn
    }

    /// Returns true if GERROR.CMDQ_ERR != GERRORN.CMDQ_ERR (error active).
    pub fn cmdq_err_active(&self) -> bool {
        let qs = self.queue_state.lock();
        qs.gerror.cmdq_err() != qs.gerrorn.cmdq_err()
    }

    /// Writes GERRORN (guest acknowledging errors) and updates the
    /// interrupt line level.
    pub fn write_gerrorn(&self, value: u32) {
        let mut qs = self.queue_state.lock();
        qs.gerrorn = registers::Gerror::from(value);
        self.update_gerror_irq(&qs);
    }

    /// Toggles GERROR.CMDQ_ERR to signal a command queue error.
    ///
    /// Updates the interrupt line level under the lock.
    pub fn toggle_cmdq_err(&self) {
        let mut qs = self.queue_state.lock();
        let new_val = !qs.gerror.cmdq_err();
        qs.gerror.set_cmdq_err(new_val);
        self.update_gerror_irq(&qs);
    }

    /// Signals an EVTQ overflow by making GERROR.EVTQ_ABT_ERR active.
    ///
    /// Per spec, sets the bit to the inverse of GERRORN.EVTQ_ABT_ERR.
    /// If the error is already active this is a no-op (the bit value
    /// doesn't change). Called from `write_event` under the same lock.
    fn signal_evtq_overflow(&self, qs: &mut QueueErrorState) {
        let new_val = !qs.gerrorn.eventq_abt_err();
        qs.gerror.set_eventq_abt_err(new_val);
        self.update_gerror_irq(qs);
    }

    /// Updates the GERROR wired interrupt line level based on current state.
    ///
    /// Must be called with the queue_state lock held. The line is held
    /// high while any error is active (GERROR != GERRORN) and deasserted
    /// when all errors are acknowledged.
    fn update_gerror_irq(&self, qs: &QueueErrorState) {
        if let Some(irq) = &self.gerror_irq {
            let active = qs.gerror_irqen && u32::from(qs.gerror) != u32::from(qs.gerrorn);
            irq.set_level(active);
        }
    }

    /// Updates the event queue consumer index (called when the guest
    /// writes EVENTQ_CONS on page 1).
    ///
    /// Deasserts the EVTQ wired interrupt if the queue is now empty.
    pub fn set_evtq_cons(&self, cons: u32) {
        let mut qs = self.queue_state.lock();
        qs.evtq_cons = cons;
        // Deassert EVTQ IRQ when the guest has drained all events.
        if qs.evtq_irqen && qs.evtq_prod == qs.evtq_cons {
            if let Some(irq) = &self.evtq_irq {
                irq.set_level(false);
            }
        }
    }

    /// Returns the current event queue producer index (for guest reads
    /// of EVENTQ_PROD on page 1).
    pub fn evtq_prod(&self) -> u32 {
        self.queue_state.lock().evtq_prod
    }

    /// Returns the current event queue consumer index (for guest reads
    /// of EVENTQ_CONS on page 1).
    pub fn evtq_cons(&self) -> u32 {
        self.queue_state.lock().evtq_cons
    }

    /// Resets event queue and GERROR state (called on device reset).
    pub fn reset_queue_state(&self) {
        let mut qs = self.queue_state.lock();
        qs.evtq_base_addr = 0;
        qs.evtq_log2size = 0;
        qs.evtq_enabled = false;
        qs.evtq_irqen = false;
        qs.evtq_prod = 0;
        qs.evtq_cons = 0;
        qs.gerror = registers::Gerror::new();
        qs.gerrorn = registers::Gerror::new();
        qs.gerror_irqen = false;
        self.update_gerror_irq(&qs);
    }

    /// Saves the queue and error state that must be persisted.
    ///
    /// Fields derived from SMMU registers (`evtq_base_addr`, `evtq_log2size`,
    /// `evtq_enabled`, `evtq_irqen`, `gerror_irqen`) are re-synced on
    /// restore and are not included in the saved state.
    pub(crate) fn save_queue_state(&self) -> SavedQueueState {
        let qs = self.queue_state.lock();
        // Exhaustively destructure to get a compile error if a field is added.
        let QueueErrorState {
            evtq_base_addr: _,
            evtq_log2size: _,
            evtq_enabled: _,
            evtq_irqen: _,
            evtq_prod,
            evtq_cons,
            gerror,
            gerrorn,
            gerror_irqen: _,
        } = *qs;
        SavedQueueState {
            evtq_prod,
            evtq_cons,
            gerror: gerror.into(),
            gerrorn: gerrorn.into(),
        }
    }

    /// Restores the queue and error state from a saved snapshot.
    ///
    /// The caller must re-sync derived fields (`set_evtq_config`,
    /// `set_evtq_enabled`, `set_irq_ctrl`) before this call, since
    /// this function uses `evtq_irqen` to sync the EVTQ interrupt line.
    pub(crate) fn restore_queue_state(&self, state: SavedQueueState) {
        let SavedQueueState {
            evtq_prod,
            evtq_cons,
            gerror,
            gerrorn,
        } = state;
        let mut qs = self.queue_state.lock();
        qs.evtq_prod = evtq_prod;
        qs.evtq_cons = evtq_cons;
        qs.gerror = registers::Gerror::from(gerror);
        qs.gerrorn = registers::Gerror::from(gerrorn);
        self.update_gerror_irq(&qs);
        // Sync EVTQ wired interrupt line to match restored queue state.
        if qs.evtq_irqen {
            if let Some(irq) = &self.evtq_irq {
                irq.set_level(qs.evtq_prod != qs.evtq_cons);
            }
        }
    }

    /// Register an accelerated backend for a VFIO device.
    ///
    /// The device's stream ID is derived dynamically from `bus_range`
    /// (which holds the guest-assigned bus number) rather than being
    /// fixed at registration time. When the guest writes `CFGI_STE` or
    /// TLBI commands, the emulator matches the command's SID against
    /// each registered device's current bus assignment.
    ///
    /// Registration is atomic with applying the SMMU's *current* policy to the
    /// new device (under the policy lock), so a freshly attached device lands
    /// in the correct boot state instead of staying fail-closed (detached).
    /// At boot the SMMU is disabled, so the policy is bypass-or-abort per
    /// `GBPA.ABORT` and is independent of the StreamID — it is applied even
    /// before the guest has assigned this device's bus number. Once the SMMU
    /// is enabled the policy depends on the per-stream STE; if the bus is not
    /// yet assigned the device is left fail-closed until the guest enumerates
    /// and issues `CFGI_STE`.
    pub fn register_accel_device(
        &self,
        bus_range: AssignedBusRange,
        stream_id_base: u32,
        backend: Arc<dyn AcceleratedStreamBackend>,
    ) {
        let _policy = self.policy_lock.lock();
        self.accel_devices.write().push(AccelDeviceRegistration {
            bus_range: bus_range.clone(),
            stream_id_base,
            backend: backend.clone(),
        });

        // Catch the new device up to the current policy.
        //
        // If the bus is assigned, compute the stream-specific policy. If not,
        // fall back to the disabled-state (StreamID-independent) policy — this
        // is what lets a boot device reach bypass/abort before the guest has
        // enumerated it. With the SMMU enabled and no bus yet, there is no
        // policy to apply: leave the device fail-closed until its `CFGI_STE`.
        let config = match compose_stream_id(&bus_range, stream_id_base, None) {
            Some(sid) => self.current_stream_config(sid),
            None => match self.disabled_policy() {
                Some(config) => config,
                None => return,
            },
        };
        if let Err(e) = backend.set_stream_config(config) {
            tracelimit::warn_ratelimited!(
                error = &*e as &dyn std::error::Error,
                "smmu: failed to apply initial stream config on register"
            );
        }
    }

    /// Computes the SMMU's current policy for the given stream.
    ///
    /// This is the single source of truth for a stream's accelerated policy:
    /// when the SMMU is disabled the result is `GBPA.ABORT ? Abort : Bypass`;
    /// when enabled the STE for `sid` is read from guest memory and decoded.
    ///
    /// The translation (`inner`) lock is only held to snapshot register state;
    /// it is released before the STE read so callers can apply the result to a
    /// backend (a blocking ioctl) without nesting the translation lock around
    /// it.
    pub(crate) fn current_stream_config(&self, sid: u32) -> StreamConfig {
        let (enabled, gbpa_abort, strtab_base, strtab_log2size) = {
            let inner = self.inner.read();
            (
                inner.enabled,
                inner.gbpa_abort,
                inner.strtab_base,
                inner.strtab_log2size,
            )
        };

        if !enabled {
            return if gbpa_abort {
                StreamConfig::Abort
            } else {
                StreamConfig::Bypass
            };
        }

        // SMMU enabled: read and decode this stream's STE.
        let max_streams = 1u64 << strtab_log2size;
        if (sid as u64) >= max_streams {
            tracelimit::warn_ratelimited!(
                sid,
                max_streams,
                "smmu: current_stream_config for out-of-range SID; aborting"
            );
            return StreamConfig::Abort;
        }
        let ste_addr = strtab_base + (sid as u64) * (crate::spec::ste::STE_SIZE as u64);
        match self
            .guest_memory
            .read_plain::<crate::spec::ste::Ste>(ste_addr)
        {
            Ok(ste) => StreamConfig::from_ste(sid, &ste),
            Err(e) => {
                tracelimit::warn_ratelimited!(
                    error = &e as &dyn std::error::Error,
                    ste_addr,
                    sid,
                    "smmu: failed to read STE for current_stream_config; aborting"
                );
                StreamConfig::Abort
            }
        }
    }

    /// Returns the StreamID-independent policy that applies while the SMMU is
    /// disabled (`Some(Bypass)` or `Some(Abort)` per `GBPA.ABORT`), or `None`
    /// when the SMMU is enabled (the policy then depends on the per-stream
    /// STE).
    fn disabled_policy(&self) -> Option<StreamConfig> {
        let inner = self.inner.read();
        (!inner.enabled).then(|| {
            if inner.gbpa_abort {
                StreamConfig::Abort
            } else {
                StreamConfig::Bypass
            }
        })
    }

    /// Re-computes and applies the current policy for a single stream's
    /// accelerated backend (if one is registered).
    ///
    /// Serialized against registration and other policy updates via the
    /// policy lock so the last write wins. Used for `CFGI_STE`.
    pub(crate) fn apply_stream_config(&self, sid: u32) {
        let _policy = self.policy_lock.lock();
        let Some(backend) = self.get_stream_backend(sid) else {
            return;
        };
        let config = self.current_stream_config(sid);
        if let Err(e) = backend.set_stream_config(config) {
            tracelimit::warn_ratelimited!(
                error = &*e as &dyn std::error::Error,
                sid,
                "smmu: failed to apply stream config"
            );
        }
    }

    /// Re-computes and applies the current policy for every registered
    /// accelerated backend.
    ///
    /// Used on events that change policy globally without otherwise mutating
    /// translation state: `CFGI_STE_RANGE` / `CFGI_ALL`. (The state-mutating
    /// events — CR0/GBPA writes, reset, restore — re-drive atomically via
    /// [`set_enabled`](Self::set_enabled),
    /// [`set_gbpa_abort`](Self::set_gbpa_abort), and
    /// [`sync_translation_state`](Self::sync_translation_state).)
    /// Serialized via the policy lock.
    pub(crate) fn apply_all_stream_configs(&self) {
        let _policy = self.policy_lock.lock();
        self.apply_all_locked();
    }

    /// Re-drives every registered backend to its current policy. The caller
    /// must already hold `policy_lock` (this is the shared body of
    /// [`apply_all_stream_configs`](Self::apply_all_stream_configs) and the
    /// state-mutating setters).
    fn apply_all_locked(&self) {
        for (sid, backend) in self.stream_backend_entries() {
            let config = self.current_stream_config(sid);
            if let Err(e) = backend.set_stream_config(config) {
                tracelimit::warn_ratelimited!(
                    error = &*e as &dyn std::error::Error,
                    sid,
                    "smmu: failed to apply stream config"
                );
            }
        }
    }

    /// Look up the accelerated backend for a stream ID.
    ///
    /// Iterates registered accel devices and computes the current SID
    /// from each device's `AssignedBusRange`. Returns `None` for
    /// emulated devices (software walk path) or devices whose bus
    /// number has not been assigned yet.
    pub fn get_stream_backend(&self, sid: u32) -> Option<Arc<dyn AcceleratedStreamBackend>> {
        let devices = self.accel_devices.read();
        for reg in devices.iter() {
            if let Some(dev_sid) = compose_stream_id(&reg.bus_range, reg.stream_id_base, None) {
                if dev_sid == sid {
                    return Some(reg.backend.clone());
                }
            }
        }
        None
    }

    /// Returns all registered accelerated stream backends.
    ///
    /// Used by CMDQ processing to forward broadcast TLBI commands to all
    /// accelerated streams.
    pub fn all_stream_backends(&self) -> Vec<Arc<dyn AcceleratedStreamBackend>> {
        self.accel_devices
            .read()
            .iter()
            .map(|reg| reg.backend.clone())
            .collect()
    }

    /// Returns all registered stream backend entries as (SID, backend) pairs.
    ///
    /// Used by CMDQ processing for broadcast CFGI_STE_RANGE (CFGI_ALL)
    /// to re-read each accelerated stream's STE. Entries whose bus number
    /// has not been assigned yet are skipped.
    pub fn stream_backend_entries(&self) -> Vec<(u32, Arc<dyn AcceleratedStreamBackend>)> {
        self.accel_devices
            .read()
            .iter()
            .filter_map(|reg| {
                let sid = compose_stream_id(&reg.bus_range, reg.stream_id_base, None)?;
                Some((sid, reg.backend.clone()))
            })
            .collect()
    }

    /// Returns the guest memory handle for reading page tables and STEs.
    pub fn guest_memory(&self) -> &GuestMemory {
        &self.guest_memory
    }

    /// Returns the stream table base address and log2 size.
    ///
    /// Used by CMDQ processing to read STE bytes for accelerated streams.
    pub fn strtab_config(&self) -> (u64, u8) {
        let inner = self.inner.read();
        (inner.strtab_base, inner.strtab_log2size)
    }

    /// Translate an IOVA to a GPA for the given stream ID.
    ///
    /// Callers that need to hold the lock across translation and a subsequent
    /// memory access should use [`translate_with`] instead.
    fn translate(&self, sid: u32, iova: u64, write: bool) -> TranslateResult {
        let inner = self.inner.read();
        self.translate_locked(&inner, sid, iova, write)
    }

    /// Translate an IOVA to a GPA while holding the read lock.
    ///
    /// The caller holds `inner` across both translation and the subsequent
    /// memory access, preventing SMMU config changes (disable, stream table
    /// base update) from creating a TOCTOU between translation and access.
    fn translate_locked(
        &self,
        inner: &SharedStateInner,
        sid: u32,
        iova: u64,
        write: bool,
    ) -> TranslateResult {
        if !inner.enabled {
            // The SMMU is disabled: GBPA selects the global policy. ABORT
            // terminates the transaction (with no event — there is no stream
            // context to fault against); otherwise transactions bypass
            // (IOVA = GPA). The matching accel policy is computed in
            // [`current_stream_config`].
            if inner.gbpa_abort {
                return TranslateResult::GlobalAbort;
            }
            return TranslateResult::Bypass;
        }

        // Look up the STE.
        let ste = match translate::lookup_ste(
            &self.guest_memory,
            inner.strtab_base,
            inner.strtab_log2size,
            sid,
            inner.oas_mask,
        ) {
            Ok(ste) => ste,
            Err(fault) => return TranslateResult::Fault(fault.event),
        };

        // Dispatch on STE config.
        let action = match translate::ste_config_action(&ste) {
            Ok(action) => action,
            Err(_) => return TranslateResult::Fault(EvtEntry::bad_ste(sid)),
        };

        match action {
            translate::SteAction::Abort => TranslateResult::Abort(EvtEntry::bad_ste(sid)),
            translate::SteAction::Bypass => TranslateResult::Bypass,
            translate::SteAction::S1Translate => {
                // Look up the CD.
                let cd =
                    match translate::lookup_cd(&self.guest_memory, &ste, sid, 0, inner.oas_mask) {
                        Ok(cd) => cd,
                        Err(fault) => return TranslateResult::Fault(fault.event),
                    };

                // Extract translation context (caps CD.IPS to device OAS).
                let ctx = match translate::translation_context(&cd, sid, inner.oas_mask) {
                    Ok(ctx) => ctx,
                    Err(fault) => return TranslateResult::Fault(fault.event),
                };

                // Walk the page table.
                match translate::walk_s1(&self.guest_memory, &ctx, iova, write, sid) {
                    Ok(tr) => TranslateResult::Translated(tr.gpa),
                    Err(fault) => TranslateResult::Fault(fault.event),
                }
            }
        }
    }

    /// Write an event record directly to the guest's event queue.
    ///
    /// Called from per-device DMA fault paths and from the emulator's
    /// command processing. If the queue is full, drops the event and
    /// logs a warning. If an event is successfully written, pulses
    /// the EVTQ wired SPI interrupt (if enabled).
    pub fn write_event(&self, event: EvtEntry) {
        let mut qs = self.queue_state.lock();
        if !qs.evtq_enabled {
            return;
        }

        let max_entries = 1u32 << qs.evtq_log2size;
        let index_mask = (max_entries << 1) - 1;
        let prod = qs.evtq_prod & index_mask;
        let cons = qs.evtq_cons & index_mask;

        // Check if the queue is full. Full when the index bits match but
        // the wrap bit differs: (prod ^ cons) == max_entries.
        if (prod ^ cons) == max_entries {
            // Signal EVTQ overflow via GERROR.EVTQ_ABT_ERR — updates
            // the GERROR register and interrupt line under the same lock.
            self.signal_evtq_overflow(&mut qs);
            tracelimit::warn_ratelimited!("smmu: EVTQ full, dropping event");
            return;
        }

        // Write the 32-byte event record to guest memory.
        let index = prod & (max_entries - 1);
        let entry_addr = qs.evtq_base_addr + (index as u64) * (EvtEntry::SIZE as u64);

        if let Err(e) = self.guest_memory.write_at(entry_addr, event.as_bytes()) {
            tracelimit::warn_ratelimited!(
                error = &e as &dyn std::error::Error,
                entry_addr,
                "smmu: failed to write EVTQ entry to guest memory"
            );
            return;
        }

        // Advance EVTQ_PROD.
        qs.evtq_prod = (prod + 1) & index_mask;

        // Assert EVTQ wired interrupt — held high while queue is non-empty.
        // Deasserted when the guest drains events via CONS writes.
        if qs.evtq_irqen {
            if let Some(irq) = &self.evtq_irq {
                irq.set_level(true);
            }
        }
    }

    /// Creates a translator for PCI devices behind this SMMU.
    ///
    /// `stream_id_base` is the offset into this SMMU's stream table for the
    /// root complex this device belongs to. The translator computes the
    /// stream ID as `stream_id_base + rid` at each access.
    pub fn translator(self: &Arc<Self>, stream_id_base: u32) -> SmmuTranslator {
        SmmuTranslator {
            shared: self.clone(),
            stream_id_base,
        }
    }

    /// Creates an SMMU irqfd wrapper for a PCI device behind this SMMU.
    ///
    /// `stream_id_base` is the offset into this SMMU's stream table for the
    /// root complex this device belongs to.
    ///
    /// Irqfd routes created from the returned wrapper will translate MSI
    /// addresses through the SMMU page tables before programming the
    /// kernel route.
    pub fn wrap_irqfd(
        self: &Arc<Self>,
        stream_id_base: u32,
        inner: Arc<dyn IrqFd>,
    ) -> Arc<SmmuIrqFd> {
        Arc::new(SmmuIrqFd {
            shared: self.clone(),
            stream_id_base,
            inner,
        })
    }
}

/// An [`IommuTranslator`](iommu_common::IommuTranslator) for the ARM SMMUv3.
///
/// One `SmmuTranslator` is shared by all PCI devices behind the same SMMU.
/// The requester ID (RID / BDF) is passed at each translation call and
/// combined with the `stream_id_base` to form the SMMU stream ID.
pub struct SmmuTranslator {
    shared: Arc<SmmuSharedState>,
    /// Offset into the SMMU's stream table for this root complex.
    stream_id_base: u32,
}

/// DMA translation error from the SMMU.
///
/// The fault event has already been queued to the SMMU's event queue;
/// this error carries the key fields for diagnostic purposes.
#[derive(Debug, thiserror::Error)]
#[error("SMMU DMA fault: event {event_id:#04x} SID {sid:#x} addr {input_addr:#x}")]
pub struct SmmuDmaFault {
    /// Event type ID.
    event_id: u8,
    /// StreamID of the faulting device.
    sid: u32,
    /// Faulting input address.
    input_addr: u64,
}

impl SmmuDmaFault {
    fn from_event(event: &EvtEntry) -> Self {
        Self {
            event_id: event.header.event_id(),
            sid: event.sid,
            input_addr: event.input_addr,
        }
    }

    /// A global abort (disabled SMMU, `GBPA.ABORT=1`). No event record is
    /// generated, so `event_id` is 0 to signify "no event".
    fn global_abort(sid: u32, input_addr: u64) -> Self {
        Self {
            event_id: 0,
            sid,
            input_addr,
        }
    }
}

impl iommu_common::IommuTranslator for SmmuTranslator {
    type Error = SmmuDmaFault;

    fn max_iova(&self) -> u64 {
        // The SMMUv3 architecture supports up to 48-bit input addresses.
        // This is the maximum across all configurations: stage-1 only,
        // stage-2 only, and nested (stage-1 IAS and stage-2 IPA width
        // are both bounded by 48 bits).
        1u64 << 48
    }

    fn translate<R>(
        &self,
        rid: u16,
        iova: u64,
        write: bool,
        op: impl FnOnce(u64) -> R,
    ) -> Result<R, iommu_common::TranslationFault<SmmuDmaFault>> {
        let sid = self.stream_id_base + (rid as u32);

        // Hold the read lock across translate + op to prevent SMMU config
        // from changing between getting the GPA and using it.
        let inner = self.shared.inner.read();
        let gpa = match self.shared.translate_locked(&inner, sid, iova, write) {
            TranslateResult::Bypass => iova,
            TranslateResult::Translated(gpa) => gpa,
            TranslateResult::GlobalAbort => {
                drop(inner);
                // Disabled SMMU with GBPA.ABORT=1: terminate with no event.
                return Err(iommu_common::TranslationFault {
                    iova,
                    error: SmmuDmaFault::global_abort(sid, iova),
                });
            }
            TranslateResult::Abort(event) | TranslateResult::Fault(event) => {
                drop(inner);
                let error = SmmuDmaFault::from_event(&event);
                self.shared.write_event(event);
                return Err(iommu_common::TranslationFault { iova, error });
            }
        };

        let result = op(gpa);
        drop(inner);
        Ok(result)
    }
}

/// A [`SignalMsi`] wrapper that translates MSI addresses through the SMMU.
///
/// When a device behind the SMMU fires an MSI, the MSI address may be an
/// IOVA (Linux maps MSI doorbell pages into the device's IOVA space via
/// `iommu_dma_prepare_msi()`). This wrapper translates the address before
/// forwarding to the inner MSI target (typically an ITS or GICv2m wrapper).
pub struct SmmuSignalMsi {
    shared: Arc<SmmuSharedState>,
    /// Offset into the SMMU's stream table for this root complex.
    stream_id_base: u32,
    inner: Arc<dyn SignalMsi>,
}

impl SmmuSignalMsi {
    /// Creates a new SMMU MSI translator wrapping the given inner target.
    pub fn new(
        shared: Arc<SmmuSharedState>,
        stream_id_base: u32,
        inner: Arc<dyn SignalMsi>,
    ) -> Self {
        Self {
            shared,
            stream_id_base,
            inner,
        }
    }
}

impl SignalMsi for SmmuSignalMsi {
    fn signal_msi(&self, devid: Option<u32>, address: u64, data: u32) {
        // MsiTarget resolves devid to a BDF before calling us.
        let Some(bdf) = devid else {
            return;
        };
        let sid = self.stream_id_base + (bdf & 0xFFFF);

        match self.shared.translate(sid, address, true) {
            TranslateResult::Bypass => {
                self.inner.signal_msi(devid, address, data);
            }
            TranslateResult::Translated(gpa) => {
                self.inner.signal_msi(devid, gpa, data);
            }
            TranslateResult::GlobalAbort => {
                // Disabled SMMU with GBPA.ABORT=1: drop the MSI, no event.
                tracelimit::warn_ratelimited!(sid, address, "smmu: MSI globally aborted (GBPA)");
            }
            TranslateResult::Abort(event) => {
                self.shared.write_event(event);
                tracelimit::warn_ratelimited!(sid, address, "smmu: MSI aborted by STE config");
            }
            TranslateResult::Fault(event) => {
                self.shared.write_event(event);
                tracelimit::warn_ratelimited!(sid, address, "smmu: MSI translation fault");
            }
        }
    }
}

/// An [`IrqFd`] wrapper that produces SMMU-translating irqfd routes.
///
/// When a device behind the SMMU programs its MSI-X table, the MSI address
/// may be an IOVA. This wrapper creates [`SmmuIrqFdRoute`] instances that
/// translate the address through the SMMU before forwarding to the inner
/// irqfd route (which may itself be an ITS wrapper).
pub struct SmmuIrqFd {
    shared: Arc<SmmuSharedState>,
    /// Offset into the SMMU's stream table for this root complex.
    stream_id_base: u32,
    inner: Arc<dyn IrqFd>,
}

impl IrqFd for SmmuIrqFd {
    fn new_irqfd_route(&self) -> anyhow::Result<Box<dyn IrqFdRoute>> {
        let inner_route = self.inner.new_irqfd_route()?;
        Ok(Box::new(SmmuIrqFdRoute {
            shared: self.shared.clone(),
            stream_id_base: self.stream_id_base,
            inner: inner_route,
        }))
    }
}

/// An [`IrqFdRoute`] wrapper that translates the MSI address through the
/// SMMU on [`enable`](IrqFdRoute::enable).
///
/// Translation happens at route-programming time (when the guest writes
/// the MSI-X table), not per-interrupt. If the guest changes SMMU page
/// tables after programming MSI-X, it must also re-program the MSI-X
/// entry (which is the normal flow — the IOMMU driver does this).
struct SmmuIrqFdRoute {
    shared: Arc<SmmuSharedState>,
    /// Offset into the SMMU's stream table for this root complex.
    stream_id_base: u32,
    inner: Box<dyn IrqFdRoute>,
}

impl IrqFdRoute for SmmuIrqFdRoute {
    fn event(&self) -> &Event {
        self.inner.event()
    }

    fn enable(&self, address: u64, data: u32, devid: Option<u32>) {
        // MsiRoute resolves devid to a BDF before calling us.
        let Some(bdf) = devid else {
            return;
        };
        let sid = self.stream_id_base + (bdf & 0xFFFF);

        match self.shared.translate(sid, address, true) {
            TranslateResult::Bypass => {
                self.inner.enable(address, data, devid);
            }
            TranslateResult::Translated(gpa) => {
                self.inner.enable(gpa, data, devid);
            }
            TranslateResult::GlobalAbort => {
                // Disabled SMMU with GBPA.ABORT=1: drop the route, no event.
                tracelimit::warn_ratelimited!(
                    sid,
                    address,
                    "smmu: irqfd MSI route globally aborted (GBPA)"
                );
            }
            TranslateResult::Abort(event) => {
                self.shared.write_event(event);
                tracelimit::warn_ratelimited!(
                    sid,
                    address,
                    "smmu: irqfd MSI route aborted by STE config"
                );
            }
            TranslateResult::Fault(event) => {
                self.shared.write_event(event);
                tracelimit::warn_ratelimited!(
                    sid,
                    address,
                    "smmu: irqfd MSI route translation fault"
                );
            }
        }
    }

    fn disable(&self) {
        self.inner.disable();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::cd::CD_SIZE;
    use crate::spec::cd::CdDw0;
    use crate::spec::cd::CdDw1;
    use crate::spec::cd::Ips;
    use crate::spec::cd::Tg0;
    use crate::spec::events::EventId;
    use crate::spec::pt::ApBits;
    use crate::spec::pt::PtDesc;
    use crate::spec::ste::STE_SIZE;
    use crate::spec::ste::Ste;
    use crate::spec::ste::SteConfig;
    use crate::spec::ste::SteDw0;
    use crate::spec::ste::SteDw1;
    use parking_lot::Mutex;
    use pci_core::bus_range::AssignedBusRange;
    use std::sync::Arc;

    // Memory layout for tests. All addresses fit within a 6 MB allocation
    // to avoid excessive memory usage in test processes.
    const STRTAB_BASE: u64 = 0x10_0000;
    const STRTAB_LOG2SIZE: u8 = 10;
    const CD_BASE: u64 = 0x20_0000;
    const PT_L1_BASE: u64 = 0x30_1000;
    const PT_L2_BASE: u64 = 0x30_2000;
    const PT_L3_BASE: u64 = 0x30_3000;
    // DATA_GPA and EVTQ_BASE are kept low so the guest memory allocation
    // does not need to span gigabytes. Tests read/write data at DATA_GPA
    // and the SMMU writes fault events at EVTQ_BASE.
    const DATA_GPA: u64 = 0x40_0000;
    /// EVTQ base GPA for tests (must not overlap other test regions).
    const EVTQ_BASE: u64 = 0x50_0000;
    /// EVTQ log2 size for tests (3 = 8 entries).
    const EVTQ_LOG2SIZE: u8 = 3;
    const TEST_SEGMENT: u16 = 0;
    /// Stream ID base for the test root complex (matches IORT output_base).
    const TEST_STREAM_ID_BASE: u32 = (TEST_SEGMENT as u32) << 16;
    const TEST_BUS: u8 = 1;
    /// The RID for the test device: (bus << 8) | devfn.
    const TEST_RID: u32 = (TEST_BUS as u32) << 8;

    /// A mock SignalMsi that records calls.
    struct MockSignalMsi {
        calls: Mutex<Vec<(Option<u32>, u64, u32)>>,
    }

    impl MockSignalMsi {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: Mutex::new(Vec::new()),
            })
        }

        fn take_calls(&self) -> Vec<(Option<u32>, u64, u32)> {
            std::mem::take(&mut *self.calls.lock())
        }
    }

    impl SignalMsi for MockSignalMsi {
        fn signal_msi(&self, devid: Option<u32>, address: u64, data: u32) {
            self.calls.lock().push((devid, address, data));
        }
    }

    fn make_bus_range() -> AssignedBusRange {
        let br = AssignedBusRange::new();
        br.set_bus_range(TEST_BUS, TEST_BUS);
        br
    }

    fn expected_sid() -> u32 {
        TEST_STREAM_ID_BASE + ((TEST_BUS as u32) << 8)
    }

    /// Test-only helper: creates a translating GuestMemory and SmmuSignalMsi
    /// pair for a device behind the SMMU.
    fn device_context(
        state: &Arc<SmmuSharedState>,
        bus_range: AssignedBusRange,
        stream_id_base: u32,
        inner_gm: &GuestMemory,
        inner_msi: Arc<dyn SignalMsi>,
    ) -> (GuestMemory, Arc<SmmuSignalMsi>) {
        let translator = state.translator(stream_id_base);
        let gm = iommu_common::TranslatingMemory::new_guest_memory(
            "smmu-translating",
            translator,
            bus_range,
            inner_gm.clone(),
        );
        let signal_msi = Arc::new(SmmuSignalMsi::new(state.clone(), stream_id_base, inner_msi));
        (gm, signal_msi)
    }

    fn write_ste(gm: &GuestMemory, sid: u32, ste: &Ste) {
        let addr = STRTAB_BASE + (sid as u64) * (STE_SIZE as u64);
        gm.write_plain(addr, ste).expect("write STE");
    }

    fn make_s1_ste(cd_base: u64) -> Ste {
        use crate::spec::cd::CD_SIZE;
        let _ = CD_SIZE;
        Ste {
            qw0: SteDw0::new()
                .with_v(true)
                .with_config(SteConfig::S1_TRANS.0)
                .with_s1_context_ptr(cd_base >> 6)
                .with_s1_cd_max(0),
            qw1: SteDw1::new(),
            _qw2_7: [0; 6],
        }
    }

    fn make_bypass_ste() -> Ste {
        Ste {
            qw0: SteDw0::new().with_v(true).with_config(SteConfig::BYPASS.0),
            qw1: SteDw1::new(),
            _qw2_7: [0; 6],
        }
    }

    fn make_abort_ste() -> Ste {
        Ste {
            qw0: SteDw0::new().with_v(true).with_config(SteConfig::ABORT.0),
            qw1: SteDw1::new(),
            _qw2_7: [0; 6],
        }
    }

    fn write_cd(gm: &GuestMemory, cd_base: u64, ssid: u32) {
        use crate::spec::cd::Cd;
        let cd = Cd {
            qw0: CdDw0::new()
                .with_v(true)
                .with_t0sz(32)
                .with_tg0(Tg0::GRAN_4K.0)
                .with_ips(Ips::IPS_40.0)
                .with_aa64(true)
                .with_a(true)
                .with_asid(1),
            qw1: CdDw1::new().with_ttb0(PT_L1_BASE >> 4),
            _qw2: 0,
            mair0: 0xFF440C0400,
            mair1: 0,
            _qw5_7: [0; 3],
        };
        let addr = cd_base + (ssid as u64) * (CD_SIZE as u64);
        gm.write_plain(addr, &cd).expect("write CD");
    }

    fn table_desc(next_table: u64) -> u64 {
        PtDesc::new()
            .with_valid(true)
            .with_desc_type(true)
            .with_addr_bits(next_table >> 12)
            .into()
    }

    fn page_desc(output_addr: u64) -> u64 {
        PtDesc::new()
            .with_valid(true)
            .with_desc_type(true)
            .with_af(true)
            .with_ap(ApBits::RW_EL1.0)
            .with_addr_bits(output_addr >> 12)
            .into()
    }

    fn write_pt_desc(gm: &GuestMemory, addr: u64, desc: u64) {
        gm.write_plain(addr, &desc).expect("write PT desc");
    }

    /// Set up a complete SMMU translation context:
    /// STE (S1_TRANS) → CD → page table mapping IOVA 0..4K → DATA_GPA.
    fn setup_translation(gm: &GuestMemory, sid: u32) {
        // Write STE.
        write_ste(gm, sid, &make_s1_ste(CD_BASE));
        // Write CD.
        write_cd(gm, CD_BASE, 0);
        // Build 3-level page table (T0SZ=32, 4K granule: L1, L2, L3).
        // L1[0] → L2
        write_pt_desc(gm, PT_L1_BASE, table_desc(PT_L2_BASE));
        // L2[0] → L3
        write_pt_desc(gm, PT_L2_BASE, table_desc(PT_L3_BASE));
        // L3[0] → page at DATA_GPA
        write_pt_desc(gm, PT_L3_BASE, page_desc(DATA_GPA));
    }

    fn make_shared_state(gm: &GuestMemory) -> Arc<SmmuSharedState> {
        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            false,
            None,
            None,
        );
        state.set_strtab(STRTAB_BASE, STRTAB_LOG2SIZE);
        state.set_enabled(true);
        // Enable EVTQ so fault events are written to guest memory.
        state.set_evtq_config(EVTQ_BASE, EVTQ_LOG2SIZE);
        state.set_evtq_enabled(true);
        state
    }

    /// Count events in the EVTQ by reading EVTQ_PROD from shared state.
    fn evtq_event_count(state: &SmmuSharedState) -> u32 {
        state.evtq_prod()
    }

    // =========================================================================
    // TranslatingMemory tests
    // =========================================================================

    #[test]
    fn test_translating_memory_basic_read() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();
        setup_translation(&gm, sid);

        // Write test data at the physical GPA.
        let data = b"hello SMMU";
        gm.write_at(DATA_GPA, data).unwrap();

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Read via IOVA 0 → should get data from DATA_GPA.
        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_translating_memory_basic_write() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();
        setup_translation(&gm, sid);

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Write via IOVA.
        let data = b"write test";
        translating_gm.write_at(0, data).unwrap();

        // Verify data appears at the physical GPA.
        let mut buf = vec![0u8; data.len()];
        gm.read_at(DATA_GPA, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_translating_memory_with_offset() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();
        setup_translation(&gm, sid);

        // Write data at GPA + 0x100.
        let data = b"offset data";
        gm.write_at(DATA_GPA + 0x100, data).unwrap();

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Read via IOVA 0x100 → DATA_GPA + 0x100.
        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0x100, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_translating_memory_cross_page() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        // Set up STE and CD.
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        write_cd(&gm, CD_BASE, 0);

        // Map two adjacent pages:
        // L3[0] → DATA_GPA (page at IOVA 0x0000)
        // L3[1] → DATA_GPA + 0x2000 (page at IOVA 0x1000)
        write_pt_desc(&gm, PT_L1_BASE, table_desc(PT_L2_BASE));
        write_pt_desc(&gm, PT_L2_BASE, table_desc(PT_L3_BASE));
        write_pt_desc(&gm, PT_L3_BASE, page_desc(DATA_GPA));
        write_pt_desc(&gm, PT_L3_BASE + 8, page_desc(DATA_GPA + 0x2000));

        // Write data spanning the page boundary.
        let data_page1 = vec![0xAAu8; 0x10];
        let data_page2 = vec![0xBBu8; 0x10];
        gm.write_at(DATA_GPA + 0xFF0, &data_page1).unwrap();
        gm.write_at(DATA_GPA + 0x2000, &data_page2).unwrap();

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Read 32 bytes starting at IOVA 0xFF0, crossing into page 2.
        let mut buf = vec![0u8; 0x20];
        translating_gm.read_at(0xFF0, &mut buf).unwrap();
        assert_eq!(&buf[..0x10], &data_page1);
        assert_eq!(&buf[0x10..], &data_page2);
    }

    #[test]
    fn test_translating_memory_bypass() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        // STE in bypass mode.
        write_ste(&gm, sid, &make_bypass_ste());

        // Write data at GPA 0x1000.
        let data = b"bypass data";
        gm.write_at(0x1000, data).unwrap();

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Read via IOVA = GPA (identity mapping in bypass mode).
        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0x1000, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_translating_memory_abort() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        // STE in abort mode.
        write_ste(&gm, sid, &make_abort_ste());

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Read should fail.
        let mut buf = vec![0u8; 4];
        let result = translating_gm.read_at(0, &mut buf);
        assert!(result.is_err());

        // Should have written an event to the EVTQ.
        assert_eq!(evtq_event_count(&state), 1);
    }

    #[test]
    fn test_translating_memory_unmapped() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        // Set up STE and CD, but NO page table entries (L1 is all zeros).
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        write_cd(&gm, CD_BASE, 0);
        // L1 is all zeros → translation fault.

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        let mut buf = vec![0u8; 4];
        let result = translating_gm.read_at(0, &mut buf);
        assert!(result.is_err());

        // Should have written a fault event to the EVTQ.
        assert_eq!(evtq_event_count(&state), 1);
        // Read the event from the EVTQ in guest memory.
        let written: EvtEntry = gm.read_plain(EVTQ_BASE).expect("read event");
        assert_eq!(written.event_id(), EventId::F_TRANSLATION);
    }

    #[test]
    fn test_translating_memory_unassigned_bus() {
        let gm = GuestMemory::allocate(0x60_0000);

        let state = make_shared_state(&gm);
        // Bus range NOT assigned (secondary_bus = 0) → RID = 0.
        // With SMMU enabled, stream ID 0 has no valid STE → fault.
        let bus_range = AssignedBusRange::new();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Should fault because STE 0 is not configured.
        let mut buf = vec![0u8; 10];
        translating_gm.read_at(0x2000, &mut buf).unwrap_err();
    }

    #[test]
    fn test_translating_memory_smmu_disabled() {
        let gm = GuestMemory::allocate(0x60_0000);

        // Write data at GPA 0x3000.
        let data = b"disabled smmu";
        gm.write_at(0x3000, data).unwrap();

        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            false,
            None,
            None,
        );
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        // Should bypass translation.
        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0x3000, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    // =========================================================================
    // SmmuSignalMsi tests
    // =========================================================================

    #[test]
    fn test_signal_msi_translated() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();
        setup_translation(&gm, sid);

        // Also map a doorbell page: IOVA 0x800 → DATA_GPA + 0x1000.
        write_pt_desc(&gm, PT_L3_BASE + 8, page_desc(DATA_GPA + 0x1000));

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) = device_context(
            &state,
            bus_range,
            TEST_STREAM_ID_BASE,
            &gm,
            mock_msi.clone(),
        );

        // Fire MSI with IOVA address 0x1040 (page 1 + offset 0x40).
        // devid is a RID — the SMMU combines it with segment to get the SID.
        smmu_msi.signal_msi(Some(TEST_RID), 0x1040, 0xDEAD);

        let calls = mock_msi.take_calls();
        assert_eq!(calls.len(), 1);
        // Translated address: DATA_GPA + 0x1000 + 0x40.
        assert_eq!(calls[0], (Some(TEST_RID), DATA_GPA + 0x1040, 0xDEAD));
    }

    #[test]
    fn test_signal_msi_bypass() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        write_ste(&gm, sid, &make_bypass_ste());

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) = device_context(
            &state,
            bus_range,
            TEST_STREAM_ID_BASE,
            &gm,
            mock_msi.clone(),
        );

        // MsiTarget resolves devid to a BDF before calling SmmuSignalMsi.
        smmu_msi.signal_msi(Some(TEST_RID), 0xFEE0_0000, 0x42);

        let calls = mock_msi.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], (Some(TEST_RID), 0xFEE0_0000, 0x42));
    }

    #[test]
    fn test_signal_msi_unmapped() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        // STE with S1 translation, but no page table entries.
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        write_cd(&gm, CD_BASE, 0);

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) = device_context(
            &state,
            bus_range,
            TEST_STREAM_ID_BASE,
            &gm,
            mock_msi.clone(),
        );

        // Fire MSI with unmapped address. devid is a RID.
        smmu_msi.signal_msi(Some(TEST_RID), 0x1000, 0x42);

        // MSI should NOT be forwarded.
        let calls = mock_msi.take_calls();
        assert!(calls.is_empty());

        // Fault event should be written to the EVTQ.
        assert_eq!(evtq_event_count(&state), 1);
    }

    #[test]
    fn test_signal_msi_devid_passthrough() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();

        write_ste(&gm, sid, &make_bypass_ste());

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) = device_context(
            &state,
            bus_range,
            TEST_STREAM_ID_BASE,
            &gm,
            mock_msi.clone(),
        );

        // devid (RID) should be passed through unchanged to the inner MSI.
        smmu_msi.signal_msi(Some(TEST_RID), 0x1000, 0x42);

        let calls = mock_msi.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, Some(TEST_RID));
    }

    #[test]
    fn test_signal_msi_no_devid() {
        let gm = GuestMemory::allocate(0x60_0000);

        let state = make_shared_state(&gm);
        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) = device_context(
            &state,
            bus_range,
            TEST_STREAM_ID_BASE,
            &gm,
            mock_msi.clone(),
        );

        // devid=None means no BDF — MSI should be dropped.
        smmu_msi.signal_msi(None, 0xFEE0_0000, 0x42);

        let calls = mock_msi.take_calls();
        assert_eq!(calls.len(), 0);
    }

    // =========================================================================
    // Stream ID remapping tests (non-zero stream_id_base)
    // =========================================================================

    #[test]
    fn test_translating_memory_nonzero_stream_id_base() {
        let gm = GuestMemory::allocate(0x60_0000);

        // Use a non-zero stream_id_base (simulating a second root complex
        // with its own region in the SMMU stream table).
        // stream_id_base=256, bus=1 → SID = 256 + 256 = 512 (within 1024).
        let stream_id_base: u32 = 256;
        let bus: u8 = 1;
        let sid = stream_id_base + ((bus as u32) << 8);

        // Set up translation for the remapped stream ID.
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        write_cd(&gm, CD_BASE, 0);
        write_pt_desc(&gm, PT_L1_BASE, table_desc(PT_L2_BASE));
        write_pt_desc(&gm, PT_L2_BASE, table_desc(PT_L3_BASE));
        write_pt_desc(&gm, PT_L3_BASE, page_desc(DATA_GPA));

        let data = b"remapped sid test";
        gm.write_at(DATA_GPA, data).unwrap();

        let state = make_shared_state(&gm);
        let bus_range = AssignedBusRange::new();
        bus_range.set_bus_range(bus, bus);
        let mock_msi = MockSignalMsi::new();

        let (translating_gm, _msi) =
            device_context(&state, bus_range, stream_id_base, &gm, mock_msi);

        // Read via IOVA 0 → should find the STE at the remapped stream ID.
        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0, &mut buf).unwrap();
        assert_eq!(&buf, data);
    }

    #[test]
    fn test_signal_msi_nonzero_stream_id_base() {
        let gm = GuestMemory::allocate(0x60_0000);

        // Non-zero base (different root complex).
        let stream_id_base: u32 = 256;
        let bus: u8 = 1;
        let sid = stream_id_base + ((bus as u32) << 8);

        // Set up bypass STE for the remapped stream ID.
        write_ste(&gm, sid, &make_bypass_ste());

        let state = make_shared_state(&gm);
        let bus_range = AssignedBusRange::new();
        bus_range.set_bus_range(bus, bus);
        let mock_msi = MockSignalMsi::new();

        let (_gm, smmu_msi) =
            device_context(&state, bus_range, stream_id_base, &gm, mock_msi.clone());

        // Fire MSI — bypass mode means address passes through unchanged.
        let rid = (bus as u32) << 8;
        smmu_msi.signal_msi(Some(rid), 0xFEE0_0000, 0x99);

        let calls = mock_msi.take_calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0], (Some(rid), 0xFEE0_0000, 0x99));
    }

    // =========================================================================
    // resolve_host_caps (accel host/guest compatibility) tests
    // =========================================================================

    /// A `HostSmmuCaps` that is compatible with everything the emulator
    /// advertises (AArch64, little-endian, 4K granule, ample OAS).
    fn compatible_host_caps() -> crate::HostSmmuCaps {
        crate::HostSmmuCaps {
            oas: Ips::IPS_48,
            ttf: registers::Idr0Ttf::new().with_aarch64(true),
            ttendian: registers::Idr0TtEndian::LE,
            gran4k: true,
        }
    }

    /// An accel-mode shared state with the given OAS policy.
    fn make_accel_state(policy: crate::SmmuOasPolicy) -> Arc<SmmuSharedState> {
        let gm = GuestMemory::allocate(0x1000);
        SmmuSharedState::new(gm, 40, policy, true, None, None)
    }

    #[test]
    fn resolve_host_caps_accepts_compatible_host() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        state.resolve_host_caps(compatible_host_caps()).unwrap();
    }

    #[test]
    fn resolve_host_caps_auto_adopts_host_oas() {
        let state = make_accel_state(crate::SmmuOasPolicy::Auto);
        let caps = crate::HostSmmuCaps {
            oas: Ips::IPS_48,
            ..compatible_host_caps()
        };
        state.resolve_host_caps(caps).unwrap();
        assert_eq!(state.oas_bits(), 48);
    }

    #[test]
    fn resolve_host_caps_rejects_fixed_oas_above_host() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(52));
        let caps = crate::HostSmmuCaps {
            oas: Ips::IPS_44,
            ..compatible_host_caps()
        };
        let err = state.resolve_host_caps(caps).unwrap_err().to_string();
        assert!(err.contains("exceeds host SMMU OAS"), "{err}");
    }

    #[test]
    fn resolve_host_caps_rejects_no_aarch64() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        // AArch32-only host (TTF bit for AArch64 not set).
        let caps = crate::HostSmmuCaps {
            ttf: registers::Idr0Ttf::new().with_aarch32(true),
            ..compatible_host_caps()
        };
        let err = state.resolve_host_caps(caps).unwrap_err().to_string();
        assert!(err.contains("AArch64"), "{err}");
    }

    #[test]
    fn resolve_host_caps_accepts_aarch32_and_aarch64_host() {
        // A host advertising both formats supports AArch64 — must be accepted.
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        let caps = crate::HostSmmuCaps {
            ttf: registers::Idr0Ttf::new()
                .with_aarch32(true)
                .with_aarch64(true),
            ..compatible_host_caps()
        };
        state.resolve_host_caps(caps).unwrap();
    }

    #[test]
    fn resolve_host_caps_rejects_big_endian_only_host() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        let caps = crate::HostSmmuCaps {
            ttendian: registers::Idr0TtEndian::BE,
            ..compatible_host_caps()
        };
        let err = state.resolve_host_caps(caps).unwrap_err().to_string();
        assert!(err.contains("little-endian"), "{err}");
    }

    #[test]
    fn resolve_host_caps_accepts_mixed_endian_host() {
        // Mixed-endian host supports little-endian — must be accepted.
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        let caps = crate::HostSmmuCaps {
            ttendian: registers::Idr0TtEndian::MIXED,
            ..compatible_host_caps()
        };
        state.resolve_host_caps(caps).unwrap();
    }

    #[test]
    fn resolve_host_caps_rejects_no_gran4k() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        let caps = crate::HostSmmuCaps {
            gran4k: false,
            ..compatible_host_caps()
        };
        let err = state.resolve_host_caps(caps).unwrap_err().to_string();
        assert!(err.contains("4KB translation granule"), "{err}");
    }

    #[test]
    fn resolve_host_caps_rejects_second_device_with_different_caps() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        state.resolve_host_caps(compatible_host_caps()).unwrap();
        // A second device backed by a different physical SMMU (different OAS).
        let other = crate::HostSmmuCaps {
            oas: Ips::IPS_44,
            ..compatible_host_caps()
        };
        let err = state.resolve_host_caps(other).unwrap_err().to_string();
        assert!(
            err.contains("cannot be backed by two physical SMMUs"),
            "{err}"
        );
    }

    #[test]
    fn resolve_host_caps_accepts_second_device_with_identical_caps() {
        let state = make_accel_state(crate::SmmuOasPolicy::Fixed(40));
        state.resolve_host_caps(compatible_host_caps()).unwrap();
        // Same caps again (another device behind the same physical SMMU).
        state.resolve_host_caps(compatible_host_caps()).unwrap();
    }

    // =========================================================================
    // Disabled-state policy (GBPA.ABORT) tests
    // =========================================================================

    /// Non-accel: while the SMMU is disabled, DMA bypasses (IOVA = GPA) when
    /// `GBPA.ABORT=0`.
    #[test]
    fn test_disabled_bypass_when_gbpa_abort_clear() {
        let gm = GuestMemory::allocate(0x60_0000);
        let data = b"disabled-bypass";
        gm.write_at(0x3000, data).unwrap();

        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            false,
            None,
            None,
        );
        // Disabled with GBPA.ABORT=0 (the reset default).
        state.set_gbpa_abort(false);
        // Enable the EVTQ so an (unexpected) abort would be observable.
        state.set_evtq_config(EVTQ_BASE, EVTQ_LOG2SIZE);
        state.set_evtq_enabled(true);

        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();
        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        let mut buf = vec![0u8; data.len()];
        translating_gm.read_at(0x3000, &mut buf).unwrap();
        assert_eq!(&buf, data);
        assert_eq!(evtq_event_count(&state), 0);
    }

    /// Non-accel: while the SMMU is disabled, DMA aborts when `GBPA.ABORT=1`.
    /// Per SMMUv3 a global abort generates **no** event record (there is no
    /// stream context to fault against), so the EVTQ stays empty even though
    /// it is enabled.
    #[test]
    fn test_disabled_abort_when_gbpa_abort_set() {
        let gm = GuestMemory::allocate(0x60_0000);

        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            false,
            None,
            None,
        );
        // Disabled with GBPA.ABORT=1.
        state.set_gbpa_abort(true);
        state.set_evtq_config(EVTQ_BASE, EVTQ_LOG2SIZE);
        state.set_evtq_enabled(true);

        let bus_range = make_bus_range();
        let mock_msi = MockSignalMsi::new();
        let (translating_gm, _msi) =
            device_context(&state, bus_range, TEST_STREAM_ID_BASE, &gm, mock_msi);

        let mut buf = vec![0u8; 4];
        translating_gm.read_at(0x3000, &mut buf).unwrap_err();
        // A global (GBPA) abort generates no event record.
        assert_eq!(evtq_event_count(&state), 0);
    }

    // =========================================================================
    // current_stream_config tests
    // =========================================================================

    #[test]
    fn test_current_stream_config_disabled_bypass() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            true,
            None,
            None,
        );
        // Disabled, GBPA.ABORT=0 → Bypass, regardless of SID.
        state.set_gbpa_abort(false);
        assert_eq!(state.current_stream_config(0), StreamConfig::Bypass);
        assert_eq!(state.current_stream_config(0x1234), StreamConfig::Bypass);
    }

    #[test]
    fn test_current_stream_config_disabled_abort() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            true,
            None,
            None,
        );
        // Disabled, GBPA.ABORT=1 → Abort, regardless of SID.
        state.set_gbpa_abort(true);
        assert_eq!(state.current_stream_config(0), StreamConfig::Abort);
        assert_eq!(state.current_stream_config(0x1234), StreamConfig::Abort);
    }

    #[test]
    fn test_current_stream_config_enabled_reads_ste() {
        let gm = GuestMemory::allocate(0x60_0000);
        let sid = expected_sid();
        let state = make_shared_state(&gm);

        // Valid S1_TRANS STE → Translate, carrying this SID.
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        assert!(matches!(
            state.current_stream_config(sid),
            StreamConfig::Translate { sid: s, .. } if s == sid
        ));

        // Bypass STE → Bypass.
        write_ste(&gm, sid, &make_bypass_ste());
        assert_eq!(state.current_stream_config(sid), StreamConfig::Bypass);

        // Abort STE → Abort.
        write_ste(&gm, sid, &make_abort_ste());
        assert_eq!(state.current_stream_config(sid), StreamConfig::Abort);

        // Invalid STE (V=0) → Abort.
        write_ste(
            &gm,
            sid,
            &Ste {
                qw0: SteDw0::new().with_v(false),
                qw1: SteDw1::new(),
                _qw2_7: [0; 6],
            },
        );
        assert_eq!(state.current_stream_config(sid), StreamConfig::Abort);
    }

    #[test]
    fn test_current_stream_config_out_of_range_sid_aborts() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_shared_state(&gm);
        // strtab has 2^STRTAB_LOG2SIZE entries; an SID past the end aborts.
        let oob_sid = 1u32 << STRTAB_LOG2SIZE;
        assert_eq!(state.current_stream_config(oob_sid), StreamConfig::Abort);
    }

    // =========================================================================
    // register_accel_device initial-policy tests
    // =========================================================================

    /// A mock accel backend that records the configs applied to it.
    struct MockBackend {
        configs: Mutex<Vec<StreamConfig>>,
    }

    impl MockBackend {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                configs: Mutex::new(Vec::new()),
            })
        }

        fn take(&self) -> Vec<StreamConfig> {
            std::mem::take(&mut *self.configs.lock())
        }
    }

    impl AcceleratedStreamBackend for MockBackend {
        fn set_stream_config(&self, config: StreamConfig) -> anyhow::Result<()> {
            self.configs.lock().push(config);
            Ok(())
        }

        fn on_tlbi(&self, _cmd_bytes: &[u8; 16]) -> anyhow::Result<()> {
            Ok(())
        }
    }

    fn make_accel_shared(gm: &GuestMemory) -> Arc<SmmuSharedState> {
        SmmuSharedState::new(
            gm.clone(),
            40,
            crate::SmmuOasPolicy::Fixed(40),
            true,
            None,
            None,
        )
    }

    /// Registering a device while the SMMU is disabled (GBPA.ABORT=0) applies
    /// Bypass immediately, even before the bus is assigned.
    #[test]
    fn test_register_applies_bypass_when_disabled() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_accel_shared(&gm);
        state.set_gbpa_abort(false);

        let backend = MockBackend::new();
        // Bus not yet assigned.
        let bus_range = AssignedBusRange::new();
        state.register_accel_device(bus_range, TEST_STREAM_ID_BASE, backend.clone());

        let applied = backend.take();
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0], StreamConfig::Bypass);
    }

    /// Registering a device while the SMMU is disabled (GBPA.ABORT=1) applies
    /// Abort immediately.
    #[test]
    fn test_register_applies_abort_when_disabled_gbpa_abort() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_accel_shared(&gm);
        state.set_gbpa_abort(true);

        let backend = MockBackend::new();
        let bus_range = AssignedBusRange::new();
        state.register_accel_device(bus_range, TEST_STREAM_ID_BASE, backend.clone());

        let applied = backend.take();
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0], StreamConfig::Abort);
    }

    /// Registering a device while the SMMU is enabled with an assigned bus
    /// applies the stream's current STE-derived policy.
    #[test]
    fn test_register_applies_ste_policy_when_enabled() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_accel_shared(&gm);
        state.set_strtab(STRTAB_BASE, STRTAB_LOG2SIZE);
        state.set_enabled(true);

        let sid = expected_sid();
        write_ste(&gm, sid, &make_bypass_ste());

        let backend = MockBackend::new();
        let bus_range = make_bus_range(); // assigned
        state.register_accel_device(bus_range, TEST_STREAM_ID_BASE, backend.clone());

        let applied = backend.take();
        assert_eq!(applied.len(), 1);
        assert_eq!(applied[0], StreamConfig::Bypass);
    }

    /// Registering a device while the SMMU is enabled but the bus is not yet
    /// assigned leaves the device fail-closed (no initial apply); a later
    /// CFGI_STE (apply_stream_config) catches it up.
    #[test]
    fn test_register_enabled_unassigned_bus_then_cfgi() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_accel_shared(&gm);
        state.set_strtab(STRTAB_BASE, STRTAB_LOG2SIZE);
        state.set_enabled(true);

        let backend = MockBackend::new();
        let bus_range = AssignedBusRange::new(); // unassigned
        state.register_accel_device(bus_range.clone(), TEST_STREAM_ID_BASE, backend.clone());
        // No config applied yet (fail-closed / detached).
        assert!(backend.take().is_empty());

        // Guest assigns the bus and programs the STE, then issues CFGI_STE.
        bus_range.set_bus_range(TEST_BUS, TEST_BUS);
        let sid = expected_sid();
        write_ste(&gm, sid, &make_s1_ste(CD_BASE));
        state.apply_stream_config(sid);

        let applied = backend.take();
        assert_eq!(applied.len(), 1);
        assert!(
            matches!(applied[0], StreamConfig::Translate { sid: s, .. } if s == sid),
            "expected Translate for sid {sid:#x}, got {:?}",
            applied[0]
        );
    }

    /// apply_all_stream_configs re-drives every registered backend (used for
    /// GBPA writes, SMMUEN transitions, and CFGI_ALL).
    #[test]
    fn test_apply_all_stream_configs_redrives() {
        let gm = GuestMemory::allocate(0x60_0000);
        let state = make_accel_shared(&gm);
        state.set_strtab(STRTAB_BASE, STRTAB_LOG2SIZE);
        state.set_gbpa_abort(false);

        let backend = MockBackend::new();
        let bus_range = make_bus_range();
        state.register_accel_device(bus_range, TEST_STREAM_ID_BASE, backend.clone());
        // Initial register applied Bypass (disabled, GBPA.ABORT=0).
        assert_eq!(backend.take().last().copied(), Some(StreamConfig::Bypass));

        // Enable the SMMU and program an abort STE, then re-drive.
        let sid = expected_sid();
        write_ste(&gm, sid, &make_abort_ste());
        state.set_enabled(true);
        state.apply_all_stream_configs();

        let applied = backend.take();
        assert_eq!(applied.last().copied(), Some(StreamConfig::Abort));
    }
}
