// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! SMMUv3 emulator for OpenVMM.
//!
//! This crate implements an Arm SMMUv3 (System Memory Management Unit)
//! emulator, providing IOVA→GPA translation for devices behind the SMMU.

#![forbid(unsafe_code)]

mod emulator;
mod shared;
mod spec;
mod translate;

pub use emulator::HostSmmuCaps;
pub use emulator::SmmuConfig;
pub use emulator::SmmuDevice;
pub use emulator::SmmuOasPolicy;
pub use shared::AcceleratedStreamBackend;
pub use shared::SmmuSharedState;
pub use shared::SmmuSignalMsi;
pub use shared::SmmuTranslator;
pub use shared::StreamConfig;

/// Valid SMMUv3 output address sizes, in bits (IDR5.OAS encodings).
pub const VALID_OAS_BITS: [u8; 7] = [32, 36, 40, 42, 44, 48, 52];

/// Returns the smallest valid SMMUv3 OAS (in bits) whose address space
/// covers `max_addr` (an exclusive upper bound on guest physical addresses),
/// clamped to the maximum supported value of 52.
pub fn min_oas_bits_for(max_addr: u64) -> u8 {
    for &bits in &VALID_OAS_BITS {
        // `bits` is at most 52, so `1 << bits` cannot overflow `u64`.
        if max_addr <= (1u64 << bits) {
            return bits;
        }
    }
    52
}

/// Converts an SMMUv3 IDR5.OAS field encoding to a size in bits, or `None`
/// if the encoding is not recognized.
pub fn oas_bits_from_encoding(encoding: u8) -> Option<u8> {
    spec::cd::Ips(encoding).bits()
}
