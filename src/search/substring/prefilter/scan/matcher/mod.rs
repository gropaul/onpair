// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage one: codes in, bit mask out.

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx512vbmi"))]
mod bitmap12;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod eq_or;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod nibble_n8;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod range;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
mod shared;
mod table;
#[cfg(test)]
mod tests;

use super::{Block, Mask};
use crate::search::substring::prefilter::ProbeCover;

#[cfg(all(test, target_arch = "x86_64", target_feature = "avx512vbmi"))]
pub(in crate::search::substring::prefilter::scan) use bitmap12::{Bitmap12, fits as fits_bitmap12};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::prefilter::scan) use eq_or::EqOr;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::prefilter::scan) use nibble_n8::{
    MAX_BATCHES, NibbleN8, PER_BATCH,
};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::prefilter::scan) use range::Range;
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
pub(in crate::search::substring::prefilter::scan) use shared::available as simd_available;
pub(in crate::search::substring::prefilter::scan) use table::Table;

/// Bit `i` of `bits` is set iff the cover admits `codes[i]`. See `README.md`.
pub(in crate::search::substring::prefilter::scan) trait Matcher:
    Sized
{
    /// Callers check `policy::takes` first.
    fn new(cover: &ProbeCover) -> Self;

    /// Fills `bits` for one block. `false` promises an empty mask, so stage two
    /// is skipped; `true` promises nothing.
    fn check(&self, codes: &Block, bits: &mut Mask) -> bool;
}
