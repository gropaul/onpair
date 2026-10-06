// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One bit per code of a 12-bit code space, the 4096 bits held in eight
//! registers: byte `c >> 3` through four 128-byte `vpermi2b` tables, picked by
//! bits 10 and 11, and bit `c & 7` through a `vpermb`. Flat in K and R. A code
//! from 4096 up never hits, so a 16-bit stream is safe; a cover is not, and
//! the caller keeps it below 4096.

use std::arch::x86_64::*;

use super::shared::{EVEN, Hits, Vectors, words};
use super::{Block, Mask, Matcher};
use crate::core::types::Token;
use crate::search::substring::prefilter::ProbeCover;

/// Codes the bitmap holds.
const CODES: usize = 1 << 12;

/// Every code the cover names is inside the bitmap.
pub(in crate::search::substring::prefilter::scan) fn fits(cover: &ProbeCover) -> bool {
    let inside = |code: Token| usize::from(code) < CODES;
    cover.points().iter().all(|&code| inside(code)) && cover.ranges().iter().all(|r| inside(r.last))
}

pub(in crate::search::substring::prefilter::scan) struct Bitmap12<
    const SKIP_MOVEMASK_IF_NO_MATCH: bool,
>([__m512i; 8]);

impl<const SKIP_MOVEMASK_IF_NO_MATCH: bool> Matcher for Bitmap12<SKIP_MOVEMASK_IF_NO_MATCH> {
    fn new(cover: &ProbeCover) -> Self {
        let mut map = [0u8; CODES / 8];
        let mut set = |code: usize| map[code >> 3] |= 1 << (code & 7);
        for &code in cover.points() {
            set(usize::from(code));
        }
        for range in cover.ranges() {
            for code in usize::from(range.begin)..=usize::from(range.last) {
                set(code);
            }
        }
        // SAFETY: each load reads 64 of the map's 512 bytes.
        Self(std::array::from_fn(|at| unsafe {
            _mm512_loadu_si512(map[64 * at..].as_ptr().cast())
        }))
    }

    fn check(&self, codes: &Block, bits: &mut Mask) -> bool {
        // SAFETY: avx512vbmi, enabled by the build and checked by `available`.
        unsafe { mask::<SKIP_MOVEMASK_IF_NO_MATCH>(&self.0, codes, bits) }
    }
}

/// Outside `check` so the closure inherits the target features and inlines.
#[target_feature(enable = "avx2")]
fn mask<const SKIP_MOVEMASK_IF_NO_MATCH: bool>(
    map: &[__m512i; 8],
    codes: &Block,
    bits: &mut Mask,
) -> bool {
    words::<SKIP_MOVEMASK_IF_NO_MATCH>(codes, bits, |codes| unsafe { probe(map, codes) })
}

#[target_feature(enable = "avx512f,avx512bw,avx512vbmi")]
unsafe fn probe(map: &[__m512i; 8], codes: Vectors) -> Hits {
    // SAFETY: EVEN is 64 bytes.
    let even = unsafe { _mm512_loadu_si512(EVEN.as_ptr().cast()) };
    let odd = _mm512_add_epi8(even, _mm512_set1_epi8(1));
    let pick = |index, lo, hi| _mm512_permutex2var_epi8(lo, index, hi);
    let low = pick(even, codes[0], codes[1]);
    let high = pick(odd, codes[0], codes[1]);
    // Bits 3..10; `vpermi2b` reads seven of them.
    let byte = pick(
        even,
        _mm512_srli_epi16::<3>(codes[0]),
        _mm512_srli_epi16::<3>(codes[1]),
    );
    let inside = _mm512_testn_epi8_mask(high, _mm512_set1_epi8(0xF0u8 as i8));
    let bit10 = _mm512_test_epi8_mask(high, _mm512_set1_epi8(0x04));
    let bit11 = _mm512_test_epi8_mask(high, _mm512_set1_epi8(0x08));
    let rows = |pair: usize| _mm512_permutex2var_epi8(map[2 * pair], byte, map[2 * pair + 1]);
    let lower = _mm512_mask_blend_epi8(bit10, rows(0), rows(1));
    let upper = _mm512_mask_blend_epi8(bit10, rows(2), rows(3));
    let row = _mm512_mask_blend_epi8(bit11, lower, upper);
    // 1 << (i & 7) at every index i.
    let want = _mm512_permutexvar_epi8(low, _mm512_set1_epi64(0x8040201008040201u64 as i64));
    _mm512_mask_test_epi8_mask(inside, row, want)
}
