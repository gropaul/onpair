// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A plan as the two type parameters [`both_stages`] wants.

use super::matcher::{self, Matcher};
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
use super::matcher::{MAX_BATCHES, PER_BATCH};
use super::policy::{Match, Plan, Resolve};
use super::{Check, both_stages, resolver};
use crate::core::offset::Offset;
use crate::core::types::Token;
use crate::search::substring::prefilter::ProbeCover;

/// The resolver half of the dispatch.
fn with_resolver<O: Offset, M: Matcher>(
    plan: Plan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    match plan.resolver {
        Resolve::LinearSeek => {
            both_stages::<M, resolver::LinearSeek<'_, O>>(cover, codes, row_offsets, check, out)
        }
        Resolve::GallopSeek => {
            both_stages::<M, resolver::GallopSeek<'_, O>>(cover, codes, row_offsets, check, out)
        }
    }
}

/// The flag half: the same kernel compiled with the pack skipped, `S`, and
/// without, `P`.
#[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
fn with_skip<O: Offset, S: Matcher, P: Matcher>(
    plan: Plan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    if plan.skip {
        with_resolver::<O, S>(plan, cover, codes, row_offsets, check, out)
    } else {
        with_resolver::<O, P>(plan, cover, codes, row_offsets, check, out)
    }
}

/// The planned kernel pair as the type parameters [`both_stages`] wants.
pub(super) fn run<O: Offset>(
    plan: Plan,
    cover: &ProbeCover,
    codes: &[Token],
    row_offsets: &[O],
    check: Check<'_>,
    out: &mut Vec<usize>,
) {
    match plan.matcher {
        Match::Table => {
            with_resolver::<O, matcher::Table>(plan, cover, codes, row_offsets, check, out)
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::EqOr => with_skip::<O, matcher::EqOr<true>, matcher::EqOr<false>>(
            plan,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::Range => with_skip::<O, matcher::Range<true>, matcher::Range<false>>(
            plan,
            cover,
            codes,
            row_offsets,
            check,
            out,
        ),
        #[cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]
        Match::NibbleN8K => {
            macro_rules! at {
                ($($n:literal)*) => {
                    match cover.points().len().div_ceil(PER_BATCH) {
                        $($n => with_skip::<O, matcher::NibbleN8<$n, true>, matcher::NibbleN8<$n, false>>(
                            plan, cover, codes, row_offsets, check, out,
                        ),)*
                        _ => unreachable!("`takes` admits at most MAX_BATCHES batches"),
                    }
                };
            }
            const _: () = assert!(MAX_BATCHES == 16, "one arm below per batch count");
            at!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
        }
    }
}
