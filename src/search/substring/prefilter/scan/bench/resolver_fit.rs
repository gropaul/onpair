// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stage two: the resolver against one mask per needle set, built once from
//! the cover itself so no matcher runs while the clock does, and the two
//! constants of `policy::stage_two_ns` fitted to what that measured.
//!
//! The sweep's axes are the mask's hit density and the row length. Rates are
//! over the code stream, so the two halves of one scan add up; in the
//! sparsest bin, where a mask holds a handful of bits, a rate says "no work"
//! rather than a speed, and the fit leaves such rows out.
//!
//! What the resolver's time is linear in:
//!
//! ```text
//! words     mask words in the blocks that hit
//! searched  emitted · log2(1 + crossed / emitted), one halving each
//! ```
//!
//! CSVs from before the linear seek was removed hold its rows beside these;
//! the fit reads only `gallop_seek` rows.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::super::policy::stage_two_ns;
use super::super::{BLOCK_SIZE_SCAN_BLOCKS, blocks, clear_from, resolver};
use super::loader::{NeedleSet, SAMPLE, load_corpus, load_needles, paths, row_layer};
use super::utils::{error, file_name, lstsq, read_csv, write_csv};
use super::{best, machine, resolve_stream};
use crate::core::types::Token;
use crate::search::substring::prefilter::ProbeCover;

/// Streams for stage two, picked for their mean row length: about seven codes
/// a row and about eighty. Only one encoding of each, `onpair16`: a wider code
/// reaches stage two through the codes a row holds and nothing else, and
/// [`MERGES`] sweeps that anyway.
const STREAMS: &[&str] = &["imdb/name/name_1m", "ch/hits/URL_1m"];

/// Row layers per stream: every `factor` rows fused into one. The mask and
/// the length distribution stay as they are, so this is the row count on its
/// own, from hundreds of rows to a block down to one row over several.
const MERGES: &[usize] = &[1, 4, 16, 64];

/// Needle counts to draw masks from. K reaches stage two only through the
/// density of the mask it leaves behind, so two of the eight bins span it.
const COUNTS: &[usize] = &[1, 16];

/// Codes per measurement, 8 MiB at two bytes a code, the working set stage
/// one sweeps, so both stages are read against one memory ceiling. A mask is
/// built for every needle set first, which is what this size costs. The IMDb
/// stream is shorter than this and runs at its own length, 2.8 Mcodes.
const CODES: usize = 1 << 22;

/// One measurement of the resolver on one mask over one row layer.
#[derive(Serialize, Deserialize)]
struct Row {
    stream: String,
    codes: usize,
    rows: usize,
    codes_per_row: usize,
    resolver: String,
    length: usize,
    count: usize,
    target: f64,
    achieved: f64,
    /// Set bits per code.
    density: f64,
    /// Share of blocks the matcher would hand over.
    blocks_hit: f64,
    /// Share of rows the mask names.
    selectivity: f64,
    gbs: f64,
    machine: String,
}

const RESOLVER: &str = "gallop_seek";

impl Row {
    /// Nanoseconds over the stream, what the fit works in.
    fn ns(&self) -> f64 {
        self.codes as f64 / self.gbs
    }

    fn words(&self) -> f64 {
        64.0 * self.blocks_hit * self.codes as f64 / BLOCK_SIZE_SCAN_BLOCKS as f64
    }

    fn emitted(&self) -> f64 {
        self.selectivity * self.rows as f64
    }

    /// The cursor stops at the last hit rather than the last row.
    fn crossed(&self) -> f64 {
        self.rows as f64 * self.emitted() / (self.emitted() + 1.0)
    }

    /// Rows with at least one bit and one row to emit: a mask with neither
    /// is no work, and its rate says nothing about the resolver.
    fn has_work(&self) -> bool {
        self.density * self.codes as f64 >= 1.0 && self.emitted() >= 1.0
    }

    /// The row in the model's terms.
    fn terms(&self) -> [f64; 2] {
        let (emitted, crossed) = (self.emitted(), self.crossed());
        [self.words(), emitted * (1.0 + crossed / emitted).log2()]
    }

    fn model(&self) -> f64 {
        stage_two_ns(self.words(), self.emitted(), self.crossed())
    }
}

/// One bit per code for the whole stream, straight off the cover's own
/// definition of a hit and laid out as the blocks the seam hands over.
fn mask_of(set: &NeedleSet, codes: &[Token]) -> Vec<u64> {
    let cover = ProbeCover::new(set.needles.to_vec(), vec![]);
    let mut mask = Vec::with_capacity(codes.len() / 64);
    let mut bits = [0u64; BLOCK_SIZE_SCAN_BLOCKS / 64];
    blocks(codes, &mut |block, _, valid| {
        bits.fill(0);
        for (at, &code) in block.iter().enumerate() {
            if cover.contains(code) {
                bits[at / 64] |= 1 << (at % 64);
            }
        }
        clear_from(&mut bits, valid);
        mask.extend_from_slice(&bits);
    });
    mask
}

/// The blocks a matcher would report non-empty. Finding them is stage
/// one's work, so it happens once and outside the timed run.
fn hit_blocks(mask: &[u64]) -> Vec<usize> {
    mask.chunks_exact(BLOCK_SIZE_SCAN_BLOCKS / 64)
        .enumerate()
        .filter(|(_, bits)| bits.iter().any(|&set| set != 0))
        .map(|(block, _)| block)
        .collect()
}

/// Every `factor` rows of the layer fused into one.
fn merge_rows(row_offsets: &[u32], factor: usize) -> Vec<u32> {
    let mut merged: Vec<u32> = row_offsets.iter().copied().step_by(factor).collect();
    let last = *row_offsets.last().unwrap();
    if *merged.last().unwrap() < last {
        merged.push(last);
    }
    merged
}

/// The resolver over the stream, checked against the rows the mask names
/// before its time counts.
fn resolve(mask: &[u64], hit: &[usize], row_offsets: &[u32], expected: &[usize]) -> f64 {
    let mut out = Vec::new();
    let seconds = best(&mut || resolve_stream(mask, hit, row_offsets, &mut out));
    assert_eq!(out, *expected, "differs from the rows the mask names");
    seconds
}

/// The resolver on each mask at every row count of [`MERGES`].
fn measure(stream: &str, machine: &str, out: &mut Vec<Row>) {
    let (corpus_path, needles_path) = paths(stream, "onpair16");
    let corpus = load_corpus(&corpus_path);
    // Whole blocks only, so the mask is exactly the blocks it holds.
    let codes = &corpus.codes
        [..CODES.min(corpus.codes.len()) / BLOCK_SIZE_SCAN_BLOCKS * BLOCK_SIZE_SCAN_BLOCKS];
    let layer = row_layer(&corpus.row_offsets, codes.len());
    let sets: Vec<NeedleSet> = load_needles(&needles_path)
        .into_iter()
        .filter(|set| set.sample == SAMPLE && COUNTS.contains(&set.count))
        .collect();
    println!(
        "\n{stream} onpair16: {} codes, {} rows, {} needle sets",
        codes.len(),
        layer.len() - 1,
        sets.len()
    );
    let mut empty = 0;
    for set in &sets {
        let mask = mask_of(set, codes);
        let hit = hit_blocks(&mask);
        let bits: u32 = mask.iter().map(|word| word.count_ones()).sum();
        // No bit in the prefix is no work at all, and no rate: the 0% bin
        // of a catalog is built to hit nothing, and the top bins hold the
        // sets their corpus could not deliver.
        if hit.is_empty() {
            empty += 1;
            continue;
        }
        for &factor in MERGES {
            let row_offsets = merge_rows(&layer, factor);
            let rows = row_offsets.len() - 1;
            let expected = resolver::expected_rows(&mask, &row_offsets);
            let seconds = resolve(&mask, &hit, &row_offsets, &expected);
            out.push(Row {
                stream: stream.to_string(),
                codes: codes.len(),
                rows,
                codes_per_row: codes.len() / rows,
                resolver: RESOLVER.to_string(),
                length: 1,
                count: set.count,
                target: set.target,
                achieved: set.achieved,
                density: f64::from(bits) / codes.len() as f64,
                blocks_hit: hit.len() as f64 / (mask.len() / (BLOCK_SIZE_SCAN_BLOCKS / 64)) as f64,
                selectivity: expected.len() as f64 / rows as f64,
                gbs: codes.len() as f64 / seconds / 1e9,
                machine: machine.to_string(),
            });
        }
    }
    if empty > 0 {
        println!("  {empty} of {} sets never hit the prefix", sets.len());
    }
}

/// The model fitted to `rows`, per machine, against the model compiled in.
fn fit(rows: &[Row], source: &str) {
    let mut group: BTreeMap<&str, Vec<&Row>> = BTreeMap::new();
    for row in rows
        .iter()
        .filter(|row| row.resolver == RESOLVER && row.has_work())
    {
        group.entry(&row.machine).or_default().push(row);
    }
    let mut fitted: BTreeMap<&str, [f64; 2]> = BTreeMap::new();
    for (&machine, row) in &group {
        let point: Vec<([f64; 2], f64)> = row.iter().map(|row| (row.terms(), row.ns())).collect();
        let beta @ [word, step] = lstsq(&point);
        if beta.iter().any(|&b| b < 0.0) {
            println!(
                "\n{machine}: a term fitted negative, so the fit is borrowing from a collinear one"
            );
        }
        let fit = |row: &Row| {
            row.terms()
                .iter()
                .zip(beta)
                .map(|(x, b)| x * b)
                .sum::<f64>()
        };
        let error_of = |predict: &dyn Fn(&Row) -> f64| {
            let point: Vec<(f64, f64)> = row.iter().map(|row| (predict(row), row.ns())).collect();
            error(&point, |predicted| predicted)
        };
        println!("\n{machine}, {} rows with work in them", row.len());
        println!(
            "  {word:.2} ns per word + {step:.2} per halving, fit {:.1}%, compiled model {:.1}%",
            error_of(&fit),
            error_of(&Row::model)
        );
        fitted.insert(machine, beta);
    }
    for (&machine, [word, step]) in &fitted {
        println!("\n/// Fitted on {machine}, from {source}.");
        println!("const WORD_NS: f64 = {word:.2};");
        println!("const GALLOP_SEEK_STEP_NS: f64 = {step:.2};");
    }
}

/// Sweep, write the CSV, fit.
#[test]
#[ignore]
fn sweep() {
    let machine = machine();
    let mut rows = Vec::new();
    for &stream in STREAMS {
        measure(stream, &machine, &mut rows);
    }
    let path = write_csv("novel_resolve", &rows);
    println!("wrote {}", path.display());
    fit(&rows, &file_name(&path));
}

/// Fit the newest `novel_resolve_*.csv`, or the one `RESOLVE_CSV` names.
#[test]
#[ignore]
fn refit() {
    let (path, rows) = read_csv::<Row>("RESOLVE_CSV", "novel_resolve_");
    println!("{}\n{} rows", path.display(), rows.len());
    fit(&rows, &file_name(&path));
}
