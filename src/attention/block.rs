use super::AttentionWindow;
use crate::{
    fastmath,
    linear::{Matrix, MatrixMut, Threads, Write, multiply},
    numeric::float,
};

const QUERY_BLOCK: usize = 256;

pub struct Geometry {
    pub length: usize,
    pub heads: usize,
    pub dimension: usize,
    pub window: AttentionWindow,
}

/// One head of one sequence; each slice starts at token 0 of that head and
/// keeps the fused `[length, 3 × heads × dimension]` interleaving.
pub struct HeadInput<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub valid: usize,
}

/// Returns the `[valid, dimension]` attention output of one head, padded with
/// zero rows to `[length, dimension]`.
pub fn head(geometry: &Geometry, input: &HeadInput<'_>) -> Vec<f32> {
    let dimension = geometry.dimension;
    let stride = 3 * geometry.heads * dimension;
    let scale = 1.0 / float(dimension).sqrt();
    let mut output = vec![0.0; geometry.length * dimension];
    let mut first = 0;
    while first < input.valid {
        let last = (first + QUERY_BLOCK).min(input.valid);
        let (key_first, key_last) = match geometry.window {
            AttentionWindow::Full => (0, input.valid),
            AttentionWindow::Band { radius } => {
                (first.saturating_sub(radius), (last + radius).min(input.valid))
            },
        };
        let (queries, keys) = (last - first, key_last - key_first);
        let mut scores = vec![0.0; queries * keys];
        multiply(
            &mut MatrixMut::row_major(&mut scores, queries, keys),
            Matrix::strided(&input.query[first * stride..], queries, dimension, stride, 1),
            Matrix::strided(&input.key[key_first * stride..], dimension, keys, 1, stride),
            Threads::Caller,
            Write::Overwrite,
        );
        for (row, weights) in scores.chunks_exact_mut(keys).enumerate() {
            let (low, high) = key_range(geometry.window, first + row, input.valid);
            softmax_row(weights, scale, low - key_first..high - key_first);
        }
        multiply(
            &mut MatrixMut::row_major(
                &mut output[first * dimension..last * dimension],
                queries,
                dimension,
            ),
            Matrix::row_major(&scores, queries, keys),
            Matrix::strided(&input.value[key_first * stride..], keys, dimension, stride, 1),
            Threads::Caller,
            Write::Overwrite,
        );
        first = last;
    }
    output
}

/// Keys `[low, high)` a query may attend to; banded windows are contiguous.
fn key_range(window: AttentionWindow, query: usize, valid: usize) -> (usize, usize) {
    match window {
        AttentionWindow::Full => (0, valid),
        AttentionWindow::Band { radius } => {
            (query.saturating_sub(radius), (query + radius + 1).min(valid))
        },
    }
}

/// Replaces scaled scores by softmax weights over `keys`; every other column
/// gets weight 0.
fn softmax_row(row: &mut [f32], scale: f32, keys: std::ops::Range<usize>) {
    let (start, end) = (keys.start, keys.end);
    row[..start].fill(0.0);
    row[end..].fill(0.0);
    let window = &mut row[start..end];
    let maximum = window.iter().fold(f32::NEG_INFINITY, |maximum, &score| maximum.max(score));
    let mut total = 0.0;
    for score in window.iter_mut() {
        *score = fastmath::exp((*score - maximum) * scale);
        total += *score;
    }
    let inverse = 1.0 / total;
    for score in window.iter_mut() {
        *score *= inverse;
    }
}
