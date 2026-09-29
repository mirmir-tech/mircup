use super::AttentionWindow;
use crate::{
    linear::{Matrix, MatrixMut, Threads, multiply},
    numeric::float,
};

const QUERY_BLOCK: usize = 64;

pub struct Geometry {
    pub length: usize,
    pub heads: usize,
    pub dimension: usize,
    pub window: AttentionWindow,
}

/// One head of one sequence; each slice starts at token 0 of that head and
/// keeps the `[length, heads, dimension]` interleaving.
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
    let stride = geometry.heads * dimension;
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
        );
        for (row, weights) in scores.chunks_exact_mut(keys).enumerate() {
            softmax_row(weights, scale, |column| {
                allowed(geometry.window, first + row, key_first + column)
            });
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
        );
        first = last;
    }
    output
}

const fn allowed(window: AttentionWindow, query: usize, key: usize) -> bool {
    match window {
        AttentionWindow::Full => true,
        AttentionWindow::Band { radius } => query.abs_diff(key) <= radius,
    }
}

/// Replaces scaled scores by softmax weights; disallowed keys get weight 0.
fn softmax_row(row: &mut [f32], scale: f32, allowed: impl Fn(usize) -> bool) {
    let mut maximum = f32::NEG_INFINITY;
    for (column, score) in row.iter_mut().enumerate() {
        if allowed(column) {
            *score *= scale;
            maximum = maximum.max(*score);
        } else {
            *score = f32::NEG_INFINITY;
        }
    }
    let mut total = 0.0;
    for score in row.iter_mut() {
        *score = if score.is_finite() {
            (*score - maximum).exp()
        } else {
            0.0
        };
        total += *score;
    }
    for score in row.iter_mut() {
        *score /= total;
    }
}
