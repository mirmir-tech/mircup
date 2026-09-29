mod block;

use rayon::prelude::*;

use crate::{Error, Result, Tensor};

/// Which keys a query may attend to, in addition to the sequence length mask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionWindow {
    /// Every valid key of the sequence.
    Full,
    /// Keys `j` with `|i - j| <= radius` for query `i`.
    Band { radius: usize },
}

/// Bidirectional scaled dot-product attention over a fused projection.
///
/// `qkv` is `[batch, length, 3 × heads × dimension]`: every token holds its
/// query heads, then key heads, then value heads, each head `dimension`
/// wide. The result is `[batch, length, heads × dimension]`. The first
/// `lengths[b]` tokens of sequence `b` attend to each other; keys past that
/// length are masked and padded query rows produce zeros.
pub fn attention(
    qkv: &Tensor,
    heads: usize,
    lengths: &[usize],
    window: AttentionWindow,
) -> Result<Tensor> {
    let &[batch, length, width] = qkv.shape() else {
        return Err(shape_error(qkv, heads));
    };
    if heads == 0 || !width.is_multiple_of(3 * heads) {
        return Err(shape_error(qkv, heads));
    }
    if lengths.len() != batch {
        return Err(Error::Shape {
            operation: "attention lengths",
            expected: vec![batch],
            actual: vec![lengths.len()],
        });
    }
    if let Some(&longest) = lengths.iter().find(|&&valid| valid > length) {
        return Err(Error::SequenceLength { length: longest, padded: length });
    }
    let dimension = width / (3 * heads);
    let geometry = block::Geometry { length, heads, dimension, window };
    let heads_out: Vec<Vec<f32>> = (0..batch * heads)
        .into_par_iter()
        .map(|index| {
            let (sequence, head) = (index / heads, index % heads);
            let token = sequence * length * width;
            let data = qkv.data();
            block::head(
                &geometry,
                &block::HeadInput {
                    query: &data[token + head * dimension..],
                    key: &data[token + (heads + head) * dimension..],
                    value: &data[token + (2 * heads + head) * dimension..],
                    valid: lengths[sequence],
                },
            )
        })
        .collect();
    let hidden = heads * dimension;
    let mut output = Tensor::zeros(vec![batch, length, hidden]);
    for (index, head_out) in heads_out.iter().enumerate() {
        let (sequence, head) = (index / heads, index % heads);
        for (token, row) in head_out.chunks_exact(dimension).enumerate() {
            let start = (sequence * length + token) * hidden + head * dimension;
            output.data_mut()[start..start + dimension].copy_from_slice(row);
        }
    }
    Ok(output)
}

fn shape_error(qkv: &Tensor, heads: usize) -> Error {
    Error::Shape {
        operation: "attention [batch, length, 3 × heads × dimension]",
        expected: vec![0, 0, 3 * heads],
        actual: qkv.shape().to_vec(),
    }
}

#[cfg(test)]
mod tests;
