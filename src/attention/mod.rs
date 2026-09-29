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

/// Bidirectional scaled dot-product attention over `[batch, length, heads,
/// dimension]` tensors.
///
/// `lengths[b]` valid tokens of sequence `b` attend to each other; keys past
/// that length are masked and padded query rows produce zeros.
pub fn attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    lengths: &[usize],
    window: AttentionWindow,
) -> Result<Tensor> {
    let &[batch, length, heads, dimension] = query.shape() else {
        return Err(shape_error(query, query));
    };
    for tensor in [key, value] {
        if tensor.shape() != query.shape() {
            return Err(shape_error(query, tensor));
        }
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
    let geometry = block::Geometry { length, heads, dimension, window };
    let heads_out: Vec<Vec<f32>> = (0..batch * heads)
        .into_par_iter()
        .map(|index| {
            let (sequence, head) = (index / heads, index % heads);
            let offset = (sequence * length * heads + head) * dimension;
            block::head(
                &geometry,
                &block::HeadInput {
                    query: &query.data()[offset..],
                    key: &key.data()[offset..],
                    value: &value.data()[offset..],
                    valid: lengths[sequence],
                },
            )
        })
        .collect();
    let mut output = Tensor::zeros(query.shape().to_vec());
    for (index, head_out) in heads_out.iter().enumerate() {
        let (sequence, head) = (index / heads, index % heads);
        for (token, row) in head_out.chunks_exact(dimension).enumerate() {
            let start = ((sequence * length + token) * heads + head) * dimension;
            output.data_mut()[start..start + dimension].copy_from_slice(row);
        }
    }
    Ok(output)
}

fn shape_error(query: &Tensor, actual: &Tensor) -> Error {
    Error::Shape {
        operation: "attention [batch, length, heads, dimension]",
        expected: query.shape().to_vec(),
        actual: actual.shape().to_vec(),
    }
}

#[cfg(test)]
mod tests;
