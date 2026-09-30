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
    let geometry = geometry(qkv, heads, lengths, window)?;
    let (batch, length, dimension) = (lengths.len(), geometry.length, geometry.dimension);
    let width = 3 * heads * dimension;
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

/// [`attention`] evaluated only for the listed query tokens.
///
/// `queries[b]` lists valid tokens of sequence `b`. The result holds
/// `[Σ queries, heads × dimension]` rows in sequence order, then listed
/// order; keys still span every valid token.
pub fn attention_at(
    qkv: &Tensor,
    heads: usize,
    lengths: &[usize],
    window: AttentionWindow,
    queries: &[Vec<usize>],
) -> Result<Tensor> {
    let geometry = geometry(qkv, heads, lengths, window)?;
    if queries.len() != lengths.len() {
        return Err(Error::Shape {
            operation: "attention queries",
            expected: vec![lengths.len()],
            actual: vec![queries.len()],
        });
    }
    for (listed, &valid) in queries.iter().zip(lengths) {
        if let Some(&query) = listed.iter().find(|&&query| query >= valid) {
            return Err(Error::SequenceLength { length: query + 1, padded: valid });
        }
    }
    let (length, dimension) = (geometry.length, geometry.dimension);
    let width = 3 * heads * dimension;
    let heads_out: Vec<Vec<f32>> = (0..lengths.len() * heads)
        .into_par_iter()
        .map(|index| {
            let (sequence, head) = (index / heads, index % heads);
            let token = sequence * length * width;
            let data = qkv.data();
            block::selected(
                &geometry,
                &block::HeadInput {
                    query: &data[token + head * dimension..],
                    key: &data[token + (heads + head) * dimension..],
                    value: &data[token + (2 * heads + head) * dimension..],
                    valid: lengths[sequence],
                },
                &queries[sequence],
            )
        })
        .collect();
    let hidden = heads * dimension;
    let total: usize = queries.iter().map(Vec::len).sum();
    let mut output = Tensor::zeros(vec![total, hidden]);
    let mut row = 0;
    for (sequence, listed) in queries.iter().enumerate() {
        for head in 0..heads {
            let values = &heads_out[sequence * heads + head];
            for (offset, head_row) in values.chunks_exact(dimension).enumerate() {
                let start = (row + offset) * hidden + head * dimension;
                output.data_mut()[start..start + dimension].copy_from_slice(head_row);
            }
        }
        row += listed.len();
    }
    Ok(output)
}

fn geometry(
    qkv: &Tensor,
    heads: usize,
    lengths: &[usize],
    window: AttentionWindow,
) -> Result<block::Geometry> {
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
    Ok(block::Geometry {
        length,
        heads,
        dimension: width / (3 * heads),
        window,
    })
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
