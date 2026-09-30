use crate::{AttentionWindow, Result, Tensor, attention, attention_at, numeric::float};

fn pattern(shape: &[usize], seed: usize) -> Result<Tensor> {
    let count = shape.iter().product::<usize>();
    let data = (0..count)
        .map(|index| (float((index * 7 + seed * 13) % 17) - 8.0) * 0.1)
        .collect();
    Tensor::new(shape.to_vec(), data)
}

/// Direct per-element evaluation of masked softmax attention.
fn naive(
    [query, key, value]: [&Tensor; 3],
    lengths: &[usize],
    window: AttentionWindow,
) -> Vec<f32> {
    let &[batch, length, heads, dimension] = query.shape() else {
        return Vec::new();
    };
    let at = |tensor: &Tensor, b: usize, t: usize, h: usize, d: usize| {
        tensor.data()[((b * length + t) * heads + h) * dimension + d]
    };
    let mut output = vec![0.0; query.data().len()];
    for b in 0..batch {
        for h in 0..heads {
            for i in 0..lengths[b] {
                let keys: Vec<usize> = (0..lengths[b])
                    .filter(|&j| match window {
                        AttentionWindow::Full => true,
                        AttentionWindow::Band { radius } => i.abs_diff(j) <= radius,
                    })
                    .collect();
                let scores: Vec<f32> = keys
                    .iter()
                    .map(|&j| {
                        (0..dimension)
                            .map(|d| at(query, b, i, h, d) * at(key, b, j, h, d))
                            .sum::<f32>()
                            / float(dimension).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let weights: Vec<f32> = scores.iter().map(|s| (s - maximum).exp()).collect();
                let total: f32 = weights.iter().sum();
                for d in 0..dimension {
                    output[((b * length + i) * heads + h) * dimension + d] = keys
                        .iter()
                        .zip(&weights)
                        .map(|(&j, w)| w / total * at(value, b, j, h, d))
                        .sum();
                }
            }
        }
    }
    output
}

/// Interleaves per-token query, key and value heads into one fused tensor.
fn fused(parts: [&Tensor; 3]) -> Result<Tensor> {
    let &[batch, length, heads, dimension] = parts[0].shape() else {
        return Ok(Tensor::zeros(vec![0]));
    };
    let token = heads * dimension;
    let mut data = Vec::with_capacity(3 * parts[0].data().len());
    for index in 0..batch * length {
        for part in parts {
            data.extend_from_slice(&part.data()[index * token..(index + 1) * token]);
        }
    }
    Tensor::new(vec![batch, length, 3 * token], data)
}

fn check(window: AttentionWindow, lengths: &[usize], length: usize) -> Result<()> {
    let shape = [lengths.len(), length, 3, 8];
    let (query, key, value) = (pattern(&shape, 1)?, pattern(&shape, 2)?, pattern(&shape, 3)?);
    let actual = attention(&fused([&query, &key, &value])?, 3, lengths, window)?;
    let expected = naive([&query, &key, &value], lengths, window);
    for (index, (actual, expected)) in actual.data().iter().zip(&expected).enumerate() {
        assert!((actual - expected).abs() < 1e-5, "{index}: {actual} != {expected}");
    }
    Ok(())
}

#[test]
fn full_attention_matches_the_direct_formula_across_blocks() -> Result<()> {
    check(AttentionWindow::Full, &[150, 97], 150)
}

#[test]
fn band_attention_matches_the_direct_formula_across_blocks() -> Result<()> {
    check(AttentionWindow::Band { radius: 5 }, &[150, 70], 150)
}

#[test]
fn padded_query_rows_are_zero() -> Result<()> {
    let tensor = pattern(&[1, 4, 6], 1)?;
    let output = attention(&tensor, 1, &[2], AttentionWindow::Full)?;
    assert!(output.data()[4..].iter().all(|value| *value == 0.0));
    Ok(())
}

#[test]
fn rejects_lengths_beyond_the_padded_length() {
    let tensor = Tensor::zeros(vec![1, 2, 6]);
    assert!(attention(&tensor, 1, &[3], AttentionWindow::Full).is_err());
    assert!(attention(&Tensor::zeros(vec![1, 2, 5]), 1, &[2], AttentionWindow::Full).is_err());
}

#[test]
fn selected_queries_match_the_full_attention_rows() -> Result<()> {
    let shape = [2, 20, 3, 8];
    let (query, key, value) = (pattern(&shape, 1)?, pattern(&shape, 2)?, pattern(&shape, 3)?);
    let qkv = fused([&query, &key, &value])?;
    for window in [AttentionWindow::Full, AttentionWindow::Band { radius: 3 }] {
        let full = attention(&qkv, 3, &[20, 11], window)?;
        let queries = vec![vec![0, 7, 19], vec![10, 2]];
        let selected = attention_at(&qkv, 3, &[20, 11], window, &queries)?;
        let expected: Vec<usize> = vec![0, 7, 19, 20 + 10, 20 + 2];
        let reference = full.gather_rows(&expected)?;
        for (selected, full) in selected.data().iter().zip(reference.data()) {
            assert!((selected - full).abs() < 1e-6, "{selected} != {full}");
        }
    }
    assert!(attention_at(&qkv, 3, &[20, 11], AttentionWindow::Full, &[vec![], vec![11]]).is_err());
    Ok(())
}
