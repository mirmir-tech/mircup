use crate::{LayerNorm, Result, Tensor};

#[test]
fn normalizes_each_row_without_bias() -> Result<()> {
    let norm = LayerNorm::new(vec![1.0, 1.0], None, 0.0)?;
    let output = norm.forward(&Tensor::new(vec![2, 2], vec![1.0, 3.0, -2.0, -2.0])?)?;
    assert_eq!(output.data()[..2], [-1.0, 1.0]);
    assert!(output.data()[2..].iter().all(|value| value.is_nan()));
    Ok(())
}

#[test]
fn applies_weight_and_bias() -> Result<()> {
    let norm = LayerNorm::new(vec![2.0, 3.0], Some(vec![0.5, -0.5]), 1e-5)?;
    let output = norm.forward(&Tensor::new(vec![1, 2], vec![0.0, 4.0])?)?;
    assert!((output.data()[0] - (-2.0 + 0.5)).abs() < 1e-4);
    assert!((output.data()[1] - (3.0 - 0.5)).abs() < 1e-4);
    Ok(())
}

#[test]
fn rejects_mismatched_bias() {
    assert!(LayerNorm::new(vec![1.0; 3], Some(vec![0.0; 2]), 1e-5).is_err());
}
