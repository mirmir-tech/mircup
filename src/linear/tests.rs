use crate::{Linear, Result, Tensor, numeric::float};

#[test]
fn multiplies_by_the_transposed_weight_and_adds_bias() -> Result<()> {
    let weight = Tensor::new(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0])?;
    let linear = Linear::new(weight, Some(vec![0.5, -0.5, 0.0]))?;
    let input = Tensor::new(vec![2, 2], vec![2.0, 3.0, -1.0, 4.0])?;
    let output = linear.forward(&input)?;
    assert_eq!(output.shape(), &[2, 3]);
    assert_eq!(output.data(), &[2.5, 2.5, 5.0, -0.5, 3.5, 3.0]);
    Ok(())
}

#[test]
fn keeps_outer_dimensions() -> Result<()> {
    let linear = Linear::new(Tensor::new(vec![1, 2], vec![1.0, 1.0])?, None)?;
    let output = linear.forward(&Tensor::new(vec![2, 2, 2], vec![1.0; 8])?)?;
    assert_eq!(output.shape(), &[2, 2, 1]);
    assert_eq!(output.data(), &[2.0; 4]);
    Ok(())
}

#[test]
fn matches_a_naive_product_on_a_large_matrix() -> Result<()> {
    let (rows, inputs, outputs) = (37, 129, 65);
    let weight: Vec<f32> = (0..outputs * inputs).map(|index| float(index % 13) * 0.01).collect();
    let input: Vec<f32> = (0..rows * inputs).map(|index| float(index % 7) - 3.0).collect();
    let linear = Linear::new(Tensor::new(vec![outputs, inputs], weight.clone())?, None)?;
    let output = linear.forward(&Tensor::new(vec![rows, inputs], input.clone())?)?;
    for row in 0..rows {
        for column in 0..outputs {
            let expected: f32 =
                (0..inputs).map(|k| input[row * inputs + k] * weight[column * inputs + k]).sum();
            let actual = output.data()[row * outputs + column];
            assert!((expected - actual).abs() < 1e-3, "{row},{column}: {expected} != {actual}");
        }
    }
    Ok(())
}

#[test]
fn rejects_a_mismatched_input_width() -> Result<()> {
    let linear = Linear::new(Tensor::zeros(vec![2, 3]), None)?;
    assert!(linear.forward(&Tensor::zeros(vec![1, 2])).is_err());
    Ok(())
}

#[test]
fn accumulates_into_an_existing_output() -> Result<()> {
    let linear = Linear::new(Tensor::new(vec![1, 2], vec![1.0, 2.0])?, Some(vec![0.5]))?;
    let mut output = Tensor::new(vec![2, 1], vec![10.0, 20.0])?;
    linear.accumulate(&Tensor::new(vec![2, 2], vec![1.0, 1.0, 2.0, 0.0])?, &mut output)?;
    assert_eq!(output.data(), &[13.5, 22.5]);
    Ok(())
}
