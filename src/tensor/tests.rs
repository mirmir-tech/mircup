use half::{bf16, f16};

use crate::{DType, Error, Result, Tensor, numeric::float};

#[test]
fn rejects_data_that_does_not_fill_the_shape() {
    assert_eq!(
        Tensor::new(vec![2, 3], vec![0.0; 5]),
        Err(Error::DataLength {
            shape: vec![2, 3],
            expected: 6,
            actual: 5
        })
    );
}

#[test]
fn decodes_half_precision_payloads_exactly() -> Result<()> {
    let halves = [f16::from_f32(1.5), f16::from_f32(-0.25)];
    let bytes: Vec<u8> = halves.iter().flat_map(|value| value.to_le_bytes()).collect();
    let tensor = Tensor::from_le_bytes(DType::F16, vec![2], &bytes)?;
    assert_eq!(tensor.data(), &[1.5, -0.25]);

    let brain: Vec<u8> = bf16::from_f32(3.0).to_le_bytes().to_vec();
    assert_eq!(Tensor::from_le_bytes(DType::BF16, vec![1], &brain)?.data(), &[3.0]);
    Ok(())
}

#[test]
fn rejects_a_truncated_payload() {
    assert_eq!(
        Tensor::from_le_bytes(DType::F32, vec![1], &[0, 0, 0]),
        Err(Error::Payload { dtype: DType::F32, bytes: 3 })
    );
}

#[test]
fn adds_a_row_to_every_row() -> Result<()> {
    let mut tensor = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0])?;
    tensor.add_row_assign(&[10.0, 20.0])?;
    assert_eq!(tensor.data(), &[11.0, 22.0, 13.0, 24.0]);
    Ok(())
}

#[test]
fn splits_and_gathers_rows() -> Result<()> {
    let tensor = Tensor::new(vec![2, 4], (0..8).map(float).collect())?;
    let parts = tensor.split_last(2)?;
    assert_eq!(parts[0].data(), &[0.0, 1.0, 4.0, 5.0]);
    assert_eq!(parts[1].data(), &[2.0, 3.0, 6.0, 7.0]);
    assert_eq!(tensor.gather_rows(&[1])?.data(), &[4.0, 5.0, 6.0, 7.0]);
    assert!(tensor.gather_rows(&[2]).is_err());
    Ok(())
}
