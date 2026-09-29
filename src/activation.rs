use rayon::prelude::*;

use crate::{Error, Result, Tensor, fastmath};

/// Elements handled by one Rayon task.
const CHUNK: usize = 16 * 1024;

/// Exact (error-function) GELU applied in place.
pub fn gelu(tensor: &mut Tensor) {
    tensor.data_mut().par_chunks_mut(CHUNK).for_each(|chunk| {
        for value in chunk {
            *value = gelu_value(*value);
        }
    });
}

/// Gated GELU: the first half of every row, passed through exact GELU,
/// times the second half. `[.., 2 × width]` becomes `[.., width]`.
pub fn geglu(input: &Tensor) -> Result<Tensor> {
    let doubled = input.width();
    if !doubled.is_multiple_of(2) {
        return Err(Error::Shape {
            operation: "GeGLU input",
            expected: vec![doubled + 1],
            actual: input.shape().to_vec(),
        });
    }
    let width = doubled / 2;
    let mut shape = input.shape().to_vec();
    if let Some(last) = shape.last_mut() {
        *last = width;
    }
    let mut output = Tensor::zeros(shape);
    let rows = (CHUNK / doubled).max(1);
    output
        .data_mut()
        .par_chunks_mut(width * rows)
        .zip(input.data().par_chunks(doubled * rows))
        .for_each(|(outputs, inputs)| {
            for (output, input) in outputs.chunks_exact_mut(width).zip(inputs.chunks_exact(doubled))
            {
                let (activated, gate) = input.split_at(width);
                for ((value, &x), &gate) in output.iter_mut().zip(activated).zip(gate) {
                    *value = gelu_value(x) * gate;
                }
            }
        });
    Ok(output)
}

fn gelu_value(x: f32) -> f32 {
    0.5 * x * (1.0 + fastmath::erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

/// Rectified linear unit applied in place.
pub fn relu(tensor: &mut Tensor) {
    tensor.data_mut().par_chunks_mut(CHUNK).for_each(|chunk| {
        for value in chunk {
            *value = value.max(0.0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{geglu, gelu, relu};
    use crate::{Result, Tensor};

    #[test]
    fn geglu_gates_the_activated_half() -> Result<()> {
        let output = geglu(&Tensor::new(vec![1, 4], vec![1.0, 0.0, 2.0, 3.0])?)?;
        assert_eq!(output.shape(), &[1, 2]);
        assert!((output.data()[0] - 1.682_689_4).abs() < 1e-6);
        assert!(output.data()[1].abs() < 1e-6);
        assert!(geglu(&Tensor::zeros(vec![1, 3])).is_err());
        Ok(())
    }

    #[test]
    fn gelu_uses_the_error_function() -> Result<()> {
        let mut tensor = Tensor::new(vec![3], vec![0.0, 1.0, -1.0])?;
        gelu(&mut tensor);
        let expected = [0.0, 0.841_344_7, -0.158_655_26];
        for (actual, expected) in tensor.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn relu_clamps_negative_values() -> Result<()> {
        let mut tensor = Tensor::new(vec![2], vec![-2.0, 3.0])?;
        relu(&mut tensor);
        assert_eq!(tensor.data(), &[0.0, 3.0]);
        Ok(())
    }
}
