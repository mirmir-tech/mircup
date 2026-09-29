use rayon::prelude::*;

use crate::Tensor;

/// Exact (error-function) GELU applied in place.
pub fn gelu(tensor: &mut Tensor) {
    tensor.data_mut().par_iter_mut().for_each(|value| {
        *value = 0.5 * *value * (1.0 + libm::erff(*value * std::f32::consts::FRAC_1_SQRT_2));
    });
}

/// Rectified linear unit applied in place.
pub fn relu(tensor: &mut Tensor) {
    tensor.data_mut().par_iter_mut().for_each(|value| *value = value.max(0.0));
}

#[cfg(test)]
mod tests {
    use super::{gelu, relu};
    use crate::{Result, Tensor};

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
