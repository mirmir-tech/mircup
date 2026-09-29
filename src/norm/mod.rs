use rayon::prelude::*;

use crate::{Error, Result, Tensor, numeric::double};

/// Layer normalisation over the innermost dimension, with an optional bias.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerNorm {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Vec<f32>, bias: Option<Vec<f32>>, eps: f32) -> Result<Self> {
        if let Some(bias) = &bias
            && bias.len() != weight.len()
        {
            return Err(Error::Shape {
                operation: "layer norm bias",
                expected: vec![weight.len()],
                actual: vec![bias.len()],
            });
        }
        Ok(Self { weight, bias, eps })
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        self.forward_in_place(&mut output)?;
        Ok(output)
    }

    pub fn forward_in_place(&self, tensor: &mut Tensor) -> Result<()> {
        let width = self.weight.len();
        if tensor.width() != width {
            return Err(Error::Shape {
                operation: "layer norm input",
                expected: vec![width],
                actual: tensor.shape().to_vec(),
            });
        }
        tensor
            .data_mut()
            .par_chunks_exact_mut(width)
            .for_each(|row| self.normalize(row));
        Ok(())
    }

    fn normalize(&self, row: &mut [f32]) {
        let count = double(row.len());
        let mean = row.iter().map(|&value| f64::from(value)).sum::<f64>() / count;
        let variance = row
            .iter()
            .map(|&value| {
                let centered = f64::from(value) - mean;
                centered * centered
            })
            .sum::<f64>()
            / count;
        let scale = 1.0 / (variance + f64::from(self.eps)).sqrt();
        for (index, value) in row.iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "activations are stored as f32")]
            let normalized = ((f64::from(*value) - mean) * scale) as f32;
            let shifted = self.bias.as_ref().map_or(0.0, |bias| bias[index]);
            *value = normalized * self.weight[index] + shifted;
        }
    }
}

#[cfg(test)]
mod tests;
