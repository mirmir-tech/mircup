use rayon::prelude::*;

use crate::{Error, Result, Tensor, numeric::float};

/// Rows normalised by one Rayon task; single rows are too small to split.
const ROWS_PER_TASK: usize = 32;

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
        let width = self.checked_width(input)?;
        let mut output = Tensor::zeros(input.shape().to_vec());
        output
            .data_mut()
            .par_chunks_mut(width * ROWS_PER_TASK)
            .zip(input.data().par_chunks(width * ROWS_PER_TASK))
            .for_each(|(outputs, inputs)| {
                for (output, input) in
                    outputs.chunks_exact_mut(width).zip(inputs.chunks_exact(width))
                {
                    let (mean, scale) = self.statistics(input);
                    for (index, (value, &x)) in output.iter_mut().zip(input).enumerate() {
                        *value = self.affine(index, (x - mean) * scale);
                    }
                }
            });
        Ok(output)
    }

    pub fn forward_in_place(&self, tensor: &mut Tensor) -> Result<()> {
        let width = self.checked_width(tensor)?;
        tensor.data_mut().par_chunks_mut(width * ROWS_PER_TASK).for_each(|rows| {
            for row in rows.chunks_exact_mut(width) {
                let (mean, scale) = self.statistics(row);
                for (index, value) in row.iter_mut().enumerate() {
                    *value = self.affine(index, (*value - mean) * scale);
                }
            }
        });
        Ok(())
    }

    fn checked_width(&self, tensor: &Tensor) -> Result<usize> {
        let width = self.weight.len();
        if tensor.width() == width {
            Ok(width)
        } else {
            Err(Error::Shape {
                operation: "layer norm input",
                expected: vec![width],
                actual: tensor.shape().to_vec(),
            })
        }
    }

    /// Mean and inverse standard deviation of one row.
    fn statistics(&self, row: &[f32]) -> (f32, f32) {
        let inverse_count = 1.0 / float(row.len());
        let mean = row.iter().sum::<f32>() * inverse_count;
        let variance =
            row.iter().map(|value| (value - mean) * (value - mean)).sum::<f32>() * inverse_count;
        (mean, 1.0 / (variance + self.eps).sqrt())
    }

    #[inline]
    fn affine(&self, index: usize, normalized: f32) -> f32 {
        self.bias.as_ref().map_or_else(
            || normalized * self.weight[index],
            |bias| normalized.mul_add(self.weight[index], bias[index]),
        )
    }
}

#[cfg(test)]
mod tests;
