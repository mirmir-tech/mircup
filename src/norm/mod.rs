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
            .par_chunks_mut(width * ROWS_PER_TASK)
            .for_each(|rows| rows.chunks_exact_mut(width).for_each(|row| self.normalize(row)));
        Ok(())
    }

    fn normalize(&self, row: &mut [f32]) {
        let inverse_count = 1.0 / float(row.len());
        let mean = row.iter().sum::<f32>() * inverse_count;
        let variance =
            row.iter().map(|value| (value - mean) * (value - mean)).sum::<f32>() * inverse_count;
        let scale = 1.0 / (variance + self.eps).sqrt();
        match &self.bias {
            Some(bias) => {
                for ((value, weight), bias) in row.iter_mut().zip(&self.weight).zip(bias) {
                    *value = ((*value - mean) * scale).mul_add(*weight, *bias);
                }
            },
            None => {
                for (value, weight) in row.iter_mut().zip(&self.weight) {
                    *value = (*value - mean) * scale * weight;
                }
            },
        }
    }
}

#[cfg(test)]
mod tests;
