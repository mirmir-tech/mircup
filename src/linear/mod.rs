mod gemm;

pub use gemm::{Matrix, MatrixMut, Threads, multiply};

use crate::{Error, Result, Tensor};

/// A dense affine map `y = x Wᵀ + b` with a `[outputs, inputs]` weight.
#[derive(Debug, Clone, PartialEq)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Vec<f32>>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Vec<f32>>) -> Result<Self> {
        let [outputs, _inputs] = weight.shape() else {
            return Err(Error::Shape {
                operation: "linear weight",
                expected: vec![0, 0],
                actual: weight.shape().to_vec(),
            });
        };
        if let Some(bias) = &bias
            && bias.len() != *outputs
        {
            return Err(Error::Shape {
                operation: "linear bias",
                expected: vec![*outputs],
                actual: vec![bias.len()],
            });
        }
        Ok(Self { weight, bias })
    }

    #[must_use]
    pub fn inputs(&self) -> usize {
        self.weight.shape()[1]
    }

    #[must_use]
    pub fn outputs(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Applies the map to every innermost row of `input`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (inputs, outputs) = (self.inputs(), self.outputs());
        if input.width() != inputs {
            return Err(Error::Shape {
                operation: "linear input",
                expected: vec![inputs],
                actual: input.shape().to_vec(),
            });
        }
        let rows = input.rows();
        let mut data = vec![0.0; rows * outputs];
        multiply(
            &mut MatrixMut::row_major(&mut data, rows, outputs),
            Matrix::row_major(input.data(), rows, inputs),
            Matrix::transposed(self.weight.data(), inputs, outputs),
            Threads::Pool,
        );
        if let Some(bias) = &self.bias {
            for row in data.chunks_exact_mut(outputs) {
                for (value, addend) in row.iter_mut().zip(bias) {
                    *value += addend;
                }
            }
        }
        let mut shape = input.shape().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = outputs;
        }
        Tensor::new(shape, data)
    }
}

#[cfg(test)]
mod tests;
