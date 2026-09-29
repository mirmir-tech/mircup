mod backend;
mod gemm;

pub use gemm::{Matrix, MatrixMut, Threads, Write, multiply};

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
        let mut shape = input.shape().to_vec();
        if let Some(last) = shape.last_mut() {
            *last = self.outputs();
        }
        let mut output = Tensor::zeros(shape);
        self.write(input, &mut output, Write::Overwrite)?;
        Ok(output)
    }

    /// Adds the map of every row of `input` to the matching row of
    /// `output`, as a residual connection does, without a temporary.
    pub fn accumulate(&self, input: &Tensor, output: &mut Tensor) -> Result<()> {
        self.write(input, output, Write::Accumulate)
    }

    fn write(&self, input: &Tensor, output: &mut Tensor, write: Write) -> Result<()> {
        let (inputs, outputs) = (self.inputs(), self.outputs());
        let rows = input.rows();
        if input.width() != inputs || output.width() != outputs || output.rows() != rows {
            return Err(Error::Shape {
                operation: "linear input and output",
                expected: vec![rows, inputs, outputs],
                actual: [input.shape(), output.shape()].concat(),
            });
        }
        multiply(
            &mut MatrixMut::row_major(output.data_mut(), rows, outputs),
            Matrix::row_major(input.data(), rows, inputs),
            Matrix::transposed(self.weight.data(), inputs, outputs),
            Threads::Pool,
            write,
        );
        if let Some(bias) = &self.bias {
            for row in output.data_mut().chunks_exact_mut(outputs) {
                for (value, addend) in row.iter_mut().zip(bias) {
                    *value += addend;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
