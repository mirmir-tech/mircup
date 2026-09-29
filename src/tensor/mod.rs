mod decode;

pub use decode::DType;

use crate::{Error, Result};

/// A dense row-major `f32` tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let expected = shape.iter().product();
        if data.len() != expected {
            return Err(Error::DataLength { shape, expected, actual: data.len() });
        }
        Ok(Self { shape, data })
    }

    #[must_use]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let data = vec![0.0; shape.iter().product()];
        Self { shape, data }
    }

    /// Decodes a little-endian checkpoint payload into `f32` values.
    pub fn from_le_bytes(dtype: DType, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        Self::new(shape, dtype.decode(bytes)?)
    }

    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Size of the innermost dimension.
    #[must_use]
    pub fn width(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    /// Number of innermost rows, the product of every outer dimension.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.data.len().checked_div(self.width()).unwrap_or(0)
    }

    pub fn reshape(self, shape: Vec<usize>) -> Result<Self> {
        Self::new(shape, self.data)
    }

    /// Adds `other` element-wise; both tensors must have the same shape.
    pub fn add_assign(&mut self, other: &Self) -> Result<()> {
        if self.shape != other.shape {
            return Err(Error::Shape {
                operation: "add",
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        for (value, addend) in self.data.iter_mut().zip(&other.data) {
            *value += addend;
        }
        Ok(())
    }

    /// Adds `row` to every innermost row.
    pub fn add_row_assign(&mut self, row: &[f32]) -> Result<()> {
        if row.len() != self.width() {
            return Err(Error::Shape {
                operation: "row add",
                expected: vec![self.width()],
                actual: vec![row.len()],
            });
        }
        for chunk in self.data.chunks_exact_mut(row.len()) {
            for (value, addend) in chunk.iter_mut().zip(row) {
                *value += addend;
            }
        }
        Ok(())
    }

    /// Copies the innermost rows at `indices` into a `[indices.len(), width]`
    /// tensor.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<Self> {
        let width = self.width();
        let rows = self.rows();
        let mut data = Vec::with_capacity(indices.len() * width);
        for &index in indices {
            if index >= rows {
                return Err(Error::Shape {
                    operation: "row gather",
                    expected: vec![rows],
                    actual: vec![index],
                });
            }
            data.extend_from_slice(&self.data[index * width..(index + 1) * width]);
        }
        Self::new(vec![indices.len(), width], data)
    }

    /// Splits the innermost dimension into equal consecutive parts.
    pub fn split_last(&self, parts: usize) -> Result<Vec<Self>> {
        let width = self.width();
        if parts == 0 || !width.is_multiple_of(parts) {
            return Err(Error::Shape {
                operation: "split",
                expected: vec![parts],
                actual: self.shape.clone(),
            });
        }
        let part = width / parts;
        let mut shape = self.shape.clone();
        if let Some(last) = shape.last_mut() {
            *last = part;
        }
        let mut outputs: Vec<Vec<f32>> =
            (0..parts).map(|_| Vec::with_capacity(self.data.len() / parts)).collect();
        for row in self.data.chunks_exact(width) {
            for (output, values) in outputs.iter_mut().zip(row.chunks_exact(part)) {
                output.extend_from_slice(values);
            }
        }
        outputs.into_iter().map(|data| Self::new(shape.clone(), data)).collect()
    }
}

#[cfg(test)]
mod tests;
