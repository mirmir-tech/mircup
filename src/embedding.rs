use crate::{DType, Error, Result, Tensor};

/// A token embedding table kept in its checkpoint encoding; rows are decoded
/// to `f32` only when looked up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingTable {
    dtype: DType,
    rows: usize,
    width: usize,
    bytes: Vec<u8>,
}

impl EmbeddingTable {
    pub fn from_le_bytes(dtype: DType, rows: usize, width: usize, bytes: Vec<u8>) -> Result<Self> {
        let expected = rows * width * dtype.bytes();
        if bytes.len() != expected {
            return Err(Error::DataLength {
                shape: vec![rows, width],
                expected: rows * width,
                actual: bytes.len() / dtype.bytes(),
            });
        }
        Ok(Self { dtype, rows, width, bytes })
    }

    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Returns `[ids.len(), width]` rows in id order.
    pub fn lookup(&self, ids: &[u32]) -> Result<Tensor> {
        let row_bytes = self.width * self.dtype.bytes();
        let mut data = Vec::with_capacity(ids.len() * self.width);
        for &id in ids {
            let row = usize::try_from(id)
                .ok()
                .filter(|&row| row < self.rows)
                .ok_or(Error::TokenOutOfRange { id, rows: self.rows })?;
            let start = row * row_bytes;
            data.extend(self.dtype.decode(&self.bytes[start..start + row_bytes])?);
        }
        Tensor::new(vec![ids.len(), self.width], data)
    }
}

#[cfg(test)]
mod tests {
    use half::f16;

    use super::EmbeddingTable;
    use crate::{DType, Result};

    #[test]
    fn decodes_only_requested_rows() -> Result<()> {
        let bytes: Vec<u8> = [0.5_f32, 1.0, 2.0, 4.0]
            .iter()
            .flat_map(|value| f16::from_f32(*value).to_le_bytes())
            .collect();
        let table = EmbeddingTable::from_le_bytes(DType::F16, 2, 2, bytes)?;
        assert_eq!(table.lookup(&[1, 0])?.data(), &[2.0, 4.0, 0.5, 1.0]);
        assert!(table.lookup(&[2]).is_err());
        Ok(())
    }
}
