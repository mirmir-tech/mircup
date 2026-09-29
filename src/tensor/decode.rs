use half::{bf16, f16};

use crate::{Error, Result};

/// Encoding of a little-endian checkpoint payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
}

impl DType {
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }

    pub(crate) fn decode(self, bytes: &[u8]) -> Result<Vec<f32>> {
        if !bytes.len().is_multiple_of(self.bytes()) {
            return Err(Error::Payload { dtype: self, bytes: bytes.len() });
        }
        Ok(match self {
            Self::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|chunk| f32::from_le_bytes(*chunk))
                .collect(),
            Self::F16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| f16::from_le_bytes(*chunk).to_f32())
                .collect(),
            Self::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|chunk| bf16::from_le_bytes(*chunk).to_f32())
                .collect(),
        })
    }
}
