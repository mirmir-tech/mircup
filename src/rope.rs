use crate::{Error, Result, Tensor, numeric::float};

/// Rotary position tables for the rotate-half (non-interleaved) layout.
#[derive(Debug, Clone, PartialEq)]
pub struct Rope {
    dimension: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Rope {
    /// Builds tables for `positions` positions with inverse frequencies
    /// `1 / theta^(2i / dimension)`, all in `f32`.
    pub fn new(dimension: usize, theta: f32, positions: usize) -> Result<Self> {
        if dimension == 0 || !dimension.is_multiple_of(2) {
            return Err(Error::RotaryDimension(dimension));
        }
        let half = dimension / 2;
        let inverse: Vec<f32> = (0..half)
            .map(|index| 1.0 / theta.powf(float(2 * index) / float(dimension)))
            .collect();
        let mut cos = Vec::with_capacity(positions * half);
        let mut sin = Vec::with_capacity(positions * half);
        for position in 0..positions {
            for frequency in &inverse {
                let angle = float(position) * frequency;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }
        Ok(Self { dimension, cos, sin })
    }

    #[must_use]
    pub const fn positions(&self) -> usize {
        self.cos.len() / (self.dimension / 2)
    }

    /// Rotates, in place, the first `heads` heads of every token of a
    /// `[batch, length, width]` tensor whose tokens are sequences of
    /// `dimension`-wide heads; token `t` of every sequence uses position `t`.
    /// A fused query-key-value projection rotates its query and key heads.
    pub fn apply(&self, tensor: &mut Tensor, heads: usize) -> Result<()> {
        let &[_, length, width] = tensor.shape() else {
            return Err(self.shape_error(tensor));
        };
        let dimension = self.dimension;
        if !width.is_multiple_of(dimension) || heads * dimension > width {
            return Err(self.shape_error(tensor));
        }
        if length > self.positions() {
            return Err(Error::PositionOutOfRange {
                position: length,
                positions: self.positions(),
            });
        }
        let half = dimension / 2;
        for (index, token) in tensor.data_mut().chunks_exact_mut(width).enumerate() {
            let position = index % length;
            let cos = &self.cos[position * half..(position + 1) * half];
            let sin = &self.sin[position * half..(position + 1) * half];
            for head in token[..heads * dimension].chunks_exact_mut(dimension) {
                let (first, second) = head.split_at_mut(half);
                for index in 0..half {
                    let (x, y) = (first[index], second[index]);
                    first[index] = x.mul_add(cos[index], -(y * sin[index]));
                    second[index] = y.mul_add(cos[index], x * sin[index]);
                }
            }
        }
        Ok(())
    }

    fn shape_error(&self, tensor: &Tensor) -> Error {
        Error::Shape {
            operation: "rotary input [batch, length, heads × dimension]",
            expected: vec![0, 0, self.dimension],
            actual: tensor.shape().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Rope;
    use crate::{Result, Tensor};

    #[test]
    fn leaves_position_zero_unchanged_and_rotates_later_positions() -> Result<()> {
        let rope = Rope::new(2, 10_000.0, 2)?;
        let mut tensor = Tensor::new(vec![1, 2, 2], vec![1.0, 2.0, 1.0, 0.0])?;
        rope.apply(&mut tensor, 1)?;
        assert_eq!(&tensor.data()[..2], &[1.0, 2.0]);
        let (cos, sin) = (1.0_f32.cos(), 1.0_f32.sin());
        assert!((tensor.data()[2] - cos).abs() < 1e-6);
        assert!((tensor.data()[3] - sin).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn rejects_sequences_longer_than_the_table() -> Result<()> {
        let rope = Rope::new(2, 10_000.0, 1)?;
        assert!(rope.apply(&mut Tensor::zeros(vec![1, 2, 2]), 1).is_err());
        Ok(())
    }

    #[test]
    fn leaves_heads_past_the_rotated_ones_unchanged() -> Result<()> {
        let rope = Rope::new(2, 10_000.0, 2)?;
        let mut tensor = Tensor::new(vec![1, 2, 4], vec![1.0, 0.0, 5.0, 6.0, 1.0, 0.0, 5.0, 6.0])?;
        rope.apply(&mut tensor, 1)?;
        assert_eq!(&tensor.data()[6..], &[5.0, 6.0]);
        assert!((tensor.data()[4] - 1.0_f32.cos()).abs() < 1e-6);
        Ok(())
    }
}
