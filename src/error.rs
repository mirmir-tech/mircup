use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("tensor data holds {actual} values, but shape {shape:?} needs {expected}")]
    DataLength {
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
    #[error("{operation} expects shape {expected:?}, got {actual:?}")]
    Shape {
        operation: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("{dtype:?} payload of {bytes} bytes is not a whole number of elements")]
    Payload { dtype: crate::DType, bytes: usize },
    #[error("token id {id} is outside an embedding table of {rows} rows")]
    TokenOutOfRange { id: u32, rows: usize },
    #[error("position {position} is outside a rotary table of {positions} positions")]
    PositionOutOfRange { position: usize, positions: usize },
    #[error("sequence length {length} exceeds the padded length {padded}")]
    SequenceLength { length: usize, padded: usize },
    #[error("rotary dimension {0} must be even and non-zero")]
    RotaryDimension(usize),
}
