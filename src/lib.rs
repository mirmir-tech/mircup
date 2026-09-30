//! Model-agnostic CPU execution primitives for Mirmir.
//!
//! Every operation is eager and runs on the calling thread pool; there is no
//! lazy graph and no hidden device state. Activations are `f32`; stored weights
//! may stay in their 16-bit checkpoint encoding until they are read.

mod activation;
mod attention;
mod embedding;
mod error;
mod fastmath;
mod linear;
mod norm;
mod numeric;
mod rope;
mod tensor;

pub use activation::{geglu, gelu, relu};
pub use attention::{AttentionWindow, attention, attention_at};
pub use embedding::EmbeddingTable;
pub use error::{Error, Result};
pub use linear::Linear;
pub use norm::LayerNorm;
pub use rope::Rope;
pub use tensor::{DType, Tensor};
