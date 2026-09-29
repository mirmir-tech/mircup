/// Converts a count or index to `f32`; every caller passes values far below
/// 2^24, where the conversion is exact.
#[expect(clippy::cast_precision_loss, reason = "counts and indices stay below 2^24")]
pub const fn float(value: usize) -> f32 {
    value as f32
}
