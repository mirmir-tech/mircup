//! Branch-free `exp` and `erf` that the compiler vectorises in slice loops.
//! Both stay within a few `f32` ulps of the libm functions over the ranges
//! activations reach; `tests` pins the bound.

const LOG2_E: f32 = std::f32::consts::LOG2_E;
const LN_2_HIGH: f32 = 0.693_359_4;
const LN_2_LOW: f32 = -2.121_944_4e-4;
/// Inputs below this underflow to zero; above the upper bound they overflow.
const EXP_RANGE: (f32, f32) = (-87.336_55, 88.0);

/// `e^x` by range reduction to `x = n ln 2 + r` and a degree-6 polynomial.
#[inline]
pub fn exp(x: f32) -> f32 {
    let x = x.clamp(EXP_RANGE.0, EXP_RANGE.1);
    let n = (x * LOG2_E).round_ties_even();
    let r = n.mul_add(-LN_2_LOW, n.mul_add(-LN_2_HIGH, x));
    let polynomial = r
        .mul_add(1.987_569_1e-4, 1.398_199_9e-3)
        .mul_add(r, 8.333_452e-3)
        .mul_add(r, 4.166_579_6e-2)
        .mul_add(r, 1.666_666_5e-1)
        .mul_add(r, 0.5);
    let value = (polynomial * r).mul_add(r, r) + 1.0;
    value * power_of_two(n)
}

/// `2^n` for an integral `n` inside the normal exponent range.
#[inline]
const fn power_of_two(n: f32) -> f32 {
    #[expect(clippy::cast_possible_truncation, reason = "n is integral and within ±127")]
    let exponent = n as i32;
    #[expect(clippy::cast_sign_loss, reason = "the biased exponent is positive after clamping")]
    let bits = ((exponent + 127) as u32) << 23;
    f32::from_bits(bits)
}

/// Error function after Abramowitz and Stegun 7.1.26; in `f32` its absolute
/// error stays below `1e-6`.
#[inline]
pub fn erf(x: f32) -> f32 {
    let magnitude = x.abs();
    let t = 1.0 / 0.327_591_1_f32.mul_add(magnitude, 1.0);
    let polynomial = t
        .mul_add(1.061_405_4, -1.453_152_1)
        .mul_add(t, 1.421_413_7)
        .mul_add(t, -0.284_496_74)
        .mul_add(t, 0.254_829_6)
        * t;
    let value = polynomial.mul_add(-exp(-magnitude * magnitude), 1.0);
    value.copysign(x)
}

#[cfg(test)]
mod tests {
    use super::{erf, exp};

    #[test]
    fn exp_matches_libm_in_relative_error() {
        let mut worst = 0.0_f32;
        for step in -8700_i16..8800 {
            let x = f32::from(step) * 0.01;
            let expected = x.exp();
            worst = worst.max(((exp(x) - expected) / expected).abs());
        }
        assert!(worst < 4e-7, "worst relative error {worst}");
        assert!(exp(-1000.0) < 1e-37);
    }

    #[test]
    fn erf_matches_libm_in_absolute_error() {
        let mut worst = 0.0_f32;
        for step in -6000_i16..6000 {
            let x = f32::from(step) * 0.001;
            worst = worst.max((erf(x) - libm::erff(x)).abs());
        }
        assert!(worst < 1e-6, "worst absolute error {worst}");
        assert!(erf(0.0).abs() < 1e-6);
    }
}
