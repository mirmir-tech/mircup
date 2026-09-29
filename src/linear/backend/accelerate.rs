use std::ffi::c_int;

use super::super::gemm::{Matrix, MatrixMut, Operand, Threads, Write};

const ROW_MAJOR: c_int = 101;
const NO_TRANSPOSE: c_int = 111;
const TRANSPOSE: c_int = 112;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: c_int,
        transpose_left: c_int,
        transpose_right: c_int,
        rows: c_int,
        columns: c_int,
        inner: c_int,
        alpha: f32,
        left: *const f32,
        left_leading: c_int,
        right: *const f32,
        right_leading: c_int,
        beta: f32,
        output: *mut f32,
        output_leading: c_int,
    );
}

/// Accelerate schedules its own threads, so `threads` does not apply.
pub fn multiply(
    output: &mut MatrixMut<'_>,
    left: &Matrix<'_>,
    right: &Matrix<'_>,
    _threads: Threads,
    write: Write,
) {
    let beta = match write {
        Write::Overwrite => 0.0,
        Write::Accumulate => 1.0,
    };
    let (rows, columns, output_leading) = output.shape();
    let (left_transpose, left_leading) = blas(left.operand());
    let (right_transpose, right_leading) = blas(right.operand());
    // SAFETY: `gemm::multiply` asserted that the dimensions agree and that
    // every view fits its slice; the operand layouts describe exactly those
    // views. `output` is a unique borrow, so it aliases neither input; with
    // `beta == 0` BLAS only writes it, with `beta == 1` it adds to it.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            left_transpose,
            right_transpose,
            int(rows),
            int(columns),
            int(left.columns()),
            1.0,
            left.as_ptr(),
            left_leading,
            right.as_ptr(),
            right_leading,
            beta,
            output.as_mut_ptr(),
            int(output_leading),
        );
    }
}

fn blas(operand: Operand) -> (c_int, c_int) {
    match operand {
        Operand::RowMajor { leading } => (NO_TRANSPOSE, int(leading)),
        Operand::Transposed { leading } => (TRANSPOSE, int(leading)),
    }
}

/// BLAS takes 32-bit dimensions; a tensor dimension beyond that is a
/// programming error of the caller.
fn int(value: usize) -> c_int {
    c_int::try_from(value)
        .unwrap_or_else(|_| unreachable!("matrix dimension exceeds the BLAS range"))
}
