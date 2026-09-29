use super::super::gemm::{Matrix, MatrixMut, Threads, Write};

pub fn multiply(
    output: &mut MatrixMut<'_>,
    left: &Matrix<'_>,
    right: &Matrix<'_>,
    threads: Threads,
    write: Write,
) {
    let accumulate = write == Write::Accumulate;
    let (rows, columns, output_stride) = output.shape();
    let (left_rows, left_columns) = left.strides();
    let (right_rows, right_columns) = right.strides();
    let parallelism = match threads {
        Threads::Pool => gemm::Parallelism::Rayon(0),
        Threads::Caller => gemm::Parallelism::None,
    };
    // SAFETY: `gemm::multiply` asserted that the dimensions agree and that
    // every element addressed through the strides lies inside its slice.
    // `output` is a unique borrow, so it aliases neither input; gemm reads
    // the destination only when accumulating into it.
    unsafe {
        gemm::gemm(
            rows,
            columns,
            left.columns(),
            output.as_mut_ptr(),
            1,
            isize(output_stride),
            accumulate,
            left.as_ptr(),
            isize(left_columns),
            isize(left_rows),
            right.as_ptr(),
            isize(right_columns),
            isize(right_rows),
            1.0,
            1.0,
            false,
            false,
            false,
            parallelism,
        );
    }
}

fn isize(stride: usize) -> isize {
    isize::try_from(stride).unwrap_or_else(|_| unreachable!("a slice stride always fits isize"))
}
