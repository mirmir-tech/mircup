/// A read-only strided view of a `rows × columns` matrix.
#[derive(Debug, Clone, Copy)]
pub struct Matrix<'a> {
    data: &'a [f32],
    rows: usize,
    columns: usize,
    row_stride: usize,
    column_stride: usize,
}

impl<'a> Matrix<'a> {
    pub const fn row_major(data: &'a [f32], rows: usize, columns: usize) -> Self {
        Self::strided(data, rows, columns, columns, 1)
    }

    /// Views a row-major `[columns, rows]` buffer as its `rows × columns`
    /// transpose.
    pub const fn transposed(data: &'a [f32], rows: usize, columns: usize) -> Self {
        Self::strided(data, rows, columns, 1, rows)
    }

    pub const fn strided(
        data: &'a [f32],
        rows: usize,
        columns: usize,
        row_stride: usize,
        column_stride: usize,
    ) -> Self {
        Self {
            data,
            rows,
            columns,
            row_stride,
            column_stride,
        }
    }

    const fn fits(&self) -> bool {
        self.rows == 0
            || self.columns == 0
            || (self.rows - 1) * self.row_stride + (self.columns - 1) * self.column_stride
                < self.data.len()
    }
}

/// A writable row-major `rows × columns` matrix whose rows may be padded.
#[derive(Debug)]
pub struct MatrixMut<'a> {
    data: &'a mut [f32],
    rows: usize,
    columns: usize,
    row_stride: usize,
}

impl<'a> MatrixMut<'a> {
    pub const fn row_major(data: &'a mut [f32], rows: usize, columns: usize) -> Self {
        Self::strided(data, rows, columns, columns)
    }

    pub const fn strided(
        data: &'a mut [f32],
        rows: usize,
        columns: usize,
        row_stride: usize,
    ) -> Self {
        Self { data, rows, columns, row_stride }
    }

    const fn fits(&self) -> bool {
        self.rows == 0
            || self.columns == 0
            || (self.rows - 1) * self.row_stride + self.columns <= self.data.len()
    }
}

/// Whether a product replaces the output or is added to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Write {
    Overwrite,
    Accumulate,
}

#[derive(Debug, Clone, Copy)]
pub enum Threads {
    /// Lets the product use every core.
    Pool,
    /// Runs on the calling thread, for callers that already parallelise.
    Caller,
}

/// Layout of one operand as a BLAS routine reads it: row-major storage, or
/// the transpose of row-major storage, with its leading dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Operand {
    RowMajor { leading: usize },
    Transposed { leading: usize },
}

impl Matrix<'_> {
    pub(super) fn operand(&self) -> Operand {
        if self.column_stride == 1 {
            Operand::RowMajor {
                leading: self.row_stride.max(self.columns),
            }
        } else {
            assert_eq!(self.row_stride, 1, "a BLAS operand needs one unit stride");
            Operand::Transposed {
                leading: self.column_stride.max(self.rows),
            }
        }
    }

    pub(super) const fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    pub(super) const fn columns(&self) -> usize {
        self.columns
    }

    /// Row and column strides.
    #[cfg(not(target_os = "macos"))]
    pub(super) const fn strides(&self) -> (usize, usize) {
        (self.row_stride, self.column_stride)
    }
}

impl MatrixMut<'_> {
    pub(super) const fn shape(&self) -> (usize, usize, usize) {
        (self.rows, self.columns, self.row_stride)
    }

    pub(super) const fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_mut_ptr()
    }
}

/// Writes `left · right` into `output`, replacing or adding to it.
///
/// Panics when the dimensions disagree or a view reaches beyond its buffer;
/// both are programming errors of this crate, never of checkpoint data.
pub fn multiply(
    output: &mut MatrixMut<'_>,
    left: Matrix<'_>,
    right: Matrix<'_>,
    threads: Threads,
    write: Write,
) {
    assert_eq!(left.rows, output.rows, "product rows disagree");
    assert_eq!(right.columns, output.columns, "product columns disagree");
    assert_eq!(left.columns, right.rows, "product inner dimensions disagree");
    assert!(output.fits() && left.fits() && right.fits(), "matrix view exceeds its buffer");
    if output.rows == 0 || output.columns == 0 {
        return;
    }
    super::backend::multiply(output, &left, &right, threads, write);
}
