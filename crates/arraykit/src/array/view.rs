//! Borrowed, strided views of an `NdArray`.

use super::dtype::ArrayElement;
use super::error::{ArrayError, Result};
use super::ndarray::NdArray;
use super::shape::Shape;
use super::slice::SliceInfo;

/// A borrowed, strided view of an array. It shares the buffer of its source.
#[derive(Clone, Debug)]
pub struct ArrayView<'a, T: ArrayElement> {
    data: &'a [T],
    offset: usize,
    shape: Shape, // dims and element strides of this view
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Create a view over `data` starting at `offset`, with the given shape.
    pub(crate) fn new(data: &'a [T], offset: usize, shape: Shape) -> Self {
        Self {
            data,
            offset,
            shape,
        }
    }

    /// Dimensions of the view.
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Element strides of the view.
    pub fn strides(&self) -> &[usize] {
        self.shape.strides()
    }

    /// Total number of elements.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Pointer to the first element of the view in the shared buffer.
    pub fn data_ptr(&self) -> *const T {
        if self.size() > 0 {
            self.data[self.offset..].as_ptr()
        } else {
            self.data.as_ptr()
        }
    }

    /// Get the element at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<T> {
        let off = self.shape.offset(index)?;
        Ok(self.data[self.offset + off])
    }

    /// True when the strides are the row-major strides of the dimensions.
    pub fn is_contiguous(&self) -> bool {
        self.strides() == Shape::new(self.dims().to_vec()).strides()
    }

    /// Copy the view into a new owned array, in row-major order.
    pub fn to_owned(&self) -> NdArray<T> {
        let dims = self.dims().to_vec();
        let ndim = self.ndim();
        let mut result = Vec::with_capacity(self.size());
        let mut idx = vec![0usize; ndim];

        for _ in 0..self.size() {
            let off = self.shape.offset(&idx).unwrap_or(0);
            result.push(self.data[self.offset + off]);

            // Increment the row-major multi-index.
            for j in (0..ndim).rev() {
                idx[j] += 1;
                if idx[j] < dims[j] {
                    break;
                }
                idx[j] = 0;
            }
        }

        NdArray::new(result, dims).expect("view size matches dims")
    }

    /// Reverse the axes of the view. The result shares the same buffer.
    pub fn transpose(&self) -> ArrayView<'a, T> {
        let dims: Vec<usize> = self.dims().iter().rev().copied().collect();
        let strides: Vec<usize> = self.strides().iter().rev().copied().collect();
        let shape = Shape::with_strides(dims, strides).expect("same length");
        ArrayView {
            data: self.data,
            offset: self.offset,
            shape,
        }
    }

    /// Permute the axes of the view. The result shares the same buffer.
    ///
    /// `axes` must be a permutation of `0..ndim`.
    pub fn transpose_axes(&self, axes: &[usize]) -> Result<ArrayView<'a, T>> {
        let ndim = self.ndim();
        if axes.len() != ndim {
            return Err(ArrayError::ShapeMismatch {
                expected: self.dims().to_vec(),
                got: axes.to_vec(),
            });
        }

        let mut seen = vec![false; ndim];
        for &ax in axes {
            if ax >= ndim {
                return Err(ArrayError::InvalidAxis { axis: ax, ndim });
            }
            if seen[ax] {
                return Err(ArrayError::ShapeMismatch {
                    expected: (0..ndim).collect(),
                    got: axes.to_vec(),
                });
            }
            seen[ax] = true;
        }

        let old_dims = self.dims();
        let old_strides = self.strides();
        let dims: Vec<usize> = axes.iter().map(|&a| old_dims[a]).collect();
        let strides: Vec<usize> = axes.iter().map(|&a| old_strides[a]).collect();
        let shape = Shape::with_strides(dims, strides).expect("same length");
        Ok(ArrayView {
            data: self.data,
            offset: self.offset,
            shape,
        })
    }

    /// Reshape the view without a copy. The new size must match and the view
    /// must be contiguous.
    pub fn reshape(&self, new_dims: impl Into<Shape>) -> Result<ArrayView<'a, T>> {
        let new_shape: Shape = new_dims.into();
        if new_shape.size() != self.size() {
            return Err(ArrayError::ShapeMismatch {
                expected: new_shape.dims().to_vec(),
                got: self.dims().to_vec(),
            });
        }
        if !self.is_contiguous() {
            return Err(ArrayError::InvalidOperation(
                "reshape of a non-contiguous view needs a copy; call to_owned first".into(),
            ));
        }
        Ok(ArrayView {
            data: self.data,
            offset: self.offset,
            shape: new_shape,
        })
    }

    /// Slice the view without a copy. The result shares the same buffer.
    pub fn slice(&self, info: &SliceInfo) -> Result<ArrayView<'a, T>> {
        // Checks the slice count and the index bounds before normalizing.
        let out_dims = info.output_shape(self.dims())?;
        let normalized = info.normalize_all(self.dims())?;

        let mut offset = self.offset;
        let mut dims = Vec::with_capacity(out_dims.len());
        let mut strides = Vec::with_capacity(out_dims.len());
        for (axis, norm) in normalized.iter().enumerate() {
            if norm.step < 0 {
                return Err(ArrayError::InvalidOperation(
                    "view slice with a negative step needs a copy; use NdArray::slice".into(),
                ));
            }
            let stride = self.strides()[axis];
            if norm.output_len > 0 {
                offset += norm.start * stride;
            }
            if !norm.is_index {
                dims.push(norm.output_len);
                strides.push(stride * norm.step as usize);
            }
        }
        debug_assert_eq!(dims, out_dims);

        let shape = Shape::with_strides(dims, strides).expect("same length");
        Ok(ArrayView {
            data: self.data,
            offset,
            shape,
        })
    }
}
