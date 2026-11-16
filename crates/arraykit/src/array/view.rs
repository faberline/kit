//! Borrowed, strided views of N-dimensional arrays.

use super::dtype::ArrayElement;
use super::error::{ArrayError, Result};
use super::ndarray::NdArray;
use super::shape::Shape;

/// A borrowed, strided view of an array. It shares the buffer of its source.
#[derive(Clone, Debug)]
pub struct ArrayView<'a, T: ArrayElement> {
    data: &'a [T],
    offset: usize,
    shape: Shape, // dims and element strides of this view
}

impl<'a, T: ArrayElement> ArrayView<'a, T> {
    /// Create a view from a buffer, a start offset and a shape.
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

    /// Total number of elements in the view.
    pub fn size(&self) -> usize {
        self.shape.size()
    }

    /// Number of dimensions of the view.
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

    /// Element at a multi-dimensional index.
    pub fn get(&self, index: &[usize]) -> Result<T> {
        let off = self.shape.offset(index)?;
        Ok(self.data[self.offset + off])
    }

    /// True when the strides are the row-major strides of the dimensions.
    pub fn is_contiguous(&self) -> bool {
        self.strides() == Shape::new(self.dims().to_vec()).strides()
    }

    /// Copy the view into a new array, in row-major order of the view.
    pub fn to_owned(&self) -> NdArray<T> {
        let dims = self.dims().to_vec();
        let mut result = Vec::with_capacity(self.size());
        let mut idx = vec![0usize; self.ndim()];

        for _ in 0..self.size() {
            result.push(self.get(&idx).unwrap_or_default());

            // Increment the multi-index in row-major order
            for j in (0..dims.len()).rev() {
                idx[j] += 1;
                if idx[j] < dims[j] {
                    break;
                }
                idx[j] = 0;
            }
        }

        NdArray::new(result, dims).expect("view size matches dims")
    }

    /// Reverse the axes of the view. No data is copied.
    pub fn transpose(&self) -> ArrayView<'a, T> {
        let dims: Vec<usize> = self.dims().iter().rev().copied().collect();
        let strides: Vec<usize> = self.strides().iter().rev().copied().collect();
        ArrayView {
            data: self.data,
            offset: self.offset,
            shape: Shape::with_strides(dims, strides).expect("same length"),
        }
    }

    /// Permute the axes of the view. No data is copied.
    ///
    /// # Arguments
    /// * `axes` - The new order of axes. Must be a permutation of 0..ndim.
    pub fn transpose_axes(&self, axes: &[usize]) -> Result<ArrayView<'a, T>> {
        if axes.len() != self.ndim() {
            return Err(ArrayError::ShapeMismatch {
                expected: self.dims().to_vec(),
                got: axes.to_vec(),
            });
        }

        // Validate axes is a valid permutation
        let mut seen = vec![false; self.ndim()];
        for &ax in axes {
            if ax >= self.ndim() {
                return Err(ArrayError::InvalidAxis {
                    axis: ax,
                    ndim: self.ndim(),
                });
            }
            if seen[ax] {
                return Err(ArrayError::ShapeMismatch {
                    expected: (0..self.ndim()).collect(),
                    got: axes.to_vec(),
                });
            }
            seen[ax] = true;
        }

        let dims: Vec<usize> = axes.iter().map(|&ax| self.dims()[ax]).collect();
        let strides: Vec<usize> = axes.iter().map(|&ax| self.strides()[ax]).collect();
        Ok(ArrayView {
            data: self.data,
            offset: self.offset,
            shape: Shape::with_strides(dims, strides)?,
        })
    }
}
