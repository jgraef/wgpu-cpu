pub mod bresenham;
pub mod scanline;
pub mod sort;
pub mod sync;

use std::ops::{
    Add,
    AddAssign,
    Mul,
};

use arrayvec::ArrayVec;
use nalgebra::SVector;
use num_traits::Zero;

pub fn make_label_owned(label: &Option<&str>) -> Option<String> {
    label.map(ToOwned::to_owned)
}

pub fn lerp<T>(x0: T, x1: T, t: f32) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T>,
{
    x0 * (1.0 - t) + x1 * t
}

#[derive(Clone, Copy, Debug)]
pub struct Barycentric<const N: usize> {
    pub coefficients: [f32; N],
}

impl<const N: usize> From<[f32; N]> for Barycentric<N> {
    fn from(value: [f32; N]) -> Self {
        Self {
            coefficients: value,
        }
    }
}

impl<const N: usize> Barycentric<N> {
    pub fn interpolate<T>(&self, values: impl AsRef<[T]>) -> T
    where
        T: Mul<f32, Output = T> + AddAssign<T> + Copy + Zero,
    {
        let values = values.as_ref();
        let mut output = Zero::zero();
        for i in 0..N {
            output += values[i] * self.coefficients[i];
        }
        output
    }
}

impl<const N: usize> naga_interpreter::bindings::Interpolate<N> for Barycentric<N> {
    fn interpolate_scalar(&self, scalars: [f32; N]) -> f32 {
        self.interpolate(scalars)
    }

    fn interpolate_vec2(&self, vectors: [[f32; 2]; N]) -> [f32; 2] {
        self.interpolate(vectors.map(array_to_vector)).into()
    }

    fn interpolate_vec3(&self, vectors: [[f32; 3]; N]) -> [f32; 3] {
        self.interpolate(vectors.map(array_to_vector)).into()
    }

    fn interpolate_vec4(&self, vectors: [[f32; 4]; N]) -> [f32; 4] {
        self.interpolate(vectors.map(array_to_vector)).into()
    }
}

fn array_to_vector<const N: usize>(array: [f32; N]) -> SVector<f32, N> {
    // yes, this is just an into, but it helps the compiler to figure out the types
    // since Barycentric::interpolate is generic
    array.into()
}

#[derive(Clone, Copy, Debug)]
pub struct ArrayChunksIter<const N: usize, I> {
    inner: I,
}

impl<const N: usize, I> Iterator for ArrayChunksIter<N, I>
where
    I: Iterator,
{
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = ArrayVec::new();

        for i in 0..N {
            if let Some(item) = self.inner.next() {
                buf.push(item);
            }
            else {
                return None;
            }
        }

        Some(buf.into_inner().ok().unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.inner.size_hint();
        (min / N, max.map(|max| max / N))
    }
}

impl<const N: usize, I> ExactSizeIterator for ArrayChunksIter<N, I> where I: ExactSizeIterator {}

pub trait IteratorExt: Iterator {
    fn array_chunks_<const N: usize>(self) -> ArrayChunksIter<N, Self>
    where
        Self: Sized,
    {
        ArrayChunksIter { inner: self }
    }
}

impl<T> IteratorExt for T where T: Iterator {}
