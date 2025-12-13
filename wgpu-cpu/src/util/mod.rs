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
use num_traits::Zero;

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
    fn interpolate<T>(&self, values: impl AsRef<[T]>) -> T
    where
        T: Mul<f32, Output = T> + AddAssign<T> + Copy + Zero,
    {
        Barycentric::interpolate(&self, values)
    }
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
