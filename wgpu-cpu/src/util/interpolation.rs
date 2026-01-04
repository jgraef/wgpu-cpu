use std::ops::{
    Add,
    Mul,
};

use nalgebra::SVector;

pub trait Interpolate<const N: usize> {
    fn interpolate<T>(&self, points: [T; N]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy;
}

impl<const N: usize, U> Interpolate<N> for &U
where
    U: Interpolate<N>,
{
    fn interpolate<T>(&self, points: [T; N]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        U::interpolate(self, points)
    }
}

/// Can interpolate between 1 point by just picking that point
#[derive(Clone, Copy, Debug, Default)]
pub struct NoInterpolation;

impl Interpolate<1> for NoInterpolation {
    fn interpolate<T>(&self, points: [T; 1]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        points[0]
    }
}

impl From<NoInterpolation> for Barycentric<1> {
    fn from(value: NoInterpolation) -> Self {
        Barycentric::at_vertex(0)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Select<const N: usize> {
    index: usize,
}

impl<const N: usize> Select<N> {
    pub fn new(index: usize) -> Self {
        assert!(index < N);
        Self { index }
    }
}

impl<const N: usize> Interpolate<N> for Select<N> {
    fn interpolate<T>(&self, points: [T; N]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        points[self.index]
    }
}

impl<const N: usize> From<Select<N>> for Barycentric<N> {
    fn from(value: Select<N>) -> Self {
        Barycentric::at_vertex(value.index)
    }
}

impl From<Select<1>> for NoInterpolation {
    fn from(value: Select<1>) -> Self {
        NoInterpolation
    }
}

impl From<Select<2>> for Lerp {
    fn from(value: Select<2>) -> Self {
        Lerp(value.index as f32)
    }
}

pub fn lerp<T>(x0: T, x1: T, t: f32) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T>,
{
    x0 * (1.0 - t) + x1 * t
}

/// Interpolates between 2 points given a coefficient `t` in [0, 1]. The
/// resulting value is:
///
/// ```plain
/// interpolate(t, [a, b]) = a * (1 - t) + b * t
/// ```
///
/// This is known as `lerp`, short for linear interpolation
#[derive(Clone, Copy, Debug)]
pub struct Lerp(pub f32);

impl Interpolate<2> for Lerp {
    fn interpolate<T>(&self, points: [T; 2]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        lerp(points[0], points[1], self.0)
    }
}

impl Mul<f32> for Lerp {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Add<Lerp> for Lerp {
    type Output = Self;

    fn add(self, rhs: Lerp) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

/// Interpolates between `N` points given `N` barycentric coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Barycentric<const N: usize> {
    pub coefficients: SVector<f32, N>,
}

impl<const N: usize> Barycentric<N> {
    pub fn at_vertex(i: usize) -> Self {
        let mut coefficients = SVector::zeros();
        coefficients[i] = 1.0;
        Self { coefficients }
    }
}

impl<const N: usize> From<[f32; N]> for Barycentric<N> {
    fn from(value: [f32; N]) -> Self {
        Self {
            coefficients: value.into(),
        }
    }
}

impl<const N: usize> Interpolate<N> for Barycentric<N> {
    fn interpolate<T>(&self, points: [T; N]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        // todo: this is a dot-product. using a existing function might improve
        // performance

        let mut accu = points[0] * self.coefficients[0];
        for i in 1..N {
            accu = accu + points[i] * self.coefficients[i];
        }
        accu
    }
}

impl<const N: usize> Mul<f32> for Barycentric<N> {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self {
            coefficients: self.coefficients * rhs,
        }
    }
}

impl<const N: usize> Add for Barycentric<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            coefficients: self.coefficients + rhs.coefficients,
        }
    }
}

fn array_to_vector<const N: usize>(array: [f32; N]) -> SVector<f32, N> {
    // yes, this is just an into, but it helps the compiler to figure out the types
    // since Barycentric::interpolate is generic
    array.into()
}
