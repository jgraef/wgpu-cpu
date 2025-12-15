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

/// Interpolates between `N` points given `N` barycentric coordinates.
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

impl<const N: usize> Interpolate<N> for Barycentric<N> {
    fn interpolate<T>(&self, points: [T; N]) -> T
    where
        T: Mul<f32, Output = T> + Add<T, Output = T> + Copy,
    {
        let mut accu = points[0] * self.coefficients[0];
        for i in 1..N {
            accu = accu + points[i] * self.coefficients[i];
        }
        accu
    }
}

fn array_to_vector<const N: usize>(array: [f32; N]) -> SVector<f32, N> {
    // yes, this is just an into, but it helps the compiler to figure out the types
    // since Barycentric::interpolate is generic
    array.into()
}
