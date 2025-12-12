pub mod bresenham;
pub mod scanline;
pub mod sort;
pub mod sync;

use std::ops::{
    Add,
    Mul,
};

use nalgebra::Vector3;

pub fn lerp<T>(x0: T, x1: T, t: f32) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T>,
{
    x0 * (1.0 - t) + x1 * t
}

pub fn trilinear_interpolation<C, T>(coefficients: Vector3<C>, points: [T; 3]) -> T
where
    C: Copy,
    T: Mul<C, Output = T> + Add<T, Output = T> + Copy,
{
    points[0] * coefficients[0] + points[1] * coefficients[1] + points[2] * coefficients[2]
}
