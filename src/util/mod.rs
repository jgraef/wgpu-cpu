use std::ops::{
    Add,
    Mul,
};

pub mod bresenham;
pub mod sync;

pub fn lerp<T>(x0: T, x1: T, t: f32) -> T
where
    T: Mul<f32, Output = T> + Add<T, Output = T>,
{
    x0 * (1.0 - t) + x1 * t
}
