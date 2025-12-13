//! Iterator-based Bresenham's line drawing algorithm
//!
//! Taken from [bresenham-rs](https://github.com/mbr/bresenham-rs/blob/master/src/lib.rs)
//!
//! [Bresenham's line drawing algorithm]
//! (https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm) is fast
//! algorithm to draw a line between two points. This crate implements the fast
//! integer variant, using an iterator-based appraoch for flexibility. It
//! calculates coordinates without knowing anything about drawing methods or
//! surfaces.

use nalgebra::{
    Point2,
    Vector2,
};

struct Octant(u8);

impl Octant {
    /// adapted from http://codereview.stackexchange.com/a/95551
    #[inline]
    fn from_points(start: Point2<i64>, end: Point2<i64>) -> Octant {
        let mut d = end - start;

        let mut octant = 0;

        if d.y < 0 {
            d.x = -d.x;
            d.y = -d.y;
            octant += 4;
        }

        if d.x < 0 {
            let tmp = d.x;
            d.x = d.y;
            d.y = -tmp;
            octant += 2
        }

        if d.x < d.y {
            octant += 1
        }

        Octant(octant)
    }

    #[inline]
    fn to_octant0(&self, p: Point2<i64>) -> Point2<i64> {
        match self.0 {
            0 => Point2::new(p.x, p.y),
            1 => Point2::new(p.y, p.x),
            2 => Point2::new(p.y, -p.x),
            3 => Point2::new(-p.x, p.y),
            4 => Point2::new(-p.x, -p.y),
            5 => Point2::new(-p.y, -p.x),
            6 => Point2::new(-p.y, p.x),
            7 => Point2::new(p.x, -p.y),
            _ => unreachable!(),
        }
    }

    #[inline]
    fn from_octant0(&self, p: Point2<i64>) -> Point2<i64> {
        match self.0 {
            0 => Point2::new(p.x, p.y),
            1 => Point2::new(p.y, p.x),
            2 => Point2::new(-p.y, p.x),
            3 => Point2::new(-p.x, p.y),
            4 => Point2::new(-p.x, -p.y),
            5 => Point2::new(-p.y, -p.x),
            6 => Point2::new(p.y, -p.x),
            7 => Point2::new(p.x, -p.y),
            _ => unreachable!(),
        }
    }
}

pub fn bresenham(start: Point2<u32>, end: Point2<u32>) -> Bresenham {
    Bresenham::new(start, end)
}

/// Line-drawing iterator
pub struct Bresenham {
    x: Point2<i64>,
    d: Vector2<i64>,
    x1: i64,
    diff: i64,
    octant: Octant,
    t: f32,
    dt: f32,
}

impl Bresenham {
    /// Creates a new iterator.
    ///
    /// Yields intermediate points between `start`and `end`. Does include
    /// `start` but not `end`.
    #[inline]
    pub fn new(start: Point2<u32>, end: Point2<u32>) -> Bresenham {
        let start = start.cast();
        let end = end.cast();

        let octant = Octant::from_points(start, end);

        let start = octant.to_octant0(start);
        let end = octant.to_octant0(end);

        let d = end - start;

        let dt = 1.0 / d.x as f32;

        Bresenham {
            x: start,
            d,
            x1: end.x,
            diff: d.y - d.x,
            octant: octant,
            t: 0.0,
            dt,
        }
    }
}

impl Iterator for Bresenham {
    type Item = (Point2<u32>, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.x.x >= self.x1 {
            return None;
        }

        let p = self.x;

        if self.diff >= 0 {
            self.x.y += 1;
            self.diff -= self.d.x;
        }

        self.diff += self.d.y;

        let t = self.t;
        self.t += self.dt;

        // loop inc
        self.x.x += 1;

        let point = self.octant.from_octant0(p);
        let point = Point2::from(point.coords.try_cast().unwrap());

        Some((point, t))
    }
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use nalgebra::Point2;

    use super::Bresenham;

    #[test]
    fn test_wp_example() {
        let bi = Bresenham::new(Point2::new(0, 1), Point2::new(6, 4));
        let res: Vec<_> = bi.map(|(p, _)| p).collect();

        assert_eq!(
            res,
            [
                Point2::new(0, 1),
                Point2::new(1, 1),
                Point2::new(2, 2),
                Point2::new(3, 2),
                Point2::new(4, 3),
                Point2::new(5, 3)
            ]
        )
    }

    #[test]
    fn test_inverse_wp() {
        let bi = Bresenham::new(Point2::new(6, 4), Point2::new(0, 1));
        let res: Vec<_> = bi.map(|(p, _)| p).collect();

        assert_eq!(
            res,
            [
                Point2::new(6, 4),
                Point2::new(5, 4),
                Point2::new(4, 3),
                Point2::new(3, 3),
                Point2::new(2, 2),
                Point2::new(1, 2)
            ]
        )
    }

    #[test]
    fn test_straight_hline() {
        let bi = Bresenham::new(Point2::new(2, 3), Point2::new(5, 3));
        let res: Vec<_> = bi.map(|(p, _)| p).collect();

        assert_eq!(
            res,
            [Point2::new(2, 3), Point2::new(3, 3), Point2::new(4, 3)]
        );
    }

    #[test]
    fn test_straight_vline() {
        let bi = Bresenham::new(Point2::new(2, 3), Point2::new(2, 6));
        let res: Vec<_> = bi.map(|(p, _)| p).collect();

        assert_eq!(
            res,
            [Point2::new(2, 3), Point2::new(2, 4), Point2::new(2, 5)]
        );
    }
}
