//! Scanline algorithm
//!
//! Instructions from [tinyrenderer][1].
//!
//! [1]: https://haqr.eu/tinyrenderer/rasterization/

use nalgebra::{
    Point2,
    Vector2,
};

use crate::util::sort::bubblesort3;

pub fn scanlines(tri: [Point2<u32>; 3]) -> Scanlines {
    let mut tri = tri.map(|p| p.coords.cast::<isize>());

    // after sorting we have two vertical boundaries: [AC] and [AB, BC]. Between
    // these we can draw horizontal lines.
    bubblesort3(&mut tri, |a, b| a.y.cmp(&b.y));

    let [a, b, c] = tri;

    let ac = c - a;
    let ab = b - a;
    let bc = c - b;

    let state = if ac.y == 0 {
        bubblesort3(&mut tri, |a, b| a.x.cmp(&b.x));
        State::DegenerateLine {
            y: a.y,
            x1: tri[0].x,
            x2: tri[2].x,
        }
    }
    else {
        let half = HalfState {
            ac,
            ab,
            bc,
            a_x: a.x,
            b_x: b.x,
            b_y: b.y,
            c_y: c.y,
            y: a.y,
            i: 0,
        };
        if ab.y == 0 {
            State::Second { half, i2: 0 }
        }
        else {
            State::First { half }
        }
    };

    Scanlines { state }
}

#[derive(Clone, Copy, Debug)]
pub struct Scanlines {
    state: State,
}

#[derive(Clone, Copy, Debug)]
struct HalfState {
    ac: Vector2<isize>,
    ab: Vector2<isize>,
    bc: Vector2<isize>,
    a_x: isize,
    b_x: isize,
    b_y: isize,
    c_y: isize,
    y: isize,
    i: isize,
}

#[derive(Clone, Copy, Debug)]
enum State {
    DegenerateLine { y: isize, x1: isize, x2: isize },
    First { half: HalfState },
    Second { half: HalfState, i2: isize },
    Done,
}

impl Iterator for Scanlines {
    type Item = Scanline;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match &mut self.state {
                State::DegenerateLine { y, x1, x2 } => {
                    debug_assert!(*x1 >= 0);
                    debug_assert!(*x2 >= 0);
                    let line = Scanline::new_from_isize(*y, *x1, *x2);
                    self.state = State::Done;
                    break Some(line);
                }
                State::Done => {
                    return None;
                }
                State::First { half } => {
                    if half.y < half.b_y {
                        let x1 = half.a_x + (half.ac.x * half.i) / half.ac.y;
                        let x2 = half.a_x + (half.ab.x * half.i) / half.ab.y;
                        debug_assert!(x1 >= 0);
                        debug_assert!(x2 >= 0);

                        let line = Scanline::new_from_isize(half.y, x1, x2);

                        half.y += 1;
                        half.i += 1;

                        break Some(line);
                    }
                    else {
                        assert_eq!(half.y, half.b_y);
                        if half.bc.y == 0 {
                            self.state = State::Done;
                        }
                        else {
                            self.state = State::Second { half: *half, i2: 0 };
                        }
                    }
                }
                State::Second { half, i2 } => {
                    if half.y <= half.c_y {
                        let x1 = half.a_x + (half.ac.x * half.i) / half.ac.y;
                        let x2 = half.b_x + (half.bc.x * *i2) / half.bc.y;
                        debug_assert!(x1 >= 0);
                        debug_assert!(x2 >= 0);
                        *i2 += 1;

                        let line = Scanline::new_from_isize(half.y, x1, x2);

                        half.y += 1;
                        half.i += 1;

                        break Some(line);
                    }
                    else {
                        self.state = State::Done;
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Scanline {
    pub y: u32,
    pub x1: u32,
    pub x2: u32,
}

impl Scanline {
    fn new_from_isize(y: isize, mut x1: isize, mut x2: isize) -> Self {
        let err = move |_| {
            panic!("scanlines generated points that doesn't fit u32: y={y}, x1={x1}, x2={x2}");
        };

        if x1 > x2 {
            std::mem::swap(&mut x1, &mut x2);
        }

        Self {
            y: y.try_into().unwrap_or_else(err),
            x1: x1.try_into().unwrap_or_else(err),
            x2: x2.try_into().unwrap_or_else(err),
        }
    }
}

impl IntoIterator for Scanline {
    type IntoIter = ScanlineIter;
    type Item = Point2<u32>;

    fn into_iter(self) -> Self::IntoIter {
        ScanlineIter { scanline: self }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScanlineIter {
    scanline: Scanline,
    // todo: interpolation
}

impl Iterator for ScanlineIter {
    type Item = Point2<u32>;

    fn next(&mut self) -> Option<Self::Item> {
        (self.scanline.x1 <= self.scanline.x2).then(|| {
            let x = self.scanline.x1;
            self.scanline.x1 += 1;
            Point2::new(x, self.scanline.y)
        })
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point2;

    use crate::util::scanline::scanlines;

    #[test]
    fn bug_negative_coordinates_generated() {
        // this tri caused a negative x2 coordinate to be emitted, which panics when
        // creating the Scanline iterator.
        let tri = [Point2::new(0, 0), Point2::new(120, 0), Point2::new(119, 26)];

        /*
        Initial state:

        Scanlines {
            state: Second {
                half: HalfState {
                    ac: [
                        [
                            119,
                            26,
                        ],
                    ],
                    ab: [
                        [
                            120,
                            0,
                        ],
                    ],
                    bc: [
                        [
                            -1,
                            26,
                        ],
                    ],
                    a_x: 0,
                    b_y: 0,
                    c_y: 26,
                    y: 0,
                    i: 0,
                },
                i2: 0,
            },
        }

        first half is degenerate, so we start in second half.
        bc.x = -1 which is multiplied with the counter i2 and subtracted from starting x coordinate.
        the error was that we used a_x instead of b_x

        Fix:

        - let x2 = half.a_x + (half.bc.x * *i2) / half.bc.y;
        + let x2 = half.b_x + (half.bc.x * *i2) / half.bc.y;
        */

        let scanlines = scanlines(tri);
        println!("{scanlines:#?}");
        scanlines.flatten().for_each(|_point| {});
    }

    #[test]
    fn bug_switch_to_second_half_assertion_failed() {
        // this tri triggered the assertion to fail when switching to the second half:
        // assert_eq!(half.y, half.b_y);
        // turns out bubblesort was wrong.
        let tri = [
            Point2::new(256, 230),
            Point2::new(257, 84),
            Point2::new(358, 75),
        ];

        let scanlines = scanlines(tri);
        println!("{scanlines:#?}");
        scanlines.flatten().for_each(|_point| {});
    }
}
