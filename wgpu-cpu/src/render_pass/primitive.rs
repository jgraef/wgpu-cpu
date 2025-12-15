use std::ops::{
    Index,
    IndexMut,
};

use itertools::Itertools;

use crate::{
    render_pass::clipper::ClipPosition,
    util::IteratorExt,
};

pub type Point<T> = Primitive<T, 1>;
pub type Line<T> = Primitive<T, 2>;
pub type Tri<T> = Primitive<T, 3>;

#[derive(
    Clone, Copy, Debug, derive_more::From, derive_more::Into, derive_more::AsRef, derive_more::AsMut,
)]
pub struct Primitive<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> AsRef<[T]> for Primitive<T, N> {
    fn as_ref(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T, const N: usize> AsMut<[T]> for Primitive<T, N> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

impl<T> AsRef<T> for Primitive<T, 1> {
    fn as_ref(&self) -> &T {
        &self.0[0]
    }
}

impl<T> AsMut<T> for Primitive<T, 1> {
    fn as_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }
}

impl<T> From<T> for Primitive<T, 1> {
    fn from(value: T) -> Self {
        Self([value])
    }
}

impl<T, const N: usize> Index<usize> for Primitive<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<T, const N: usize> IndexMut<usize> for Primitive<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl<T, const N: usize> Primitive<T, N> {
    pub fn new(inner: [T; N]) -> Self {
        Self(inner)
    }

    pub fn map<U>(self, f: impl FnMut(T) -> U) -> Primitive<U, N> {
        Primitive(self.0.map(f))
    }

    pub fn each<U>(&self) -> [U; N]
    where
        T: AsRef<U>,
        U: Clone,
    {
        self.0.each_ref().map(|c| c.as_ref().clone())
    }

    pub fn each_ref<U>(&self) -> [&U; N]
    where
        T: AsRef<U>,
    {
        self.0.each_ref().map(|c| c.as_ref())
    }

    pub fn each_mut<U>(&mut self) -> [&mut U; N]
    where
        T: AsMut<U>,
    {
        self.0.each_mut().map(|c| c.as_mut())
    }
}

impl<T, const N: usize> Primitive<T, N>
where
    T: AsRef<ClipPosition>,
{
    pub fn clip_positions(&self) -> [ClipPosition; N] {
        self.each::<ClipPosition>()
    }
}

impl<T, const N: usize> Primitive<T, N>
where
    T: AsMut<ClipPosition>,
{
    pub fn clip_positions_mut(&mut self) -> [&mut ClipPosition; N] {
        self.each_mut::<ClipPosition>()
    }
}

impl<T> Primitive<T, 3>
where
    T: AsRef<ClipPosition>,
{
    pub fn front_face(&self) -> wgpu::FrontFace {
        let clip_positions = self.clip_positions();

        let ab = clip_positions[1].0 - clip_positions[0].0;
        let ac = clip_positions[2].0 - clip_positions[1].0;

        if ab.x * ac.y < ab.y * ac.x {
            wgpu::FrontFace::Ccw
        }
        else {
            wgpu::FrontFace::Cw
        }
    }
}

impl<T, const N: usize> IntoIterator for Primitive<T, N> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

pub trait AssemblePrimitives<T, const PRIMITIVE_SIZE: usize> {
    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = T>,
    ) -> impl IntoIterator<Item = Primitive<T, PRIMITIVE_SIZE>>;
}

impl<A, T, const PRIMITIVE_SIZE: usize> AssemblePrimitives<T, PRIMITIVE_SIZE> for &mut A
where
    A: AssemblePrimitives<T, PRIMITIVE_SIZE>,
{
    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = T>,
    ) -> impl IntoIterator<Item = Primitive<T, PRIMITIVE_SIZE>> {
        A::assemble(self, vertices)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimitiveList;

impl<T, const PRIMITIVE_SIZE: usize> AssemblePrimitives<T, PRIMITIVE_SIZE> for PrimitiveList {
    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = T>,
    ) -> impl IntoIterator<Item = Primitive<T, PRIMITIVE_SIZE>> {
        vertices
            .into_iter()
            .array_chunks_::<PRIMITIVE_SIZE>()
            .map(Into::into)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimitiveStrip;

impl<T> AssemblePrimitives<T, 2> for PrimitiveStrip
where
    T: Clone,
{
    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = T>,
    ) -> impl IntoIterator<Item = Primitive<T, 2>> {
        vertices
            .into_iter()
            .tuple_windows()
            .map(|(a, b)| Primitive([a, b]))
    }
}

impl<T> AssemblePrimitives<T, 3> for PrimitiveStrip
where
    T: Clone + 'static,
{
    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = T>,
    ) -> impl IntoIterator<Item = Primitive<T, 3>> {
        TriangleStripIter::new(vertices.into_iter()).map(Primitive)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriangleStripIter<I, T> {
    inner: TriangleStripIterInner<I, T>,
}

#[derive(Clone, Copy, Debug)]
enum TriangleStripIterInner<I, T> {
    Iter {
        inner: I,
        even: bool,
        buffer: [T; 2],
    },
    Empty,
}

impl<I, T> TriangleStripIter<I, T>
where
    I: Iterator<Item = T>,
{
    pub fn new(mut inner: I) -> Self {
        // try to fetch 2 points to fill the buffer. if we can't we won't yield any
        // triangles.
        let try_create = move || {
            let shift = [inner.next()?, inner.next()?];
            Some(TriangleStripIterInner::Iter {
                inner,
                even: false,
                buffer: shift,
            })
        };

        Self {
            inner: try_create().unwrap_or(TriangleStripIterInner::Empty),
        }
    }
}

impl<I, T> Iterator for TriangleStripIter<I, T>
where
    I: Iterator<Item = T>,
    T: Clone,
{
    type Item = [T; 3];

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.inner {
            TriangleStripIterInner::Iter {
                inner,
                even,
                buffer: shift,
            } => {
                if let Some(item) = inner.next() {
                    // keep buffer of 2 points. when we get a new point, first construct the output
                    // from the buffer and new point. then put the new point
                    // into the buffer at alternating indices (starting with 0)

                    let out = [shift[0].clone(), shift[1].clone(), item.clone()];

                    shift[*even as usize] = item;
                    *even = !*even;

                    Some(out)
                }
                else {
                    self.inner = TriangleStripIterInner::Empty;
                    None
                }
            }
            TriangleStripIterInner::Empty => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::render_pass::primitive::TriangleStripIter;

    #[test]
    fn triangle_strip() {
        // example from https://gpuweb.github.io/gpuweb/#primitive-assembly

        let expected = [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]];
        let got = TriangleStripIter::new(0..6).collect::<Vec<_>>();
        assert_eq!(got, expected);
    }
}
