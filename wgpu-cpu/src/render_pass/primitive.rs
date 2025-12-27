use arrayvec::ArrayVec;
use itertools::Itertools;
use nalgebra::{
    Point3,
    Vector3,
};

use crate::{
    render_pass::clipper::ClipPosition,
    util::IteratorExt,
};

pub trait ProcessItem {
    type Inner;
    type Processed<U>;

    fn process<U>(self, f: impl FnOnce(Self::Inner) -> U) -> Self::Processed<U>;
    //fn try_into_inner(self) -> Option<Self::Inner>;
}

#[derive(Clone, Copy, Debug)]
pub enum Separated<T> {
    Separator,
    Vertex(T),
}

impl<T> ProcessItem for Separated<T> {
    type Inner = T;
    type Processed<U> = Separated<U>;

    fn process<U>(self, f: impl FnOnce(T) -> U) -> Self::Processed<U> {
        match self {
            Separated::Separator => Separated::Separator,
            Separated::Vertex(item) => Separated::Vertex(f(item)),
        }
    }

    /*fn try_into_inner(self) -> Option<T> {
        match self {
            Separated::Separator => None,
            Separated::Vertex(item) => Some(item),
        }
    }*/
}

impl ProcessItem for u32 {
    type Inner = u32;
    type Processed<U> = U;

    fn process<U>(self, f: impl FnOnce(Self::Inner) -> U) -> Self::Processed<U> {
        f(self)
    }
}

pub type Point<Vertex> = Primitive<Vertex, 1>;
pub type Line<Vertex> = Primitive<Vertex, 2>;
pub type Tri<Vertex> = Primitive<Vertex, 3, TriFace>;

#[derive(Clone, Copy, Debug)]
pub struct Primitive<Vertex, const NUM_VERTICES: usize, Face = ()> {
    pub vertices: [Vertex; NUM_VERTICES],
    pub face: Face,
}

impl<Vertex, const NUM_VERTICES: usize, Face> Primitive<Vertex, NUM_VERTICES, Face> {
    pub fn new(vertices: [Vertex; NUM_VERTICES], face: Face) -> Self {
        Self { vertices, face }
    }

    pub fn map_vertices<U>(self, f: impl FnMut(Vertex) -> U) -> Primitive<U, NUM_VERTICES, Face> {
        Primitive {
            vertices: self.vertices.map(f),
            face: self.face,
        }
    }

    pub fn map_face<U>(self, f: impl FnOnce(Face) -> U) -> Primitive<Vertex, NUM_VERTICES, U> {
        Primitive {
            vertices: self.vertices,
            face: f(self.face),
        }
    }

    pub fn each_vertex<U>(&self) -> [U; NUM_VERTICES]
    where
        Vertex: AsRef<U>,
        U: Clone,
    {
        self.vertices.each_ref().map(|c| c.as_ref().clone())
    }

    pub fn each_vertex_ref<U>(&self) -> [&U; NUM_VERTICES]
    where
        Vertex: AsRef<U>,
    {
        self.vertices.each_ref().map(|c| c.as_ref())
    }

    pub fn each_vertex_mut<U>(&mut self) -> [&mut U; NUM_VERTICES]
    where
        Vertex: AsMut<U>,
    {
        self.vertices.each_mut().map(|c| c.as_mut())
    }
}

impl<Vertex, const NUM_VERTICES: usize, Face> Primitive<Vertex, NUM_VERTICES, Face>
where
    Vertex: AsRef<ClipPosition>,
{
    pub fn clip_positions(&self) -> [ClipPosition; NUM_VERTICES] {
        self.each_vertex::<ClipPosition>()
    }
}

impl<Vertex, const NUM_VERTICES: usize, Face> Primitive<Vertex, NUM_VERTICES, Face>
where
    Vertex: AsMut<ClipPosition>,
{
    pub fn clip_positions_mut(&mut self) -> [&mut ClipPosition; NUM_VERTICES] {
        self.each_vertex_mut::<ClipPosition>()
    }
}

impl<Vertex, const NUM_VERTICES: usize, Face> IntoIterator
    for Primitive<Vertex, NUM_VERTICES, Face>
{
    type Item = Vertex;
    type IntoIter = std::array::IntoIter<Vertex, NUM_VERTICES>;

    fn into_iter(self) -> Self::IntoIter {
        self.vertices.into_iter()
    }
}

pub trait AsFrontFace {
    /// Whether this is a front face, if the primitive has a face.
    ///
    /// This only gives the winding order. In order to know if this is actually
    /// a front face it has to be compared to the pipeline setting.
    fn try_front_face(&self) -> Option<wgpu::FrontFace>;
}

impl AsFrontFace for () {
    fn try_front_face(&self) -> Option<wgpu::FrontFace> {
        None
    }
}

impl<Vertex, const NUM_VERTICES: usize, Face> AsFrontFace for Primitive<Vertex, NUM_VERTICES, Face>
where
    Face: AsFrontFace,
{
    fn try_front_face(&self) -> Option<wgpu::FrontFace> {
        self.face.try_front_face()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriFace {
    /// Vector area of the tri.
    ///
    /// If `vector_area.z > 0.0` this is [`Ccw`][wgpu::FrontFace::Ccw],
    /// otherwise it's [`Cw`][wgpu::FrontFace::Cw]
    pub vector_area: Vector3<f32>,
}

impl TriFace {
    pub fn new<Vertex>(vertices: &[Vertex; 3]) -> TriFace
    where
        Vertex: AsRef<ClipPosition>,
    {
        // todo: there's probably a better way to handle this in homogeneous coordinates

        let [a, b, c] = vertices
            .each_ref()
            .map(|vertex| Point3::from_homogeneous(vertex.as_ref().0).expect("clip position z=0"));

        let ab = b - a;
        let ac = c - a;
        let vector_area = ab.cross(&ac);

        Self { vector_area }
    }

    /// Whether this is a front face.
    ///
    /// This only gives the winding order. In order to know if this is actually
    /// a front face it has to be compared to the pipeline setting.
    pub fn front_face(&self) -> wgpu::FrontFace {
        if self.vector_area.z > 0.0 {
            wgpu::FrontFace::Ccw
        }
        else {
            wgpu::FrontFace::Cw
        }
    }
}

impl AsFrontFace for TriFace {
    fn try_front_face(&self) -> Option<wgpu::FrontFace> {
        Some(self.front_face())
    }
}

pub trait Assemble<Vertex, const NUM_VERTICES: usize, const SEP: bool> {
    type Item;
    type Face;

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Self::Item>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, NUM_VERTICES, Self::Face>>;
}

#[derive(Clone, Copy, Debug, Default)]
pub struct List<const N: usize>;

impl<Vertex> Assemble<Vertex, 1, false> for List<1> {
    type Item = Vertex;
    type Face = ();

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 1, Self::Face>> {
        vertices
            .into_iter()
            .map(|vertices| Primitive::new([vertices], ()))
    }
}

impl<Vertex> Assemble<Vertex, 2, false> for List<2> {
    type Item = Vertex;
    type Face = ();

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 2, Self::Face>> {
        vertices
            .into_iter()
            .array_chunks_::<2>()
            .map(|vertices| Primitive::new(vertices, ()))
    }
}

impl<Vertex> Assemble<Vertex, 3, false> for List<3>
where
    Vertex: AsRef<ClipPosition>,
{
    type Item = Vertex;
    type Face = TriFace;

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 3, Self::Face>> {
        vertices.into_iter().array_chunks_::<3>().map(|vertices| {
            let face = TriFace::new(&vertices);
            Primitive::new(vertices, face)
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Strip<const N: usize>;

impl<Vertex> Assemble<Vertex, 2, false> for Strip<2>
where
    Vertex: Clone,
{
    type Item = Vertex;
    type Face = ();

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 2, Self::Face>> {
        vertices
            .into_iter()
            .tuple_windows()
            .map(|(a, b)| Primitive::new([a, b], ()))
    }
}

impl<Vertex> Assemble<Vertex, 2, true> for Strip<2>
where
    Vertex: Clone,
{
    type Item = Separated<Vertex>;
    type Face = ();

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Self::Item>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 2, Self::Face>> {
        // todo: better impl
        let mut buffer = ArrayVec::<_, 2>::new();
        let mut vertices = vertices.into_iter();

        std::iter::from_fn(move || {
            while let Some(item) = vertices.next() {
                match item {
                    Separated::Separator => buffer.clear(),
                    Separated::Vertex(vertex) => {
                        buffer.push(vertex);
                        if buffer.is_full() {
                            let line = buffer
                                .take()
                                .into_inner()
                                .unwrap_or_else(|_| unreachable!());
                            buffer.push(line[1].clone());
                            return Some(Primitive::new(line, ()));
                        }
                    }
                }
            }
            None
        })
    }
}

impl<Vertex> Assemble<Vertex, 3, false> for Strip<3>
where
    Vertex: Clone + AsRef<ClipPosition>,
{
    type Item = Vertex;
    type Face = TriFace;

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 3, Self::Face>> {
        TriStripIter::new(vertices.into_iter()).map(|vertices| {
            let face = TriFace::new(&vertices);
            Primitive::new(vertices, face)
        })
    }
}

impl<Vertex> Assemble<Vertex, 3, true> for Strip<3>
where
    Vertex: Clone + AsRef<ClipPosition>,
{
    type Item = Separated<Vertex>;
    type Face = TriFace;

    fn assemble(
        &self,
        vertices: impl IntoIterator<Item = Separated<Vertex>>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 3, Self::Face>> {
        ResetTriStripIter::new(vertices.into_iter()).map(|vertices| {
            let face = TriFace::new(&vertices);
            Primitive::new(vertices, face)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriStripGenerator<Vertex> {
    odd: bool,
    buffer: [Vertex; 2],
}

impl<Vertex> TriStripGenerator<Vertex> {
    fn new(buffer: [Vertex; 2]) -> Self {
        Self { odd: false, buffer }
    }
}

impl<Vertex> TriStripGenerator<Vertex>
where
    Vertex: Clone,
{
    pub fn push(&mut self, vertex: Vertex) -> [Vertex; 3] {
        // keep buffer of 2 points. when we get a new point, first construct the output
        // from the buffer and new point. then put the new point
        // into the buffer at alternating indices (starting with 0)

        let out = [
            self.buffer[0].clone(),
            self.buffer[1].clone(),
            vertex.clone(),
        ];

        self.buffer[self.odd as usize] = vertex;
        self.odd = !self.odd;

        out
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriStripIter<Vertices, Vertex> {
    inner: Vertices,
    state: Option<TriStripGenerator<Vertex>>,
}

impl<Vertices, Vertex> TriStripIter<Vertices, Vertex>
where
    Vertices: Iterator<Item = Vertex>,
{
    pub fn new(mut inner: Vertices) -> Self {
        // try to fetch 2 points to fill the buffer. if we can't we won't yield any
        // triangles.
        let mut try_fill_buffer = || Some([inner.next()?, inner.next()?]);
        let state = try_fill_buffer().map(TriStripGenerator::new);
        Self { inner, state }
    }
}

impl<Vertices, Vertex> Iterator for TriStripIter<Vertices, Vertex>
where
    Vertices: Iterator<Item = Vertex>,
    Vertex: Clone,
{
    type Item = [Vertex; 3];

    fn next(&mut self) -> Option<Self::Item> {
        self.state.as_mut().and_then(|state| {
            let vertex = self.inner.next()?;
            Some(state.push(vertex))
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ResetTriStripIter<Vertices, Vertex> {
    inner: Vertices,
    state: Option<TriStripGenerator<Vertex>>,
}

impl<Vertices, Vertex> ResetTriStripIter<Vertices, Vertex>
where
    Vertices: Iterator<Item = Separated<Vertex>>,
{
    pub fn new(mut inner: Vertices) -> Self {
        let state = Self::try_fill_buffer(&mut inner).map(TriStripGenerator::new);
        Self { inner, state }
    }

    fn try_fill_buffer(inner: &mut Vertices) -> Option<[Vertex; 2]> {
        // try to fetch 2 points to fill the buffer. if we can't we won't yield any
        // triangles.
        let mut buffer = ArrayVec::new();
        while let Some(item) = inner.next() {
            match item {
                Separated::Separator => {
                    buffer.clear();
                }
                Separated::Vertex(vertex) => {
                    buffer.push(vertex);
                }
            }

            if buffer.is_full() {
                let buffer = buffer.into_inner().unwrap_or_else(|_| unreachable!());
                return Some(buffer);
            }
        }
        None
    }
}

impl<Vertices, Vertex> Iterator for ResetTriStripIter<Vertices, Vertex>
where
    Vertices: Iterator<Item = Separated<Vertex>>,
    Vertex: Clone,
{
    type Item = [Vertex; 3];

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(state) = &mut self.state {
            match self.inner.next()? {
                Separated::Separator => {
                    self.state = Self::try_fill_buffer(&mut self.inner).map(TriStripGenerator::new);
                }
                Separated::Vertex(vertex) => {
                    let tri = state.push(vertex);
                    return Some(tri);
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::Point3;

    use crate::render_pass::{
        clipper::ClipPosition,
        primitive::{
            TriFace,
            TriStripIter,
        },
    };

    #[test]
    fn triangle_strip() {
        // example from https://gpuweb.github.io/gpuweb/#primitive-assembly

        let expected = [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]];
        let got = TriStripIter::new(0..6).collect::<Vec<_>>();
        assert_eq!(got, expected);
    }

    #[test]
    fn triangle_front_face() {
        #[rustfmt::skip]
        let tri_cw = [
            [-1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [ 1.0, 0.0, 0.0],
        ];
        let tri_cw = tri_cw.map(|v| ClipPosition(Point3::from(v).to_homogeneous()));
        let face = TriFace::new(&tri_cw);
        assert_eq!(face.front_face(), wgpu::FrontFace::Cw);

        #[rustfmt::skip]
        let tri_ccw = [
            [ 1.0, 0.0, 0.0],
            [ 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ];
        let tri_ccw = tri_ccw.map(|v| ClipPosition(Point3::from(v).to_homogeneous()));
        let face = TriFace::new(&tri_ccw);
        assert_eq!(face.front_face(), wgpu::FrontFace::Ccw);
    }
}
