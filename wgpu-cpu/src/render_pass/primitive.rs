use itertools::Itertools;
use nalgebra::{
    Point3,
    Vector3,
};

use crate::{
    render_pass::clipper::ClipPosition,
    util::IteratorExt,
};

pub type Point<Vertex> = Primitive<Vertex, 1>;
pub type Line<Vertex> = Primitive<Vertex, 2>;
pub type Tri<Vertex> = Primitive<Vertex, 3>;

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

pub trait AssemblePrimitives<Vertex, const NUM_VERTICES: usize> {
    type Face;

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, NUM_VERTICES, Self::Face>>;
}

impl<A, Vertex, const NUM_VERTICES: usize> AssemblePrimitives<Vertex, NUM_VERTICES> for &mut A
where
    A: AssemblePrimitives<Vertex, NUM_VERTICES>,
{
    type Face = A::Face;

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, NUM_VERTICES, Self::Face>> {
        A::assemble(self, vertices)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimitiveList;

impl<Vertex> AssemblePrimitives<Vertex, 1> for PrimitiveList {
    type Face = ();

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 1, Self::Face>> {
        vertices
            .into_iter()
            .map(|vertex| Primitive::new([vertex], ()))
    }
}

impl<Vertex> AssemblePrimitives<Vertex, 2> for PrimitiveList {
    type Face = ();

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 2, Self::Face>> {
        vertices
            .into_iter()
            .array_chunks_::<2>()
            .map(|vertices| Primitive::new(vertices, ()))
    }
}

impl<Vertex> AssemblePrimitives<Vertex, 3> for PrimitiveList
where
    Vertex: AsRef<ClipPosition>,
{
    type Face = TriFace;

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 3, Self::Face>> {
        vertices.into_iter().array_chunks_::<3>().map(|vertices| {
            let face = TriFace::new(&vertices);
            Primitive::new(vertices, face)
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PrimitiveStrip;

impl<Vertex> AssemblePrimitives<Vertex, 2> for PrimitiveStrip
where
    Vertex: Clone,
{
    type Face = ();

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 2>> {
        vertices
            .into_iter()
            .tuple_windows()
            .map(|(a, b)| Primitive::new([a, b], ()))
    }
}

impl<Vertex> AssemblePrimitives<Vertex, 3> for PrimitiveStrip
where
    Vertex: Clone + AsRef<ClipPosition>,
{
    type Face = TriFace;

    fn assemble(
        &mut self,
        vertices: impl IntoIterator<Item = Vertex>,
    ) -> impl IntoIterator<Item = Primitive<Vertex, 3, Self::Face>> {
        TriangleStripIter::new(vertices.into_iter()).map(|vertices| {
            let face = TriFace::new(&vertices);
            Primitive::new(vertices, face)
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriangleStripIter<I, Vertex> {
    inner: TriangleStripIterInner<I, Vertex>,
}

#[derive(Clone, Copy, Debug)]
enum TriangleStripIterInner<I, Vertex> {
    Iter {
        inner: I,
        even: bool,
        buffer: [Vertex; 2],
    },
    Empty,
}

impl<I, Vertex> TriangleStripIter<I, Vertex>
where
    I: Iterator<Item = Vertex>,
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

impl<I, Vertex> Iterator for TriangleStripIter<I, Vertex>
where
    I: Iterator<Item = Vertex>,
    Vertex: Clone,
{
    type Item = [Vertex; 3];

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
    use nalgebra::Point3;

    use crate::render_pass::{
        clipper::ClipPosition,
        primitive::{
            TriFace,
            TriangleStripIter,
        },
    };

    #[test]
    fn triangle_strip() {
        // example from https://gpuweb.github.io/gpuweb/#primitive-assembly

        let expected = [[0, 1, 2], [2, 1, 3], [2, 3, 4], [4, 3, 5]];
        let got = TriangleStripIter::new(0..6).collect::<Vec<_>>();
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
