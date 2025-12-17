//! https://gpuweb.github.io/gpuweb/#rasterization

use std::{
    num::NonZero,
    ops::Index,
};

use nalgebra::{
    Matrix2x4,
    Point2,
    Vector2,
};

use crate::{
    render_pass::{
        clipper::ClipPosition,
        primitive::{
            Line,
            Point,
            Primitive,
            Tri,
        },
    },
    util::{
        ArrayExt,
        bresenham::bresenham,
        interpolation::{
            Barycentric,
            Interpolate,
            Lerp,
            NoInterpolation,
        },
        scanline::scanlines,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct SamplePositions {
    pub _0: Vector2<f32>,
    pub _4: [Vector2<f32>; 4],
}

impl SamplePositions {
    pub const WEBGPU: Self = Self {
        _0: Vector2::new(0.5, 0.5),
        _4: [
            Vector2::new(0.375, 0.125),
            Vector2::new(0.875, 0.375),
            Vector2::new(0.125, 0.625),
            Vector2::new(0.625, 0.875),
        ],
    };

    pub fn for_multisample_count(&self, multisample_count: u32) -> &[Vector2<f32>] {
        match multisample_count {
            0 => std::slice::from_ref(&self._0),
            4 => self._4.as_slice(),
            _ => panic!("Invalid multisample count: {multisample_count}"),
        }
    }
}

impl Index<u32> for SamplePositions {
    type Output = [Vector2<f32>];

    fn index(&self, index: u32) -> &Self::Output {
        self.for_multisample_count(index)
    }
}

impl Default for SamplePositions {
    fn default() -> Self {
        Self::WEBGPU
    }
}

/// https://gpuweb.github.io/gpuweb/#rasterization
#[derive(Clone, Copy, Debug)]
pub struct FragmentDestination {
    pub position: Point2<u32>,
    pub sample_index: Option<NonZero<u8>>,
}

/// https://gpuweb.github.io/gpuweb/#rasterization
#[derive(Clone, Copy, Debug)]
pub struct RasterizationPoint<Primitive, Inter> {
    pub destination: FragmentDestination,
    pub coverage_mask: u32,
    pub front_face: Option<wgpu::FrontFace>,
    pub perspective_divisor: f32,
    pub depth: f32,
    pub primitive_vertices: Primitive,
    pub interpolation: Inter,
}

impl<Primitive, Inter> RasterizationPoint<Primitive, Inter> {
    pub fn map_interpolation<U>(
        self,
        f: impl FnOnce(Inter) -> U,
    ) -> RasterizationPoint<Primitive, U> {
        RasterizationPoint {
            destination: self.destination,
            coverage_mask: self.coverage_mask,
            front_face: self.front_face,
            perspective_divisor: self.perspective_divisor,
            depth: self.depth,
            primitive_vertices: self.primitive_vertices,
            interpolation: f(self.interpolation),
        }
    }
}

pub trait Rasterize<Primitive> {
    type Interpolation;

    fn rasterize(
        &self,
        primitive: Primitive,
    ) -> impl IntoIterator<Item = RasterizationPoint<Primitive, Self::Interpolation>>;
}

#[derive(Clone, Copy, Debug)]
pub struct ToTargetRaster {
    pub size: Vector2<u32>,
    pub to_raster: Matrix2x4<f32>,
}

impl ToTargetRaster {
    pub fn new(size: Vector2<u32>) -> Self {
        let size_f32 = size.cast::<f32>();

        // this matrix maps vectors produced by the pipeline to raster coordinates. the
        // result only needs to be cast to u32 to get a pixel coordinate.
        let mut to_raster = Matrix2x4::default();

        // shift by half the target size
        to_raster[(0, 3)] = 0.5 * size_f32.x;
        to_raster[(1, 3)] = 0.5 * size_f32.y;

        // scale by half the target size and flip y axis
        to_raster[(0, 0)] = 0.5 * (size_f32.x - 1.0);
        to_raster[(1, 1)] = -0.5 * (size_f32.y - 1.0);

        Self { size, to_raster }
    }

    pub fn to_raster(&self, position: ClipPosition) -> (Point2<f32>, Option<Point2<u32>>) {
        let raster_f32 = Point2::from(self.to_raster * position.0);
        let raster_u32 = raster_f32.coords.try_cast::<u32>().map(Point2::from);
        (raster_f32, raster_u32)
    }

    pub fn to_raster_primitive<const N: usize>(
        &self,
        primitive: Primitive<ClipPosition, N>,
    ) -> ([Point2<f32>; N], Option<[Point2<u32>; N]>) {
        let out_f32 = primitive
            .vertices
            .map(|position| Point2::from(self.to_raster * position.0));

        let out_u32 =
            out_f32.try_map_(|raster_f32| raster_f32.coords.try_cast::<u32>().map(Point2::from));

        (out_f32, out_u32)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullRasterizer;

impl<Primitive> Rasterize<Primitive> for NullRasterizer {
    type Interpolation = NoInterpolation;

    fn rasterize(
        &self,
        primitive: Primitive,
    ) -> impl IntoIterator<Item = RasterizationPoint<Primitive, NoInterpolation>> {
        []
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointRasterizer {
    target: ToTargetRaster,
}

impl PointRasterizer {
    pub fn new(target_size: nalgebra::Vector2<u32>) -> Self {
        Self {
            target: ToTargetRaster::new(target_size),
        }
    }
}

impl<T> Rasterize<Point<T>> for PointRasterizer
where
    T: AsRef<ClipPosition>,
{
    type Interpolation = NoInterpolation;

    fn rasterize(
        &self,
        primitive: Point<T>,
    ) -> impl IntoIterator<Item = RasterizationPoint<Point<T>, Self::Interpolation>> {
        let (_point_raster_f32, point_raster_u32) =
            self.target.to_raster(primitive.clip_positions()[0]);

        point_raster_u32.map(|raster| {
            /*RasterizationPoint {
                raster,
                position: primitive.into(),
                interpolation: (),
            }*/
            todo!();
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineRasterizer {
    target: ToTargetRaster,
}

impl LineRasterizer {
    pub fn new(target_size: nalgebra::Vector2<u32>) -> Self {
        Self {
            target: ToTargetRaster::new(target_size),
        }
    }
}

impl<T> Rasterize<Line<T>> for LineRasterizer
where
    T: AsRef<ClipPosition>,
{
    type Interpolation = Lerp;

    fn rasterize(
        &self,
        primitive: Line<T>,
    ) -> impl IntoIterator<Item = RasterizationPoint<Line<T>, Self::Interpolation>> {
        let (_line_raster_f32, line_raster_u32) = self
            .target
            .to_raster_primitive(Primitive::new(primitive.clip_positions(), ()));

        line_raster_u32
            .map(|[start, end]| {
                bresenham(start, end).map(move |(raster, lerp)| {
                    let lerp = Lerp(lerp);
                    /*Fragment {
                        raster,
                        position: lerp.interpolate(primitive.into()),
                        interpolation: lerp,
                    }*/
                    todo!();
                })
            })
            .into_iter()
            .flatten()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriRasterizer {
    target: ToTargetRaster,
}

impl TriRasterizer {
    pub fn new(target_size: Vector2<u32>) -> Self {
        Self {
            target: ToTargetRaster::new(target_size),
        }
    }
}

impl<T> Rasterize<Tri<T>> for TriRasterizer
where
    T: AsRef<ClipPosition> + Clone + 'static,
{
    type Interpolation = Barycentric<3>;

    fn rasterize(
        &self,
        primitive: Tri<T>,
    ) -> impl IntoIterator<Item = RasterizationPoint<Tri<T>, Self::Interpolation>> {
        let (tri_raster_f32, tri_raster_u32) = self
            .target
            .to_raster_primitive(Primitive::new(primitive.clip_positions(), ()));

        fn shoelace([a, b, c]: [Point2<f32>; 3]) -> f32 {
            // omitted factor 1/2 since it is cancelled out when calculating barycentric
            // coordinates.
            (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x)
        }

        tri_raster_u32
            .map(move |tri_raster_u32| {
                let total_area = shoelace(tri_raster_f32);

                let barycentric = move |raster_u32: Point2<u32>| {
                    // https://haqr.eu/tinyrenderer/barycentric/

                    let raster_f32 = Point2::from(raster_u32.coords.cast::<f32>());

                    Barycentric::from([
                        shoelace([raster_f32, tri_raster_f32[1], tri_raster_f32[2]]) / total_area,
                        shoelace([raster_f32, tri_raster_f32[2], tri_raster_f32[0]]) / total_area,
                        shoelace([raster_f32, tri_raster_f32[0], tri_raster_f32[1]]) / total_area,
                    ])
                };

                scanlines(tri_raster_u32).flat_map(move |scanline| {
                    let primitive = primitive.clone();
                    scanline.into_iter().filter_map(move |raster| {
                        (raster.x < self.target.size.x && raster.y < self.target.size.y).then(
                            || {
                                let barycentric = barycentric(raster);
                                let clip_position = barycentric.interpolate(
                                    primitive
                                        .clip_positions()
                                        .map(|ClipPosition(vertex)| vertex),
                                );
                                let depth = clip_position.z;
                                let perspective_divisor = clip_position.w;

                                RasterizationPoint {
                                    destination: FragmentDestination {
                                        position: raster,
                                        sample_index: None,
                                    },
                                    coverage_mask: !0,
                                    front_face: None,
                                    perspective_divisor,
                                    depth,
                                    primitive_vertices: primitive.clone(),
                                    interpolation: barycentric,
                                }
                            },
                        )
                    })
                })
            })
            .into_iter()
            .flatten()
    }
}

/*
#[derive(Clone, Copy, Debug)]
pub struct BarycentricFromTri {
    total_area: f32,
    tri: Tri,
}

impl BarycentricFromTri {
    pub fn new(tri: Tri) -> Self {
        /*Self {
            total_area: todo!(),
            tri,
        }*/
        todo!();
    }
}
 */

// -------------snip------------
//
// might not need these

/*#[derive(Clone, Copy, Debug)]
pub struct LinePointRasterizer {
    point_rasterizer: PointRasterizer,
}

impl LinePointRasterizer {
    pub fn new(target_size: Vector2<u32>) -> Self {
        Self {
            point_rasterizer: PointRasterizer::new(target_size),
        }
    }
}

impl<T> Rasterize<Line<T>> for LinePointRasterizer
where
    T: AsRef<ClipPosition>,
{
    type Interpolation = NoInterpolation;

    fn rasterize(
        &self,
        primitive: Line<T>,
    ) -> impl IntoIterator<Item = RasterizationPoint<Line<T>, Self::Interpolation>> {
        primitive
            .map(|point| {
                let clip_position = point.as_ref();
                let rasterization_point = self.point_rasterizer.rasterize(clip_position);
                todo!();
            })
            .into_iter()
            .flatten()
    }
}*/

/*
#[derive(Clone, Copy, Debug)]
pub struct TriPointRasterizer {
    point_rasterizer: PointRasterizer,
}

impl TriPointRasterizer {
    pub fn new(target_size: Vector2<u32>) -> Self {
        Self {
            point_rasterizer: PointRasterizer::new(target_size),
        }
    }
}

impl Rasterize<Tri> for TriPointRasterizer {
    type Interpolation = NoInterpolation;

    fn rasterize(
        &self,
        primitive: Tri,
    ) -> impl IntoIterator<Item = RasterizationPoint<Self::Interpolation>> {
        #![allow(unused)]
        /*primitive
        .0
        .map(|point| self.point_rasterizer.rasterize(point.into()))
        .into_iter()
        .flatten()*/
        todo!();
        []
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriLineRasterizer {
    line_rasterizer: LineRasterizer,
}

impl TriLineRasterizer {
    pub fn new(target_size: Vector2<u32>) -> Self {
        Self {
            line_rasterizer: LineRasterizer::new(target_size),
        }
    }
}

impl Rasterize<Tri> for TriLineRasterizer {
    type Interpolation = Lerp;

    fn rasterize(
        &self,
        primitive: Tri,
    ) -> impl IntoIterator<Item = RasterizationPoint<Self::Interpolation>> {
        #![allow(unused)]
        /*const LINES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

        LINES.map(|[start_index, end_index]| {
            self.line_rasterizer
                .rasterize(Line([primitive.0[start_index], primitive.0[end_index]]))
                .into_iter()
                .map(|fragment| {
                    fragment.map_interpolation(|lerp| {
                        todo!();
                    })
                })
        });*/

        todo!();
        []
    }
} */
