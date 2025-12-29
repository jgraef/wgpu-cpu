//! https://gpuweb.github.io/gpuweb/#rasterization

use std::{
    num::NonZero,
    ops::Index,
};

use nalgebra::{
    Point2,
    Point3,
    Vector2,
    Vector4,
};

use crate::{
    render_pass::{
        clipper::{
            ClipPosition,
            Clipped,
        },
        primitive::Primitive,
    },
    util::{
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

#[derive(Clone, Copy, Debug)]
pub struct RasterizerOutput<Inter> {
    pub framebuffer: Point2<u32>,
    pub fragment: Vector4<f32>,
    pub sample_index: Option<NonZero<u8>>,
    pub interpolation: Inter,
}

pub trait Rasterize<const NUM_VERTICES: usize> {
    type Interpolation;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, Self::Interpolation>, NUM_VERTICES, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>
    where
        Vertex: AsRef<ClipPosition>;
}

impl<R, const NUM_VERTICES: usize> Rasterize<NUM_VERTICES> for &R
where
    R: Rasterize<NUM_VERTICES>,
{
    type Interpolation = R::Interpolation;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, Self::Interpolation>, NUM_VERTICES, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        R::rasterize(self, primitive)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RasterPoint {
    pub framebuffer: Point2<u32>,
    // viewport coordinates (framebuffer as f32), depth, perspective divisor
    pub fragment: Vector4<f32>,
}

#[derive(Clone, Copy, Debug)]
pub struct ToRaster {
    pub size: Vector2<u32>,
    pub translation: Vector2<f32>,
    pub scaling: Vector2<f32>,
}

impl ToRaster {
    pub fn new(target_size: Vector2<u32>) -> Self {
        let size_f32 = target_size.cast::<f32>();

        let translation = Vector2::new(0.5 * size_f32.x, 0.5 * size_f32.y);

        let scaling = Vector2::new(0.5 * (size_f32.x - 1.0), -0.5 * (size_f32.y - 1.0));

        Self {
            size: target_size,
            translation,
            scaling,
        }
    }

    pub fn to_raster(&self, clip_position: ClipPosition) -> Option<RasterPoint> {
        let ndc = Point3::from_homogeneous(clip_position.0)?;
        let perspective_divisor = 1.0 / clip_position.0.w;

        let mut viewport = ndc.coords.xy().component_mul(&self.scaling) + self.translation;
        viewport.x = viewport.x.max(0.0);
        viewport.y = viewport.y.max(0.0);

        let mut framebuffer = viewport.try_cast::<u32>().unwrap_or_default();
        framebuffer.x = framebuffer.x.min(self.size.x - 1);
        framebuffer.y = framebuffer.y.min(self.size.y - 1);

        let fragment = Vector4::new(viewport.x, viewport.y, ndc.z, perspective_divisor);
        Some(RasterPoint {
            framebuffer: framebuffer.into(),
            fragment,
        })
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullRasterizer;

impl<const NUM_VERTICES: usize> Rasterize<NUM_VERTICES> for NullRasterizer {
    type Interpolation = NoInterpolation;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, NoInterpolation>, NUM_VERTICES, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<NoInterpolation>> {
        []
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointRasterizer {
    target: ToRaster,
}

impl PointRasterizer {
    pub fn new(target_size: nalgebra::Vector2<u32>) -> Self {
        Self {
            target: ToRaster::new(target_size),
        }
    }
}

impl Rasterize<1> for PointRasterizer {
    type Interpolation = NoInterpolation;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, NoInterpolation>, 1, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        let raster = self.target.to_raster(*primitive.vertices[0].as_ref());

        raster.map(|raster| {
            RasterizerOutput {
                framebuffer: raster.framebuffer,
                fragment: raster.fragment,
                sample_index: None,
                interpolation: NoInterpolation::default(),
            }
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LineRasterizer {
    target: ToRaster,
}

impl LineRasterizer {
    pub fn new(target_size: nalgebra::Vector2<u32>) -> Self {
        Self {
            target: ToRaster::new(target_size),
        }
    }
}

impl Rasterize<2> for LineRasterizer {
    type Interpolation = Lerp;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, Lerp>, 2, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        let start = self.target.to_raster(*primitive.vertices[0].as_ref());
        let end = self.target.to_raster(*primitive.vertices[1].as_ref());

        start
            .zip(end)
            .map(|(start, end)| {
                bresenham(start.framebuffer, end.framebuffer).map(move |(framebuffer, lerp)| {
                    let lerp = Lerp(lerp).interpolate(
                        primitive
                            .vertices
                            .each_ref()
                            .map(|vertex| vertex.interpolation),
                    );
                    let fragment = lerp.interpolate([start.fragment, end.fragment]);

                    RasterizerOutput {
                        framebuffer,
                        fragment,
                        sample_index: None,
                        interpolation: lerp,
                    }
                })
            })
            .into_iter()
            .flatten()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TriRasterizer {
    target: ToRaster,
}

impl TriRasterizer {
    pub fn new(target_size: Vector2<u32>) -> Self {
        Self {
            target: ToRaster::new(target_size),
        }
    }
}

impl Rasterize<3> for TriRasterizer {
    type Interpolation = Barycentric<3>;

    fn rasterize<Vertex, Face>(
        &self,
        primitive: &Primitive<Clipped<Vertex, Barycentric<3>>, 3, Face>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        let a = self.target.to_raster(*primitive.vertices[0].as_ref());
        let b = self.target.to_raster(*primitive.vertices[1].as_ref());
        let c = self.target.to_raster(*primitive.vertices[2].as_ref());

        fn shoelace(a: Point2<f32>, b: Point2<f32>, c: Point2<f32>) -> f32 {
            // omitted factor 1/2 since it is cancelled out when calculating barycentric
            // coordinates.
            (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x)
        }

        a.zip(b)
            .zip(c)
            .map(move |((a, b), c)| {
                let vp_a = a.fragment.xy().into();
                let vp_b = b.fragment.xy().into();
                let vp_c = c.fragment.xy().into();

                // do we have to do this here? can we have the scanline iterator yield something
                // to interpolate with?
                let total_area = shoelace(vp_a, vp_b, vp_c);

                let barycentric = move |vp_p: Point2<f32>| {
                    // https://haqr.eu/tinyrenderer/barycentric/

                    Barycentric::from([
                        shoelace(vp_p, vp_b, vp_c) / total_area,
                        shoelace(vp_p, vp_c, vp_a) / total_area,
                        shoelace(vp_p, vp_a, vp_b) / total_area,
                    ])
                };

                scanlines([a.framebuffer, b.framebuffer, c.framebuffer]).flat_map(move |scanline| {
                    scanline.into_iter().filter_map(move |framebuffer| {
                        let viewport = framebuffer.coords.cast::<f32>().into();

                        let barycentric = barycentric(viewport);
                        let barycentric = barycentric.interpolate(
                            primitive
                                .vertices
                                .each_ref()
                                .map(|vertex| vertex.interpolation),
                        );

                        let fragment =
                            barycentric.interpolate([a.fragment, b.fragment, c.fragment]);

                        Some(RasterizerOutput {
                            framebuffer,
                            fragment,
                            sample_index: None,
                            interpolation: barycentric,
                        })
                    })
                })
            })
            .into_iter()
            .flatten()
    }
}
