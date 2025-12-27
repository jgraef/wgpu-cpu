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
        clipper::ClipPosition,
        primitive::Primitive,
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

#[derive(Clone, Copy, Debug)]
pub struct RasterizerOutput<Inter> {
    pub framebuffer: Point2<u32>,
    pub fragment: Vector4<f32>,
    pub sample_index: Option<NonZero<u8>>,
    pub interpolation: Inter,
}

pub trait Rasterize<const NUM_VERTICES: usize> {
    type Interpolation;

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, NUM_VERTICES>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>>;
}

impl<R, const NUM_VERTICES: usize> Rasterize<NUM_VERTICES> for &R
where
    R: Rasterize<NUM_VERTICES>,
{
    type Interpolation = R::Interpolation;

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, NUM_VERTICES>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>> {
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

        let viewport = ndc.coords.xy().component_mul(&self.scaling) + self.translation;
        let framebuffer = viewport.try_cast::<u32>().map(Point2::from)?;
        if framebuffer.x < self.size.x && framebuffer.y < self.size.y {
            let fragment = Vector4::new(viewport.x, viewport.y, ndc.z, perspective_divisor);
            Some(RasterPoint {
                framebuffer,
                fragment,
            })
        }
        else {
            None
        }
    }

    pub fn to_raster_primitive<const N: usize>(
        &self,
        primitive: Primitive<ClipPosition, N>,
    ) -> Option<[RasterPoint; N]> {
        primitive
            .vertices
            .try_map_(|clip_position| self.to_raster(clip_position))
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullRasterizer;

impl<const NUM_VERTICES: usize> Rasterize<NUM_VERTICES> for NullRasterizer {
    type Interpolation = NoInterpolation;

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, NUM_VERTICES>,
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

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, 1>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>> {
        let raster = self.target.to_raster(primitive.vertices[0]);

        raster.map(|raster| {
            RasterizerOutput {
                framebuffer: raster.framebuffer,
                fragment: raster.fragment,
                sample_index: None,
                interpolation: NoInterpolation,
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

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, 2>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>> {
        let start = self.target.to_raster(primitive.vertices[0]);
        let end = self.target.to_raster(primitive.vertices[1]);

        start
            .zip(end)
            .map(|(start, end)| {
                bresenham(start.framebuffer, end.framebuffer).map(move |(framebuffer, lerp)| {
                    let lerp = Lerp(lerp);
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

    fn rasterize(
        &self,
        primitive: Primitive<ClipPosition, 3>,
    ) -> impl IntoIterator<Item = RasterizerOutput<Self::Interpolation>> {
        // todo: take TriFace with primitive as input so we don't have to recompute the
        // area
        let a = self.target.to_raster(primitive.vertices[0]);
        let b = self.target.to_raster(primitive.vertices[1]);
        let c = self.target.to_raster(primitive.vertices[2]);

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
                    let primitive = primitive.clone();

                    scanline.into_iter().filter_map(move |framebuffer| {
                        let viewport = framebuffer.coords.cast::<f32>().into();

                        let barycentric = barycentric(viewport);
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
