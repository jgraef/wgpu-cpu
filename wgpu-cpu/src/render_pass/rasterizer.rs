use nalgebra::{
    Matrix2x4,
    Point2,
    Vector2,
    Vector4,
};

use crate::{
    render_pass::primitive::{
        Line,
        Tri,
    },
    util::{
        Barycentric,
        bresenham::bresenham,
        lerp,
        scanline::scanlines,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct Fragment<B> {
    pub raster: Point2<u32>,
    pub position: Vector4<f32>,
    pub barycentric: B,
}

#[derive(Debug)]
pub struct Rasterizer {
    target_size: Vector2<u32>,
    to_raster: Matrix2x4<f32>,
    cull_mode: Option<wgpu::Face>,
}

impl Rasterizer {
    pub fn new(target_size: Vector2<u32>, cull_mode: Option<wgpu::Face>) -> Self {
        let raster_size = target_size.cast::<f32>();

        // this matrix maps vectors produced by the pipeline to raster coordinates. the
        // result only needs to be cast to u32 to get a pixel coordinate.
        let mut to_raster = Matrix2x4::default();
        // shift by half the target size
        to_raster[(0, 3)] = 0.5 * raster_size.x + 0.5;
        to_raster[(1, 3)] = 0.5 * raster_size.y + 0.5;
        // scale by half the target size and flip y axis
        to_raster[(0, 0)] = 0.5 * (raster_size.x - 1.0);
        to_raster[(1, 1)] = -0.5 * (raster_size.y - 1.0);

        Self {
            target_size,
            to_raster,
            cull_mode,
        }
    }

    pub fn tri_fill(&self, tri: Tri) -> impl Iterator<Item = Fragment<Barycentric<3>>> {
        let tri_raster_f32 = tri.0.map(|x| Point2::from(self.to_raster * x));

        let tri_raster_u32 = move || {
            Some([
                Point2::from(tri_raster_f32[0].coords.try_cast::<u32>()?),
                Point2::from(tri_raster_f32[1].coords.try_cast::<u32>()?),
                Point2::from(tri_raster_f32[2].coords.try_cast::<u32>()?),
            ])
        };

        fn shoelace([a, b, c]: [Point2<f32>; 3]) -> f32 {
            // omitted factor 1/2 since it is cancelled out when calculating barycentric
            // coordinates.
            (b.y - a.y) * (b.x + a.x) + (c.y - b.y) * (c.x + b.x) + (a.y - c.y) * (a.x + c.x)
        }

        tri_raster_u32()
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
                    scanline.into_iter().map(move |raster| {
                        let barycentric = barycentric(raster);
                        let position = barycentric.interpolate(tri);

                        Fragment {
                            raster,
                            position,
                            barycentric,
                        }
                    })
                })
            })
            .into_iter()
            .flatten()
    }

    pub fn tri_lines(&self, tri: Tri) -> impl Iterator<Item = Fragment<f32>> {
        tri.lines().into_iter().flat_map(|line| self.line(line))
    }

    pub fn line(&self, line: Line) -> impl Iterator<Item = Fragment<f32>> {
        let line_raster = || {
            let start = Point2::from((self.to_raster * line.0[0]).try_cast::<u32>()?);
            let end = Point2::from((self.to_raster * line.0[1]).try_cast::<u32>()?);
            Some((start, end))
        };

        line_raster()
            .map(|(start, end)| {
                bresenham(start, end).map(move |(raster, barycentric)| {
                    Fragment {
                        raster,
                        position: lerp(line.0[0], line.0[1], barycentric),
                        barycentric,
                    }
                })
            })
            .into_iter()
            .flatten()
    }
}
