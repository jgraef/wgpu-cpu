//! https://gpuweb.github.io/gpuweb/#primitive-clipping

// https://en.wikipedia.org/wiki/Clip_coordinates lists clipping algorithms
// !(v.x >= -v.w && v.x <= v.w && v.y >= -v.w && v.y <= v.w && v.z >= -v.w &&
// v.z <= v.w)

use std::ops::Index;

use bytemuck::{
    Pod,
    Zeroable,
};
use nalgebra::Vector4;

use crate::render_pass::primitive::Primitive;

#[derive(Clone, Copy, Debug, Default, Pod, Zeroable, derive_more::From, derive_more::Into)]
#[repr(C)]
pub struct ClipPosition(pub Vector4<f32>);

impl AsRef<ClipPosition> for ClipPosition {
    fn as_ref(&self) -> &ClipPosition {
        self
    }
}

impl AsMut<ClipPosition> for ClipPosition {
    fn as_mut(&mut self) -> &mut ClipPosition {
        self
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Clipped<T> {
    pub unclipped: T,
    pub clipped: ClipPosition,
}

impl<T> AsRef<ClipPosition> for Clipped<T> {
    fn as_ref(&self) -> &ClipPosition {
        &self.clipped
    }
}

impl<T> AsRef<Clipped<T>> for Clipped<T> {
    fn as_ref(&self) -> &Clipped<T> {
        self
    }
}

impl<T> Clipped<T>
where
    T: AsRef<ClipPosition>,
{
    pub fn passhtrough(unclipped: T) -> Self {
        let clipped = *unclipped.as_ref();
        Self { unclipped, clipped }
    }
}

pub trait Clip<const N: usize> {
    fn clip<Vertex, Face>(
        &self,
        primitive: Primitive<Vertex, N, Face>,
    ) -> impl IntoIterator<Item = Primitive<Clipped<Vertex>, N, Face>>
    where
        Vertex: AsRef<ClipPosition>;
}

impl<C, const N: usize> Clip<N> for &C
where
    C: Clip<N>,
{
    fn clip<Vertex, Face>(
        &self,
        primitive: Primitive<Vertex, N, Face>,
    ) -> impl IntoIterator<Item = Primitive<Clipped<Vertex>, N, Face>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        C::clip(self, primitive)
    }
}

impl<C, const N: usize> Clip<N> for &mut C
where
    C: Clip<N>,
{
    fn clip<Vertex, Face>(
        &self,
        primitive: Primitive<Vertex, N, Face>,
    ) -> impl IntoIterator<Item = Primitive<Clipped<Vertex>, N, Face>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        C::clip(self, primitive)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NoClipper;

impl<const N: usize> Clip<N> for NoClipper {
    fn clip<Vertex, Face>(
        &self,
        primitive: Primitive<Vertex, N, Face>,
    ) -> impl IntoIterator<Item = Primitive<Clipped<Vertex>, N, Face>>
    where
        Vertex: AsRef<ClipPosition>,
    {
        [primitive.map_vertices(Clipped::passhtrough)]
    }
}

/// Clip plane defined by a 3D homogeneous normal vector.
///
/// We use the convention that the normal vector points into the clip volume.
#[derive(Clone, Copy, Debug)]
pub struct ClipPlane(pub Vector4<f32>);

impl ClipPlane {
    pub fn clip_distance(&self, point: ClipPosition) -> f32 {
        point.0.dot(&self.0)
    }
}

/// Clip volume define by 6 clip planes.
#[derive(Clone, Copy, Debug)]
pub struct ClipVolume(pub [ClipPlane; 6]);

impl ClipVolume {
    /// Clip planes used by WebGPU
    #[rustfmt::skip]
    const WEBGPU: Self = Self([
                                                        // point is outside if dot(point, plane) < 0.
                                                        // which corresponds to:
        ClipPlane(Vector4::new( 1.0,  0.0,  0.0, 1.0)), // left:   x < -w
        ClipPlane(Vector4::new(-1.0,  0.0,  0.0, 1.0)), // right:  x >  w
        ClipPlane(Vector4::new( 0.0,  1.0,  0.0, 1.0)), // bottom: y < -w
        ClipPlane(Vector4::new( 0.0, -1.0,  0.0, 1.0)), // top:    y >  w
        ClipPlane(Vector4::new( 0.0,  0.0,  1.0, 0.0)), // front:  z <  0
        ClipPlane(Vector4::new( 0.0,  0.0, -1.0, 1.0)), // back:   z >  w

    ]);

    /// Calculates the signed distance of a point to a single clip plane.
    ///
    /// This is negative if the point is outside the clip volume.
    pub fn clip_distance(&self, plane: usize, point: ClipPosition) -> f32 {
        self.0[plane].clip_distance(point)
    }

    /// Calculates signed clip distances for all clip planes in the volume.
    ///
    /// Each returned clip distance is positive if the point is considered
    /// inside the clip volume relative to that plane.
    pub fn clip_distances(&self, point: ClipPosition) -> ClipDistances {
        ClipDistances(self.0.map(|clip_plane| clip_plane.clip_distance(point)))
    }
}

impl Default for ClipVolume {
    fn default() -> Self {
        Self::WEBGPU
    }
}

impl Index<usize> for ClipVolume {
    type Output = ClipPlane;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

/// Holds distances to each clip plane for a point
#[derive(Clone, Copy, Debug)]
pub struct ClipDistances(pub [f32; 6]);

impl Index<usize> for ClipDistances {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

pub type LineClipper = cohen_sutherland::CohenSutherland;

pub mod cohen_sutherland {
    //! Cohen-Sutherland line clipping algorithm
    //!
    //! This has been adapted from [wikipedia][1] and [this blog post][2] for
    //! use with 3D homogeneous coordinates.
    //!
    //! [1]: https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
    //! [2]: https://chaosinmotion.com/2016/05/22/3d-clipping-in-homogeneous-coordinates/

    use bitflags::bitflags;

    use crate::{
        render_pass::{
            clipper::{
                Clip,
                ClipDistances,
                ClipPosition,
                ClipVolume,
                Clipped,
            },
            primitive::Primitive,
        },
        util::interpolation::lerp,
    };

    bitflags! {
        /// Each bit describes if the point in question is inside relative to that plane
        #[derive(Clone, Copy, Debug, Default, PartialEq , Eq)]
        pub struct OutCode: u8 {
            const LEFT   = 0b000001;
            const RIGHT  = 0b000010;
            const BOTTOM = 0b000100;
            const TOP    = 0b001000;
            const FRONT  = 0b010000;
            const BACK   = 0b100000;
        }
    }

    impl OutCode {
        pub const fn for_clip_plane(i: usize) -> Self {
            Self::from_bits_retain(1 << i)
        }

        pub fn from_clip_distances(clip_distances: ClipDistances) -> Self {
            let mut outcode = Self::empty();

            for i in 0..6 {
                if clip_distances.0[i] < 0.0 {
                    outcode |= Self::for_clip_plane(i);
                }
            }

            outcode
        }
    }

    pub fn clip(
        mut points: [ClipPosition; 2],
        clip_volume: &ClipVolume,
    ) -> Option<[ClipPosition; 2]> {
        let clip_distances = points.map(|point| clip_volume.clip_distances(point));
        let outcodes = clip_distances.map(OutCode::from_clip_distances);

        // a | b
        let outcodes_or = outcodes[0].union(outcodes[1]);
        // a & b
        let outcodes_and = outcodes[0].intersection(outcodes[1]);

        //dbg!(points);
        //dbg!(clip_distances);
        //dbg!(outcodes);
        //dbg!(outcodes_or);
        //dbg!(outcodes_and);

        if outcodes_or.is_empty() {
            // both points inside clip volume: trivial accept
            return Some(points);
        }
        else if !outcodes_and.is_empty() {
            // both points share an outside zone: trivial reject
            return None;
        }
        else {
            // clip

            // end points along the line as [0, 1]
            // todo: shouldn't alpha[1] = 1.0?
            //let mut alphas = [0.0; 2];
            let mut alphas = [0.0, 1.0];

            for clip_plane in 0..6 {
                let clip_plane_outcode = OutCode::for_clip_plane(clip_plane);

                if outcodes_or.contains(clip_plane_outcode) {
                    //println!("clip against {clip_plane}: {clip_plane_outcode:?}");

                    // calculate alpha: the intersection along the line
                    let alpha = {
                        let a = clip_distances[0][clip_plane];
                        let b = clip_distances[1][clip_plane];
                        a / (a - b)
                    };
                    //dbg!(alpha);

                    // adjust endpoint alpha depending on which one is outside
                    if outcodes[0].contains(clip_plane_outcode) {
                        if alpha > alphas[0] {
                            alphas[0] = alpha;
                        }
                    }
                    else {
                        if alpha < alphas[1] {
                            alphas[1] = alpha;
                        }
                    };
                    //dbg!(alphas);

                    if alphas[0] > alphas[1] {
                        // in this case the line was clipped to a "negative length", i.e. both
                        // points are outside in different regions and the line segment inside the
                        // clip volume is empty.
                        return None;
                    }
                }
            }

            // update points
            let mut update_point = |i: usize| {
                let updated = lerp(points[0].0, points[1].0, alphas[i]);
                //println!(
                //    "update point {i} with alpha={}: {:?} -> {:?}",
                //    alphas[i], points[i].0, updated
                //);
                points[i].0 = updated;
            };
            if !outcodes[0].is_empty() {
                update_point(0);
            }
            if !outcodes[1].is_empty() {
                update_point(1);
            }

            Some(points)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct CohenSutherland {
        pub clip_planes: ClipVolume,
    }

    impl Clip<2> for CohenSutherland {
        fn clip<Vertex, Face>(
            &self,
            primitive: Primitive<Vertex, 2, Face>,
        ) -> impl IntoIterator<Item = Primitive<Clipped<Vertex>, 2, Face>>
        where
            Vertex: AsRef<ClipPosition>,
        {
            // todo: do we actually need to do interpolation here? we could have `clip`
            // return the alpha values with the points and use that.
            // then we could possibly just return some interpolation (`Lerp` or
            // `Barycentric<N>`) with the primitives. but we think that clipping
            // the positions is enough as the rasterizer will take care of interpolation

            if let Some(clipped) = clip(primitive.clip_positions(), &self.clip_planes) {
                let [a, b] = primitive.vertices;
                Some(Primitive::new(
                    [
                        Clipped {
                            unclipped: a,
                            clipped: clipped[0],
                        },
                        Clipped {
                            unclipped: b,
                            clipped: clipped[1],
                        },
                    ],
                    primitive.face,
                ))
            }
            else {
                None
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use approx::assert_abs_diff_eq;
        use nalgebra::Vector4;

        use crate::render_pass::clipper::{
            ClipVolume,
            cohen_sutherland::clip,
        };

        #[test]
        fn line_trivially_inside() {
            let a = Vector4::new(-0.5, -0.5, 0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, a.0);
            assert_abs_diff_eq!(out[1].0, b.0);
        }

        #[test]
        fn line_trivially_outside() {
            let a = Vector4::new(-1.5, -1.5, 0.5, 1.0).into();
            let b = Vector4::new(-1.5, 1.5, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(1.5, -1.5, 0.5, 1.0).into();
            let b = Vector4::new(1.5, 1.5, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(-1.5, -1.5, 0.5, 1.0).into();
            let b = Vector4::new(1.5, -1.5, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(-1.5, 1.5, 0.5, 1.0).into();
            let b = Vector4::new(1.5, 1.5, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());
        }

        #[test]
        fn line_non_trivially_outside() {
            let a = Vector4::new(-10.0, 0.0, 0.5, 1.0).into();
            let b = Vector4::new(0.0, -10.0, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(10.0, 0.0, 0.5, 1.0).into();
            let b = Vector4::new(0.0, -10.0, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(-10.0, 0.0, 0.5, 1.0).into();
            let b = Vector4::new(0.0, 10.0, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());

            let a = Vector4::new(10.0, 0.0, 0.5, 1.0).into();
            let b = Vector4::new(0.0, 10.0, 0.5, 1.0).into();
            let out = clip([a, b], &ClipVolume::WEBGPU);
            assert!(out.is_none());
        }

        #[test]
        fn clip_line_against_left() {
            let a = Vector4::new(-1.5, -0.5, 0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(-1.0, -0.25, 0.5, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }

        #[test]
        fn clip_line_against_right() {
            let a = Vector4::new(1.5, -0.5, 0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(1.0, 0.0, 0.5, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }

        #[test]
        fn clip_line_against_bottom() {
            let a = Vector4::new(-0.5, -1.5, 0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(-0.25, -1.0, 0.5, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }

        #[test]
        fn clip_line_against_top() {
            let a = Vector4::new(-0.5, 1.5, 0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(0.0, 1.0, 0.5, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }

        #[test]
        fn clip_line_against_back() {
            let a = Vector4::new(0.5, -0.5, 1.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(0.5, 0.0, 1.0, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }

        #[test]
        fn clip_line_against_front() {
            let a = Vector4::new(0.5, -0.5, -0.5, 1.0).into();
            let b = Vector4::new(0.5, 0.5, 0.5, 1.0).into();
            let c = Vector4::new(0.5, 0.0, 0.0, 1.0);

            let out = clip([a, b], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, c);
            assert_abs_diff_eq!(out[1].0, b.0);

            let out = clip([b, a], &ClipVolume::WEBGPU).unwrap();
            assert_abs_diff_eq!(out[0].0, b.0);
            assert_abs_diff_eq!(out[1].0, c);
        }
    }
}
