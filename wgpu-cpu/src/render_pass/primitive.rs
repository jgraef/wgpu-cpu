use nalgebra::Vector4;

#[derive(Clone, Copy, Debug, derive_more::From, derive_more::Into)]
pub struct Tri(pub [Vector4<f32>; 3]);

impl Tri {
    pub fn lines(&self) -> [Line; 3] {
        [
            Line([self.0[0], self.0[1]]),
            Line([self.0[1], self.0[2]]),
            Line([self.0[2], self.0[0]]),
        ]
    }

    pub fn front_face(&self, front_face: wgpu::FrontFace) -> wgpu::Face {
        let ab = self.0[1] - self.0[0];
        let ac = self.0[2] - self.0[1];

        let ccw_face = if ab.x * ac.y < ab.y * ac.x {
            wgpu::Face::Front
        }
        else {
            wgpu::Face::Back
        };

        match (ccw_face, front_face) {
            (wgpu::Face::Front, wgpu::FrontFace::Ccw) => wgpu::Face::Front,
            (wgpu::Face::Front, wgpu::FrontFace::Cw) => wgpu::Face::Back,
            (wgpu::Face::Back, wgpu::FrontFace::Ccw) => wgpu::Face::Back,
            (wgpu::Face::Back, wgpu::FrontFace::Cw) => wgpu::Face::Front,
        }
    }
}

impl AsRef<[Vector4<f32>]> for Tri {
    fn as_ref(&self) -> &[Vector4<f32>] {
        self.0.as_slice()
    }
}

#[derive(Clone, Copy, Debug, derive_more::From, derive_more::Into)]
pub struct Line(pub [Vector4<f32>; 2]);

impl AsRef<[Vector4<f32>]> for Line {
    fn as_ref(&self) -> &[Vector4<f32>] {
        self.0.as_slice()
    }
}
