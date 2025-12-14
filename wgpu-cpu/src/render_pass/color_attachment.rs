use nalgebra::{
    Point2,
    Point3,
    Vector4,
};

use crate::texture::{
    TextureInfo,
    TextureViewAttachment,
    TextureWriteGuard,
};

#[derive(Debug)]
pub struct ColorAttachment {
    pub view: TextureViewAttachment,
    pub depth_slice: u32,
    pub resolve_target: Option<TextureViewAttachment>,
    pub ops: wgpu::Operations<wgpu::Color>,
}

impl ColorAttachment {
    pub fn new(color_attachment: &wgpu::RenderPassColorAttachment) -> Self {
        Self {
            view: TextureViewAttachment::from_wgpu(&color_attachment.view).unwrap(),
            depth_slice: color_attachment.depth_slice.unwrap_or_default(),
            resolve_target: color_attachment
                .resolve_target
                .map(|texture| TextureViewAttachment::from_wgpu(texture).unwrap()),
            ops: color_attachment.ops,
        }
    }
}

#[derive(Debug)]
pub struct AcquiredColorAttachment<'color> {
    texture_guard: TextureWriteGuard<'color>,
    texture_info: &'color TextureInfo,
    ops: wgpu::Operations<wgpu::Color>,
    depth_slice: u32,
}

impl<'color> AcquiredColorAttachment<'color> {
    pub fn new(color_attachment: &'color ColorAttachment) -> Self {
        let texture_guard = color_attachment.view.write();
        let texture_info = &color_attachment.view.info;

        if color_attachment.resolve_target.is_some() {
            todo!("color attachment resolve target");
        }

        Self {
            texture_guard,
            texture_info,
            ops: color_attachment.ops,
            depth_slice: color_attachment.depth_slice,
        }
    }

    pub fn load(&mut self) {
        match self.ops.load {
            wgpu::LoadOp::Clear(clear_color) => {
                self.texture_guard.clear_color(clear_color);
            }
            wgpu::LoadOp::Load => {
                // nop
            }
            wgpu::LoadOp::DontCare(_) => {
                // nop
            }
        }
    }

    pub fn store(&mut self) {
        // todo: what to do?
        match self.ops.store {
            wgpu::StoreOp::Store => {}
            wgpu::StoreOp::Discard => {}
        }
    }

    pub fn put_pixel(&mut self, position: Point2<u32>, color: Vector4<f32>) {
        let position = Point3::new(position.x, position.y, self.depth_slice);
        self.texture_guard.put_pixel(position, color);
    }
}
