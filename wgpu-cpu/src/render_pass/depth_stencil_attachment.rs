use nalgebra::Point2;

use crate::texture::{
    TextureInfo,
    TextureViewAttachment,
    TextureWriteGuard,
};

#[derive(Debug)]
pub struct DepthStencilAttachment {
    pub view: TextureViewAttachment,
    pub depth_ops: Option<wgpu::Operations<f32>>,
    pub stencil_ops: Option<wgpu::Operations<u32>>,
}

impl DepthStencilAttachment {
    pub fn new(depth_stencil_attachment: &wgpu::RenderPassDepthStencilAttachment) -> Self {
        Self {
            view: TextureViewAttachment::from_wgpu(&depth_stencil_attachment.view).unwrap(),
            depth_ops: depth_stencil_attachment.depth_ops,
            stencil_ops: depth_stencil_attachment.stencil_ops,
        }
    }
}

#[derive(Debug)]
pub struct AcquiredDepthStencilAttachment<'a> {
    texture_guard: TextureWriteGuard<'a>,
    texture_info: &'a TextureInfo,
    depth_ops: Option<wgpu::Operations<f32>>,
    stencil_ops: Option<wgpu::Operations<u32>>,
}

impl<'a> AcquiredDepthStencilAttachment<'a> {
    pub fn new(depth_stencil_attachment: &'a DepthStencilAttachment) -> Self {
        let texture_guard = depth_stencil_attachment.view.write();
        let texture_info = &depth_stencil_attachment.view.info;

        Self {
            texture_guard,
            texture_info,
            depth_ops: depth_stencil_attachment.depth_ops,
            stencil_ops: depth_stencil_attachment.stencil_ops,
        }
    }

    pub fn load(&mut self) {
        if let Some(depth_ops) = self.depth_ops {
            match depth_ops.load {
                wgpu::LoadOp::Clear(depth) => {
                    self.texture_guard.clear_depth(depth);
                }
                wgpu::LoadOp::Load => {
                    // nop
                }
                wgpu::LoadOp::DontCare(_) => {
                    // nop
                }
            }
        }

        if let Some(stencil_ops) = self.stencil_ops {
            todo!("stencil_ops");
        }
    }

    pub fn store(&mut self) {
        // todo: what to do?
    }

    pub fn get_depth(&self, position: Point2<u32>) -> f32 {
        self.texture_guard.get_depth(position)
    }

    pub fn put_depth(&mut self, position: Point2<u32>, depth: f32) {
        self.texture_guard.put_depth(position, depth);
    }
}
