use std::sync::Arc;

use parking_lot::Mutex;

use crate::render_pass::{
    RenderPassCommand,
    RenderPassEncoder,
};

#[derive(Clone, Debug)]
pub struct CommandEncoder {
    inner: Arc<Mutex<Inner>>,
}

impl CommandEncoder {
    pub fn new(descriptor: wgpu::wgt::CommandEncoderDescriptor<Option<String>>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner {
                descriptor,
                commands: vec![],
            })),
        }
    }

    pub fn push_all(&self, buffer: Vec<Command>) {
        let mut inner = self.inner.lock();
        inner.commands.extend(buffer);
    }

    pub fn push(&self, command: Command) {
        let mut inner = self.inner.lock();
        inner.commands.push(command);
    }
}

impl wgpu::custom::CommandEncoderInterface for CommandEncoder {
    fn copy_buffer_to_buffer(
        &self,
        source: &wgpu::custom::DispatchBuffer,
        source_offset: wgpu::BufferAddress,
        destination: &wgpu::custom::DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
        copy_size: Option<wgpu::BufferAddress>,
    ) {
        todo!()
    }

    fn copy_buffer_to_texture(
        &self,
        source: wgpu::TexelCopyBufferInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        todo!()
    }

    fn copy_texture_to_buffer(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyBufferInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        todo!()
    }

    fn copy_texture_to_texture(
        &self,
        source: wgpu::TexelCopyTextureInfo<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        copy_size: wgpu::Extent3d,
    ) {
        todo!()
    }

    fn begin_compute_pass(
        &self,
        desc: &wgpu::ComputePassDescriptor<'_>,
    ) -> wgpu::custom::DispatchComputePass {
        todo!()
    }

    fn begin_render_pass(
        &self,
        desc: &wgpu::RenderPassDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPass {
        wgpu::custom::DispatchRenderPass::custom(RenderPassEncoder::new(self.clone(), desc))
    }

    fn finish(&mut self) -> wgpu::custom::DispatchCommandBuffer {
        let commands = {
            let mut commands = self.inner.lock();
            std::mem::take(&mut commands.commands)
        };

        wgpu::custom::DispatchCommandBuffer::custom(CommandBuffer {
            commands: Arc::new(Mutex::new(commands)),
        })
    }

    fn clear_texture(
        &self,
        texture: &wgpu::custom::DispatchTexture,
        subresource_range: &wgpu::ImageSubresourceRange,
    ) {
        todo!()
    }

    fn clear_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferAddress>,
    ) {
        todo!()
    }

    fn insert_debug_marker(&self, label: &str) {
        todo!()
    }

    fn push_debug_group(&self, label: &str) {
        todo!()
    }

    fn pop_debug_group(&self) {
        todo!()
    }

    fn write_timestamp(&self, query_set: &wgpu::custom::DispatchQuerySet, query_index: u32) {
        todo!()
    }

    fn resolve_query_set(
        &self,
        query_set: &wgpu::custom::DispatchQuerySet,
        first_query: u32,
        query_count: u32,
        destination: &wgpu::custom::DispatchBuffer,
        destination_offset: wgpu::BufferAddress,
    ) {
        todo!()
    }

    fn mark_acceleration_structures_built<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a wgpu::Blas>,
        tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        todo!()
    }

    fn build_acceleration_structures<'a>(
        &self,
        blas: &mut dyn Iterator<Item = &'a wgpu::BlasBuildEntry<'a>>,
        tlas: &mut dyn Iterator<Item = &'a wgpu::Tlas>,
    ) {
        todo!()
    }

    fn transition_resources<'a>(
        &mut self,
        buffer_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::BufferTransition<&'a wgpu::custom::DispatchBuffer>,
        >,
        texture_transitions: &mut dyn Iterator<
            Item = wgpu::wgt::TextureTransition<&'a wgpu::custom::DispatchTexture>,
        >,
    ) {
        todo!()
    }
}

#[derive(Debug, Default)]
pub struct CommandBuffer {
    pub commands: Arc<Mutex<Vec<Command>>>,
}

impl CommandBuffer {
    pub fn take(&self) -> Vec<Command> {
        let mut commands = self.commands.lock();
        std::mem::take(&mut *commands)
    }
}

impl wgpu::custom::CommandBufferInterface for CommandBuffer {}

#[derive(Debug)]
pub enum Command {
    RenderPass(RenderPassCommand),
}

#[derive(Debug, Default)]
struct Inner {
    descriptor: wgpu::wgt::CommandEncoderDescriptor<Option<String>>,
    commands: Vec<Command>,
}
