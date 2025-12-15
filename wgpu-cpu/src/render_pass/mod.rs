pub mod clipper;
pub mod fragment;
pub mod primitive;
pub mod rasterizer;
pub mod state;
pub mod vertex;

use std::{
    ops::Range,
    time::Instant,
};

use derive_more::Debug;
use naga::Binding;

use crate::{
    buffer::BufferSlice,
    command::{
        Command,
        CommandEncoder,
    },
    pipeline::RenderPipeline,
    render_pass::{
        fragment::{
            ColorAttachment,
            DepthStencilAttachment,
        },
        state::State,
    },
};

#[derive(Debug)]
pub struct RenderPassEncoder {
    command_encoder: CommandEncoder,
    commands: Vec<RenderPassSubCommand>,
    descriptor: Option<RenderPassDescriptor>,
}

impl RenderPassEncoder {
    pub fn new(command_encoder: CommandEncoder, desc: &wgpu::RenderPassDescriptor) -> Self {
        Self {
            command_encoder,
            commands: vec![],
            descriptor: Some(RenderPassDescriptor {
                label: desc.label.map(ToOwned::to_owned),
                color_attachments: desc
                    .color_attachments
                    .into_iter()
                    .map(|color_attachment| color_attachment.as_ref().map(ColorAttachment::new))
                    .collect(),
                depth_stencil_attachment: desc
                    .depth_stencil_attachment
                    .as_ref()
                    .map(DepthStencilAttachment::new),
            }),
        }
    }
}

impl wgpu::custom::RenderPassInterface for RenderPassEncoder {
    fn set_pipeline(&mut self, pipeline: &wgpu::custom::DispatchRenderPipeline) {
        let pipeline = pipeline.as_custom::<RenderPipeline>().unwrap().clone();
        self.commands
            .push(RenderPassSubCommand::SetPipeline { pipeline })
    }

    fn set_bind_group(
        &mut self,
        index: u32,
        bind_group: Option<&wgpu::custom::DispatchBindGroup>,
        offsets: &[wgpu::DynamicOffset],
    ) {
        todo!()
    }

    fn set_index_buffer(
        &mut self,
        buffer: &wgpu::custom::DispatchBuffer,
        index_format: wgpu::IndexFormat,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        let buffer_slice = BufferSlice::from_wgpu_dispatch(buffer, offset, size);
        self.commands.push(RenderPassSubCommand::SetIndexBuffer {
            buffer_slice,
            index_format,
        });
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        let buffer_slice = BufferSlice::from_wgpu_dispatch(buffer, offset, size);
        self.commands
            .push(RenderPassSubCommand::SetVertexBuffer { buffer_slice, slot });
    }

    fn set_immediates(&mut self, stages: wgpu::ShaderStages, offset: u32, data: &[u8]) {
        todo!()
    }

    fn set_blend_constant(&mut self, color: wgpu::Color) {
        todo!()
    }

    fn set_scissor_rect(&mut self, x: u32, y: u32, width: u32, height: u32) {
        todo!()
    }

    fn set_viewport(
        &mut self,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    ) {
        todo!()
    }

    fn set_stencil_reference(&mut self, reference: u32) {
        todo!()
    }

    fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.commands.push(RenderPassSubCommand::Draw {
            vertices,
            instances,
        })
    }

    fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.commands.push(RenderPassSubCommand::DrawIndexed {
            indices,
            base_vertex,
            instances,
        })
    }

    fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        todo!()
    }

    fn draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        todo!()
    }

    fn draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        todo!()
    }

    fn draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
    ) {
        todo!()
    }

    fn multi_draw_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        todo!()
    }

    fn multi_draw_indexed_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        todo!()
    }

    fn multi_draw_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &wgpu::custom::DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        todo!()
    }

    fn multi_draw_mesh_tasks_indirect(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count: u32,
    ) {
        todo!()
    }

    fn multi_draw_indexed_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &wgpu::custom::DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        todo!()
    }

    fn multi_draw_mesh_tasks_indirect_count(
        &mut self,
        indirect_buffer: &wgpu::custom::DispatchBuffer,
        indirect_offset: wgpu::BufferAddress,
        count_buffer: &wgpu::custom::DispatchBuffer,
        count_buffer_offset: wgpu::BufferAddress,
        max_count: u32,
    ) {
        todo!()
    }

    fn insert_debug_marker(&mut self, label: &str) {
        todo!()
    }

    fn push_debug_group(&mut self, group_label: &str) {
        todo!()
    }

    fn pop_debug_group(&mut self) {
        todo!()
    }

    fn write_timestamp(&mut self, query_set: &wgpu::custom::DispatchQuerySet, query_index: u32) {
        todo!()
    }

    fn begin_occlusion_query(&mut self, query_index: u32) {
        todo!()
    }

    fn end_occlusion_query(&mut self) {
        todo!()
    }

    fn begin_pipeline_statistics_query(
        &mut self,
        query_set: &wgpu::custom::DispatchQuerySet,
        query_index: u32,
    ) {
        todo!()
    }

    fn end_pipeline_statistics_query(&mut self) {
        todo!()
    }

    fn execute_bundles(
        &mut self,
        render_bundles: &mut dyn Iterator<Item = &wgpu::custom::DispatchRenderBundle>,
    ) {
        todo!()
    }

    fn end(&mut self) {
        // somehow this is not called when the render pass is dropped. is this a bug?

        if let Some(descriptor) = self.descriptor.take() {
            let commands = std::mem::take(&mut self.commands);

            self.command_encoder
                .push(Command::RenderPass(RenderPassCommand {
                    descriptor,
                    commands,
                }));
        }
    }
}

// todo: this might be a bug that we have to call it ourself
impl Drop for RenderPassEncoder {
    fn drop(&mut self) {
        wgpu::custom::RenderPassInterface::end(self)
    }
}

#[derive(Debug)]
pub struct RenderPassDescriptor {
    pub label: Option<String>,
    pub color_attachments: Vec<Option<ColorAttachment>>,
    pub depth_stencil_attachment: Option<DepthStencilAttachment>,
}

#[derive(Debug)]
pub struct RenderPassCommand {
    descriptor: RenderPassDescriptor,
    commands: Vec<RenderPassSubCommand>,
}

impl RenderPassCommand {
    pub fn execute(self) {
        let t_start = Instant::now();

        let mut state = State::new(&self.descriptor);
        state.load();

        for command in self.commands {
            match command {
                RenderPassSubCommand::SetPipeline { pipeline } => {
                    state.set_pipeline(pipeline);
                }
                RenderPassSubCommand::SetIndexBuffer {
                    buffer_slice,
                    index_format,
                } => {
                    state.set_index_buffer(buffer_slice, index_format);
                }
                RenderPassSubCommand::SetVertexBuffer { buffer_slice, slot } => {
                    state.set_vertex_buffer(buffer_slice, slot);
                }
                RenderPassSubCommand::Draw {
                    vertices,
                    instances,
                } => {
                    state.draw(vertices, instances);
                }
                RenderPassSubCommand::DrawIndexed {
                    indices,
                    base_vertex,
                    instances,
                } => {
                    state.draw_indexed(indices, base_vertex, instances);
                }
            }
        }

        state.store();

        let elapsed = t_start.elapsed();
        tracing::debug!(?elapsed, "render pass time");
    }
}

#[derive(derive_more::Debug)]
pub enum RenderPassSubCommand {
    SetPipeline {
        #[debug(skip)]
        pipeline: RenderPipeline,
    },
    SetIndexBuffer {
        buffer_slice: BufferSlice,
        index_format: wgpu::IndexFormat,
    },
    SetVertexBuffer {
        buffer_slice: BufferSlice,
        slot: u32,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
}

#[track_caller]
fn invalid_binding(binding: &Binding) -> ! {
    panic!("Binding not supported: {binding:?}");
}

fn bytes_of_bool_as_u32(b: bool) -> &'static [u8] {
    if b {
        bytemuck::bytes_of(&1u32)
    }
    else {
        bytemuck::bytes_of(&0u32)
    }
}

fn evaluate_compare_function<T>(
    compare_function: wgpu::CompareFunction,
    value: T,
    reference: T,
) -> bool
where
    T: PartialOrd<T>,
{
    match compare_function {
        wgpu::CompareFunction::Never => false,
        wgpu::CompareFunction::Less => value < reference,
        wgpu::CompareFunction::Equal => value == reference,
        wgpu::CompareFunction::LessEqual => value <= reference,
        wgpu::CompareFunction::Greater => value > reference,
        wgpu::CompareFunction::NotEqual => value != reference,
        wgpu::CompareFunction::GreaterEqual => value >= reference,
        wgpu::CompareFunction::Always => true,
    }
}
