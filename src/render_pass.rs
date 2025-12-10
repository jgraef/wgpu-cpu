use crate::{
    command::{
        Command,
        CommandEncoder,
    },
    texture::TextureViewAttachment,
};

#[derive(Debug)]
pub struct RenderPass {
    command_encoder: CommandEncoder,
    commands: Vec<RenderPassSubCommand>,
    ended: bool,
    label: Option<String>,
    color_attachments: Vec<Option<ColorAttachment>>,
    // todo: depth_stencil_attachment
}

impl RenderPass {
    pub fn new(command_encoder: CommandEncoder, desc: &wgpu::RenderPassDescriptor) -> Self {
        Self {
            command_encoder,
            commands: vec![],
            ended: false,
            label: desc.label.map(ToOwned::to_owned),
            color_attachments: desc
                .color_attachments
                .into_iter()
                .map(|color_attachment| color_attachment.as_ref().map(ColorAttachment::new))
                .collect(),
        }
    }
}

impl wgpu::custom::RenderPassInterface for RenderPass {
    fn set_pipeline(&mut self, pipeline: &wgpu::custom::DispatchRenderPipeline) {
        todo!()
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
        todo!()
    }

    fn set_vertex_buffer(
        &mut self,
        slot: u32,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: Option<wgpu::BufferSize>,
    ) {
        todo!()
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

    fn draw(&mut self, vertices: std::ops::Range<u32>, instances: std::ops::Range<u32>) {
        todo!()
    }

    fn draw_indexed(
        &mut self,
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    ) {
        todo!()
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

        if !self.ended {
            let color_attachments = std::mem::take(&mut self.color_attachments);
            let commands = std::mem::take(&mut self.commands);

            self.command_encoder
                .push(Command::RenderPass(RenderPassCommand {
                    label: self.label.clone(),
                    color_attachments,
                    commands,
                }));
            self.ended = true;
        }
    }
}

// todo: this might be a bug that we have to call it ourself
impl Drop for RenderPass {
    fn drop(&mut self) {
        wgpu::custom::RenderPassInterface::end(self)
    }
}

#[derive(Debug)]
pub struct ColorAttachment {
    pub view: TextureViewAttachment,
    pub depth_slice: Option<u32>,
    pub resolve_target: Option<TextureViewAttachment>,
    pub ops: wgpu::Operations<wgpu::Color>,
}

impl ColorAttachment {
    pub fn new(color_attachment: &wgpu::RenderPassColorAttachment) -> Self {
        Self {
            view: TextureViewAttachment::from_wgpu(&color_attachment.view).unwrap(),
            depth_slice: color_attachment.depth_slice,
            resolve_target: color_attachment
                .resolve_target
                .map(|texture| TextureViewAttachment::from_wgpu(texture).unwrap()),
            ops: color_attachment.ops,
        }
    }
}

#[derive(Debug)]
pub struct RenderPassCommand {
    label: Option<String>,
    color_attachments: Vec<Option<ColorAttachment>>,
    commands: Vec<RenderPassSubCommand>,
}

impl RenderPassCommand {
    pub fn execute(self) {
        // todo: sort them in some canonical order to avoid deadlocks due to
        // interleaving locks
        let texture_guards = self
            .color_attachments
            .iter()
            .map(|color_attachment| {
                color_attachment.as_ref().map(|color_attachment| {
                    let mut texture_guard = color_attachment.view.write();
                    match color_attachment.ops.load {
                        wgpu::LoadOp::Clear(clear_color) => {
                            texture_guard.clear(clear_color);
                        }
                        wgpu::LoadOp::Load => {
                            // nop
                        }
                        wgpu::LoadOp::DontCare(_) => {
                            // nop
                        }
                    }
                    texture_guard
                })
            })
            .collect::<Vec<_>>();

        for command in self.commands {
            match command {
                // todo
            }
        }
    }
}

#[derive(Debug)]
pub enum RenderPassSubCommand {
    // todo
}
