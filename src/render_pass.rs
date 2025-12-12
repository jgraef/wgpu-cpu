use std::ops::Range;

use itertools::{
    Either,
    Itertools,
};
use nalgebra::{
    Matrix2x4,
    Point2,
    Vector2,
    Vector3,
    Vector4,
};

use crate::{
    command::{
        Command,
        CommandEncoder,
    },
    pipeline::RenderPipeline,
    shader::{
        bindings::{
            FragmentInput,
            FragmentOutput,
            VertexInput,
            VertexOutput,
        },
        interpreter::VirtualMachine,
        memory::NullMemory,
    },
    texture::{
        TextureInfo,
        TextureViewAttachment,
        TextureWriteGuard,
    },
    util::{
        bresenham::bresenham,
        lerp,
        scanline::scanlines,
        trilinear_interpolation,
    },
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
        let pipeline = pipeline.as_custom::<RenderPipeline>().unwrap();
        self.commands.push(RenderPassSubCommand::SetPipeline {
            pipeline: pipeline.clone(),
        })
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
        self.commands.push(RenderPassSubCommand::Draw {
            vertices: vertices.clone(),
            instances: instances.clone(),
        })
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
        let mut state = State::new(&self.color_attachments);
        state.load();

        for command in self.commands {
            match command {
                RenderPassSubCommand::SetPipeline { pipeline } => {
                    state.pipeline = Some(pipeline);
                }
                RenderPassSubCommand::Draw {
                    vertices,
                    instances,
                } => {
                    state.draw(vertices, instances);
                }
            }
        }

        state.store();
    }
}

#[derive(derive_more::Debug)]
pub enum RenderPassSubCommand {
    SetPipeline {
        #[debug(skip)]
        pipeline: RenderPipeline,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
}

#[derive(Debug)]
pub struct ColorAttachmentState<'a> {
    texture_guard: TextureWriteGuard<'a>,
    texture_info: &'a TextureInfo,
    ops: wgpu::Operations<wgpu::Color>,
}

impl<'a> ColorAttachmentState<'a> {
    pub fn new(color_attachment: &'a ColorAttachment) -> Self {
        let texture_guard = color_attachment.view.write();
        let texture_info = &color_attachment.view.info;

        Self {
            texture_guard,
            texture_info,
            ops: color_attachment.ops,
        }
    }

    pub fn load(&mut self) {
        match self.ops.load {
            wgpu::LoadOp::Clear(clear_color) => {
                self.texture_guard.clear(clear_color);
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

    pub fn put_pixel(&mut self, raster: Point2<u32>, color: Vector4<f32>) {
        self.texture_guard.put_pixel(raster, color);
    }
}

#[derive(Debug)]
struct State<'color> {
    color_attachments: Vec<Option<ColorAttachmentState<'color>>>,
    clipper: Clipper,
    rasterizer: Rasterizer,
    pipeline: Option<RenderPipeline>,
}

impl<'color> State<'color> {
    pub fn new(color_attachments: &'color [Option<ColorAttachment>]) -> Self {
        // todo: sort them in some canonical order to avoid deadlocks due to
        // interleaving locks
        let mut target_size = None;

        let color_attachments = color_attachments
            .iter()
            .map(|color_attachment| {
                color_attachment.as_ref().map(|color_attachment| {
                    let size = Vector2::new(
                        color_attachment.view.info.size.width,
                        color_attachment.view.info.size.height,
                    );
                    if let Some(target_size) = target_size {
                        assert_eq!(
                            target_size, size,
                            "All render attachments must be the same size"
                        );
                    }
                    else {
                        target_size = Some(size);
                    }

                    ColorAttachmentState::new(color_attachment)
                })
            })
            .collect::<Vec<_>>();

        let clipper = Clipper {};
        let rasterizer = Rasterizer::new(target_size.unwrap_or_default(), None);

        Self {
            color_attachments,
            clipper,
            rasterizer,
            pipeline: None,
        }
    }

    pub fn load(&mut self) {
        for color_attachment in &mut self.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.load();
            }
        }
    }

    pub fn store(&mut self) {
        for color_attachment in &mut self.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.store();
            }
        }
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if let Some(pipeline) = &self.pipeline {
            assert!(
                pipeline.descriptor.primitive.topology == wgpu::PrimitiveTopology::TriangleList
            );

            let vertex_state = &pipeline.descriptor.vertex;
            let mut vertex_vm = VirtualMachine::new(vertex_state.module.clone(), NullMemory);

            let mut fragment_state = pipeline.descriptor.fragment.as_ref().map(|fragment_state| {
                let vm = VirtualMachine::new(fragment_state.module.clone(), NullMemory);
                (fragment_state, vm)
            });

            for instance_index in instances {
                for (primitive_index, primitive) in
                    vertices.clone().chunks(3).into_iter().enumerate()
                {
                    let mut outputs = [VertexOutput::default(); 3];

                    for (i, vertex_index) in primitive.enumerate() {
                        vertex_vm.run_entry_point(
                            vertex_state.entry_point_index,
                            &VertexInput {
                                vertex_index,
                                instance_index,
                            },
                            &mut outputs[i],
                        );
                    }

                    let tri = Tri(outputs.map(|output| output.position));
                    tracing::debug!(?tri, "triangle!");

                    if let Some((fragment_state, fragment_vm)) = &mut fragment_state {
                        for tri in self.clipper.clip(tri) {
                            let face = tri.front_face(pipeline.descriptor.primitive.front_face);

                            if let Some(cull_face) = pipeline.descriptor.primitive.cull_mode
                                && face == cull_face
                            {
                                tracing::debug!(?tri, "culled");
                                continue;
                            }

                            for fragment in self.rasterizer.tri_fill(tri) {
                                fragment_vm.run_entry_point(
                                    fragment_state.entry_point_index,
                                    &FragmentInput {
                                        position: fragment.position,
                                        front_facing: face == wgpu::Face::Front,
                                        primitive_index: primitive_index as u32,
                                        sample_index: 0,
                                        sample_mask: !0,
                                    },
                                    &mut FragmentOutput {
                                        color_attachments: &mut *self.color_attachments,
                                        raster: fragment.raster,
                                    },
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
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

#[derive(Clone, Copy, Debug)]
pub struct Line(pub [Vector4<f32>; 2]);

#[derive(Debug)]
pub struct Clipper {
    //
}

impl Clipper {
    fn clip(&self, tri: Tri) -> impl Iterator<Item = Tri> {
        let clips = tri.0.map(|v| {
            !(v.x >= -v.w && v.x <= v.w && v.y >= -v.w && v.y <= v.w && v.z >= -v.w && v.z <= v.w)
        });

        let any = clips.iter().any(|x| *x);
        //let all = clips.iter().all(|x| *x);

        if any {
            tracing::debug!(?tri, "clipped");
            // todo only discard if all clip. otherwise we need to split it
            Either::Left(std::iter::empty())
        }
        else {
            Either::Right(std::iter::once(tri))
        }
    }
}

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

    pub fn tri_fill(&self, tri: Tri) -> impl Iterator<Item = Fragment<Vector3<f32>>> {
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

                    Vector3::new(
                        shoelace([raster_f32, tri_raster_f32[1], tri_raster_f32[2]]),
                        shoelace([raster_f32, tri_raster_f32[2], tri_raster_f32[0]]),
                        shoelace([raster_f32, tri_raster_f32[0], tri_raster_f32[1]]),
                    ) / total_area
                };

                scanlines(tri_raster_u32).flat_map(move |scanline| {
                    scanline.into_iter().map(move |raster| {
                        let barycentric = barycentric(raster);
                        let position = trilinear_interpolation(barycentric, tri.0);

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
