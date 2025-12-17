use std::{
    fmt::Debug,
    ops::Range,
};

use naga_interpreter::{
    Interpreter,
    bindings::UserDefinedIoLayout,
    memory::NullMemory,
};
use nalgebra::{
    Point2,
    Vector2,
};
use wgpu::Extent3d;

use crate::{
    buffer::BufferSlice,
    pipeline::RenderPipeline,
    render_pass::{
        RenderPassDescriptor,
        clipper::{
            Clip,
            ClipPosition,
            Clipped,
            NoClipper,
        },
        fragment::{
            AcquiredColorAttachment,
            AcquiredDepthStencilAttachment,
            FragmentInput,
            FragmentOutput,
        },
        index::{
            DirectIndices,
            IndexBufferBinding,
            IndexResolution,
        },
        primitive::{
            AsFrontFace,
            AssemblePrimitives,
            Primitive,
            PrimitiveList,
        },
        raster::{
            Rasterize,
            TriFillRasterizer,
        },
        vertex::VertexProcessingState,
    },
    shader::{
        UserDefinedInterStagePoolBuffer,
        UserDefinedIoBufferPool,
    },
    util::interpolation::Interpolate,
};

#[derive(Debug)]
pub struct State<'pass> {
    pub index_buffer_binding: Option<IndexBufferBinding>,
    pub pipeline_state: Option<RenderPipelineState>,
    pub render_state: RenderState<'pass>,

    // todo: should we lock the vertex buffers when we get them?
    pub vertex_buffers: Vec<Option<BufferSlice>>,
}

impl<'pass> State<'pass> {
    pub fn new(descriptor: &'pass RenderPassDescriptor) -> Self {
        // todo: order the lock acquisition in some canonical order to avoid dead locks.
        // a deadlock can happen if render pass A acquires color attachments [1, 2]
        // while render pass B acquires color attachments [2, 1]. If they both lock
        // their first color attachment at the same time they can't lock the second
        // color attachment. this generally applies to all resource acquisitions.

        let mut framebuffer_size = None;
        let mut figure_out_framebuffer_size = |attachment_size: Extent3d| {
            let size = Vector2::new(attachment_size.width, attachment_size.height);
            if let Some(framebuffer_size) = framebuffer_size {
                assert_eq!(
                    framebuffer_size, size,
                    "All render attachments must be the same size"
                );
            }
            else {
                framebuffer_size = Some(size);
            }
        };

        let color_attachments = descriptor
            .color_attachments
            .iter()
            .map(|color_attachment| {
                color_attachment.as_ref().map(|color_attachment| {
                    figure_out_framebuffer_size(color_attachment.view.info.size);
                    AcquiredColorAttachment::new(color_attachment)
                })
            })
            .collect::<Vec<_>>();

        let depth_stencil_attachment =
            descriptor
                .depth_stencil_attachment
                .as_ref()
                .map(|depth_stencil_attachment| {
                    figure_out_framebuffer_size(depth_stencil_attachment.view.info.size);
                    AcquiredDepthStencilAttachment::new(depth_stencil_attachment)
                });

        let framebuffer_size = framebuffer_size.unwrap_or_default();

        Self {
            index_buffer_binding: None,
            pipeline_state: None,
            render_state: RenderState {
                occlusion_query_index: 0,
                viewport: Viewport::new(framebuffer_size),
                scissor_rect: ScissorRect::new(framebuffer_size),
                blend_constant: Default::default(), // what does this default to?
                stencil_reference: 0,
                color_attachments,
                depth_stencil_attachment,
            },
            vertex_buffers: vec![],
        }
    }

    pub fn load(&mut self) {
        for color_attachment in &mut self.render_state.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.load();
            }
        }

        if let Some(depth_stencil_attachment) = &mut self.render_state.depth_stencil_attachment {
            depth_stencil_attachment.load();
        }
    }

    pub fn store(&mut self) {
        for color_attachment in &mut self.render_state.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.store();
            }
        }

        if let Some(depth_stencil_attachment) = &mut self.render_state.depth_stencil_attachment {
            depth_stencil_attachment.store();
        }
    }

    pub fn set_pipeline(&mut self, pipeline: RenderPipeline) {
        let vertex_state = &pipeline.descriptor.vertex;

        // todo: we should probably just store one layout for inter-stage variables and
        // verify at pipeline creation that vertex/fragment layouts match
        let vertex_output_layout = match &vertex_state
            .module
            .as_ref()
            .user_defined_io_layout(vertex_state.entry_point_index)
        {
            UserDefinedIoLayout::Vertex { output } => output,
            _ => panic!("user defined io layouts for entry point are not for vertex stage"),
        };

        let interstage_user_pool = UserDefinedIoBufferPool::new(vertex_output_layout.clone());

        self.pipeline_state = Some(RenderPipelineState {
            pipeline,
            interstage_user_pool,
        });
    }

    pub fn set_index_buffer(&mut self, buffer_slice: BufferSlice, index_format: wgpu::IndexFormat) {
        self.index_buffer_binding = Some(IndexBufferBinding {
            buffer_slice,
            index_format,
        });
    }

    pub fn set_vertex_buffer(&mut self, buffer_slice: BufferSlice, slot: u32) {
        let index = slot as usize;
        if index >= self.vertex_buffers.len() {
            self.vertex_buffers.resize_with(index + 1, || None);
        }

        self.vertex_buffers[index] = Some(buffer_slice);
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        let Some(pipeline_state) = &self.pipeline_state
        else {
            // should we panic?
            panic!("No pipeline bound");
        };

        let vertex_processing_state =
            VertexProcessingState::new(&pipeline_state.pipeline, &self.vertex_buffers);

        self.render_state.draw_with_pipeline_rasterizer(
            &pipeline_state,
            instances,
            vertices,
            DirectIndices,
            vertex_processing_state,
        );
    }

    pub fn set_blend_constant(&mut self, color: wgpu::Color) {
        self.render_state.blend_constant = color;
    }

    pub fn set_scissor_rect(&mut self, scissor_rect: ScissorRect) {
        self.render_state.scissor_rect = scissor_rect;
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.render_state.viewport = viewport;
    }

    pub fn set_stencil_reference(&mut self, stencil_reference: StencilValue) {
        self.render_state.stencil_reference = stencil_reference;
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        let Some(pipeline_state) = &self.pipeline_state
        else {
            // should we panic?
            panic!("No pipeline bound");
        };

        let Some(index_buffer_binding) = &self.index_buffer_binding
        else {
            panic!("No index buffer bound");
        };

        let vertex_processing_state =
            VertexProcessingState::new(&pipeline_state.pipeline, &self.vertex_buffers);

        macro_rules! dispatch_index_format {
            ($($variant:ident => $ty:ty,)*) => {
                match index_buffer_binding.index_format {
                    $(
                        wgpu::IndexFormat::$variant => {
                            self.render_state.draw_with_pipeline_rasterizer(
                                &pipeline_state,
                                instances,
                                indices,
                                index_buffer_binding.begin::<$ty>(base_vertex),
                                vertex_processing_state
                            );
                        }
                    )*
                }
            };
        }

        dispatch_index_format!(
            Uint16 => u16,
            Uint32 => u32,
        );
    }
}

#[derive(Debug)]
pub struct RenderPipelineState {
    pub pipeline: RenderPipeline,
    pub interstage_user_pool: UserDefinedIoBufferPool,
}

type VertexOutput = crate::render_pass::vertex::VertexOutput<UserDefinedInterStagePoolBuffer>;

/// https://gpuweb.github.io/gpuweb/#renderstate
#[derive(Debug)]
pub struct RenderState<'pass> {
    pub occlusion_query_index: u32,
    pub viewport: Viewport,
    pub scissor_rect: ScissorRect,
    pub blend_constant: wgpu::Color,
    pub stencil_reference: StencilValue,
    pub color_attachments: Vec<Option<AcquiredColorAttachment<'pass>>>,
    pub depth_stencil_attachment: Option<AcquiredDepthStencilAttachment<'pass>>,
}

impl<'pass> RenderState<'pass> {
    fn draw_with_pipeline_rasterizer(
        &mut self,
        pipeline_state: &RenderPipelineState,
        instances: Range<u32>,
        indices: Range<u32>,
        index_resolution: impl IndexResolution,
        vertex_processing: VertexProcessingState,
    ) {
        let primitive_state = &pipeline_state.pipeline.descriptor.primitive;
        assert_eq!(
            primitive_state.topology,
            wgpu::PrimitiveTopology::TriangleList
        );
        assert_eq!(primitive_state.polygon_mode, wgpu::PolygonMode::Fill);

        // todo
        let framebuffer_size = self.scissor_rect.size;

        let primitive_assembly = PrimitiveList;

        let clipper = NoClipper;

        let rasterizer = TriFillRasterizer::new(framebuffer_size);

        self.draw(
            pipeline_state,
            instances,
            indices,
            index_resolution,
            vertex_processing,
            primitive_assembly,
            clipper,
            rasterizer,
        );

        /*
        macro_rules! draw {
            ($primitive:ty, $rasterizer:path, $n:expr) => {
                self.draw::<$n, $primitive>(
                    pipeline_state,
                    instances,
                    draw_mode,
                    <$rasterizer>::new(framebuffer_size),
                );
            };
        }

        match (primitive_state.topology, primitive_state.polygon_mode) {
            (wgpu::PrimitiveTopology::PointList, _) => {
                draw!(Point, PointRasterizer, 1);
            }
            (wgpu::PrimitiveTopology::LineList, wgpu::PolygonMode::Point) => {
                draw!(Line, LinePointRasterizer, 1);
            }
            (wgpu::PrimitiveTopology::LineList, wgpu::PolygonMode::Line) => {
                draw!(Line, LineRasterizer, 2);
            }
            (wgpu::PrimitiveTopology::LineList, wgpu::PolygonMode::Fill) => {
                draw!(Line, LineRasterizer, 2);
            }
            (wgpu::PrimitiveTopology::TriangleList, wgpu::PolygonMode::Point) => {
                draw!(Tri, TriPointRasterizer, 3);
            }
            (wgpu::PrimitiveTopology::TriangleList, wgpu::PolygonMode::Line) => {
                draw!(Tri, TriLineRasterizer, 3);
            }
            (wgpu::PrimitiveTopology::TriangleList, wgpu::PolygonMode::Fill) => {
                draw!(Tri, TriFillRasterizer, 3);
            }
            (wgpu::PrimitiveTopology::LineStrip, _)
            | (wgpu::PrimitiveTopology::TriangleStrip, _) => {
                todo!("drawing of {:?} not implemented", primitive_state.topology);
            }
        } */
    }

    fn draw<const PRIMITIVE_SIZE: usize, Index, Assembly, Rasterizer>(
        &mut self,
        pipeline_state: &RenderPipelineState,
        instances: Range<u32>,
        indices: Range<u32>,
        index_resolution: Index,
        mut vertex_processing: VertexProcessingState,
        mut primitive_assembly: Assembly,
        clipper: impl Clip<PRIMITIVE_SIZE>,
        rasterizer: Rasterizer,
    ) where
        Index: IndexResolution,
        Assembly: AssemblePrimitives<VertexOutput, PRIMITIVE_SIZE>,
        <Assembly as AssemblePrimitives<VertexOutput, PRIMITIVE_SIZE>>::Face: AsFrontFace,
        Rasterizer: Rasterize<Primitive<ClipPosition, PRIMITIVE_SIZE>>,
        Rasterizer::Interpolation: Interpolate<PRIMITIVE_SIZE>,
    {
        let pipeline = &pipeline_state.pipeline;

        let mut fragment_state = pipeline.descriptor.fragment.as_ref().map(|fragment_state| {
            let vm = Interpreter::new(
                fragment_state.module.clone(),
                NullMemory,
                fragment_state.entry_point_index,
            );
            (fragment_state, vm)
        });

        for instance_index in instances {
            // resolve indices
            let vertex_indices = indices.clone().map(|index| index_resolution.resolve(index));

            // process vertices
            let vertices = vertex_processing
                .process(pipeline_state, instance_index, vertex_indices)
                .into_iter();

            // assemble primitives
            let primitives = primitive_assembly.assemble(vertices).into_iter();

            if let Some((fragment_state, fragment_vm)) = &mut fragment_state {
                for (primitive_index, primitive) in primitives.enumerate() {
                    for primitive in clipper.clip(primitive) {
                        let (front_facing, cull_face) =
                            primitive
                                .try_front_face()
                                .map_or((true, false), |front_face| {
                                    let front_facing =
                                        front_face == pipeline.descriptor.primitive.front_face;

                                    let cull_face = match pipeline.descriptor.primitive.cull_mode {
                                        Some(wgpu::Face::Front) => front_facing,
                                        Some(wgpu::Face::Back) => !front_facing,
                                        None => false,
                                    };

                                    (front_facing, cull_face)
                                });

                        if cull_face {
                            tracing::trace!(primitive = ?primitive.clip_positions(), "culled");
                            continue;
                        }

                        //let front_facing = rasterization_point.front_face;

                        for rasterization_point in
                            rasterizer.rasterize(Primitive::new(primitive.clip_positions(), ()))
                        {
                            let sample_index = rasterization_point
                                .destination
                                .sample_index
                                .map_or(0, |sample_index| sample_index.get().into());
                            let sample_mask = rasterization_point.coverage_mask;

                            let input = FragmentInput {
                                //position: fragment.position,
                                position: Default::default(), // todo
                                front_facing,
                                primitive_index: primitive_index as u32,
                                sample_index,
                                sample_mask,
                                interpolation_coefficients: rasterization_point.interpolation,
                                inter_stage_variables: primitive
                                    .each_vertex_ref::<Clipped<VertexOutput>>()
                                    .map(|vertex_output| {
                                        vertex_output.unclipped.inter_stage_variables.clone()
                                    }),
                            };

                            let mut output = FragmentOutput {
                                position: rasterization_point.destination.position,
                                frag_depth: rasterization_point.depth,
                                sample_mask,
                                color_attachments: &mut self.color_attachments,
                                depth_stencil_attachment: self.depth_stencil_attachment.as_mut(),
                                depth_stencil_state: pipeline.descriptor.depth_stencil.as_ref(),
                            };

                            // perform early depth test
                            if !fragment_vm.early_depth_test(|| output.depth_test()) {
                                // early depth test rejected
                                continue;
                            }

                            // run fragment shader
                            fragment_vm.run_entry_point(&input, &mut output);
                        }
                    }
                }
            }
            else {
                // no fragment state bound. just consume the iterator so that the vertex shaders
                // run
                primitives.for_each(|_| ());
            }
        }
    }
}

/// https://gpuweb.github.io/gpuweb/#renderstate
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub offset: Point2<f32>,
    pub size: Vector2<f32>,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl Viewport {
    pub fn new(framebuffer_size: Vector2<u32>) -> Self {
        Self {
            offset: Point2::origin(),
            size: framebuffer_size.cast(),
            min_depth: 0.0,
            max_depth: 1.0,
        }
    }
}

/// https://gpuweb.github.io/gpuweb/#renderstate
#[derive(Clone, Copy, Debug)]
pub struct ScissorRect {
    pub offset: Point2<u32>,
    pub size: Vector2<u32>,
}

impl ScissorRect {
    pub fn new(framebuffer_size: Vector2<u32>) -> Self {
        Self {
            offset: Point2::origin(),
            size: framebuffer_size,
        }
    }
}

pub type StencilValue = u32;
