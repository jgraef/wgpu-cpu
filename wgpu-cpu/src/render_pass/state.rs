use std::ops::Range;

use bytemuck::Pod;
use naga_interpreter::{
    Interpreter,
    bindings::UserDefinedIoLayout,
    memory::NullMemory,
};
use nalgebra::Vector2;

use crate::{
    buffer::{
        BufferReadGuard,
        BufferSlice,
        OwnedBufferReadGuard,
    },
    pipeline::RenderPipeline,
    render_pass::{
        RenderPassDescriptor,
        clipper::Clipper,
        fragment::{
            AcquiredColorAttachment,
            AcquiredDepthStencilAttachment,
            FragmentInput,
            FragmentOutput,
        },
        primitive::Tri,
        rasterizer::Rasterizer,
        vertex::{
            IndexBufferBinding,
            VertexBufferInput,
            VertexInput,
            VertexOutput,
        },
    },
    shader::UserDefinedIoBufferPool,
    util::IteratorExt,
};

#[derive(Debug)]
pub struct State<'pass> {
    pub index_buffer_binding: Option<IndexBufferBinding>,
    pub draw_state: DrawState<'pass>,
}

impl<'pass> State<'pass> {
    pub fn new(descriptor: &'pass RenderPassDescriptor) -> Self {
        // todo: order the lock acquisition in some canonical order to avoid dead locks.
        // a deadlock can happen if render pass A acquires color attachments [1, 2]
        // while render pass B acquires color attachments [2, 1]. If they both lock
        // their first color attachment at the same time they can't lock the second
        // color attachment. this generally applies to all resource acquisitions.

        let mut target_size = None;
        let color_attachments = descriptor
            .color_attachments
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

                    AcquiredColorAttachment::new(color_attachment)
                })
            })
            .collect::<Vec<_>>();

        let depth_stencil_attachment = descriptor
            .depth_stencil_attachment
            .as_ref()
            .map(AcquiredDepthStencilAttachment::new);

        let clipper = Clipper {};
        let rasterizer = Rasterizer::new(target_size.unwrap_or_default(), None);

        Self {
            index_buffer_binding: None,
            draw_state: DrawState {
                pipeline_state: None,
                clipper,
                rasterizer,
                vertex_buffer_bindings: vec![],
                color_attachments,
                depth_stencil_attachment,
            },
        }
    }

    pub fn load(&mut self) {
        for color_attachment in &mut self.draw_state.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.load();
            }
        }

        if let Some(depth_stencil_attachment) = &mut self.draw_state.depth_stencil_attachment {
            depth_stencil_attachment.load();
        }
    }

    pub fn store(&mut self) {
        for color_attachment in &mut self.draw_state.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.store();
            }
        }

        if let Some(depth_stencil_attachment) = &mut self.draw_state.depth_stencil_attachment {
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

        self.draw_state.pipeline_state = Some(RenderPipelineState {
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
        if index >= self.draw_state.vertex_buffer_bindings.len() {
            self.draw_state
                .vertex_buffer_bindings
                .resize_with(index + 1, || None);
        }

        // is this fine?
        self.draw_state.vertex_buffer_bindings[index] = Some(buffer_slice.read_owned());
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.draw_state.draw(instances, DrawDirect { vertices });
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        let Some(index_buffer_binding) = &self.index_buffer_binding
        else {
            panic!("No index buffer bound");
        };

        fn draw_indexed_inner<'pass, IndexFormat>(
            draw_state: &'pass mut DrawState,
            indices: Range<u32>,
            base_vertex: i32,
            instances: Range<u32>,
            index_buffer_guard: BufferReadGuard<'pass>,
        ) where
            IndexFormat: Pod,
            u32: From<IndexFormat>,
        {
            let index_buffer = bytemuck::cast_slice::<u8, IndexFormat>(&*index_buffer_guard);
            draw_state.draw(
                instances,
                DrawIndexed {
                    indices,
                    base_vertex,
                    index_buffer,
                },
            );
        }

        let index_buffer_guard = index_buffer_binding.buffer_slice.read();
        match index_buffer_binding.index_format {
            wgpu::IndexFormat::Uint16 => {
                draw_indexed_inner::<u16>(
                    &mut self.draw_state,
                    indices,
                    base_vertex,
                    instances,
                    index_buffer_guard,
                )
            }
            wgpu::IndexFormat::Uint32 => {
                draw_indexed_inner::<u32>(
                    &mut self.draw_state,
                    indices,
                    base_vertex,
                    instances,
                    index_buffer_guard,
                )
            }
        }
    }
}

#[derive(Debug)]
pub struct RenderPipelineState {
    pub pipeline: RenderPipeline,
    pub interstage_user_pool: UserDefinedIoBufferPool,
}

trait DrawMode {
    fn vertex_indices(&self) -> impl Iterator<Item = u32>;
}

#[derive(Clone, Debug)]
struct DrawDirect {
    vertices: Range<u32>,
}

impl DrawMode for DrawDirect {
    fn vertex_indices(&self) -> impl Iterator<Item = u32> {
        self.vertices.clone().into_iter()
    }
}

#[derive(Clone, Debug)]
struct DrawIndexed<'pass, IndexFormat> {
    indices: Range<u32>,
    base_vertex: i32,
    index_buffer: &'pass [IndexFormat],
}

impl<'pass, IndexFormat> DrawMode for DrawIndexed<'pass, IndexFormat>
where
    IndexFormat: Copy,
    u32: From<IndexFormat>,
{
    fn vertex_indices(&self) -> impl Iterator<Item = u32> {
        self.indices.clone().map(|index| {
            u32::from(self.index_buffer[index as usize]).strict_add_signed(self.base_vertex)
        })
    }
}

#[derive(Debug)]
struct DrawIndexedState {
    index_buffer_guard: OwnedBufferReadGuard,
}

#[derive(Debug)]
pub struct DrawState<'pass> {
    pub pipeline_state: Option<RenderPipelineState>,
    pub clipper: Clipper,
    pub rasterizer: Rasterizer,
    pub vertex_buffer_bindings: Vec<Option<OwnedBufferReadGuard>>,
    pub color_attachments: Vec<Option<AcquiredColorAttachment<'pass>>>,
    pub depth_stencil_attachment: Option<AcquiredDepthStencilAttachment<'pass>>,
}

impl<'pass> DrawState<'pass> {
    fn draw(&mut self, instances: Range<u32>, draw_mode: impl DrawMode) {
        let Some(pipeline_state) = &self.pipeline_state
        else {
            // should we panic?
            panic!("No pipeline bound");
        };
        let pipeline = &pipeline_state.pipeline;

        assert!(
            pipeline.descriptor.primitive.topology == wgpu::PrimitiveTopology::TriangleList,
            "todo: implement primitives other than triangles"
        );

        let vertex_state = &pipeline.descriptor.vertex;
        let mut vertex_vm = Interpreter::new(
            vertex_state.module.clone(),
            NullMemory,
            vertex_state.entry_point_index,
        );

        let vertex_buffers = vertex_state
            .vertex_buffer_layouts
            .iter()
            .enumerate()
            .map(|(buffer_index, layout)| {
                let Some(Some(buffer_guard)) = self.vertex_buffer_bindings.get(buffer_index)
                else {
                    panic!("Buffer {buffer_index} not bound")
                };
                VertexBufferInput::new(buffer_guard, layout.array_stride, layout.step_mode)
            })
            .collect::<Vec<_>>();
        tracing::debug!(?vertex_buffers, vertex_locations = ?vertex_state.vertex_buffer_locations);

        let mut fragment_state = pipeline.descriptor.fragment.as_ref().map(|fragment_state| {
            let vm = Interpreter::new(
                fragment_state.module.clone(),
                NullMemory,
                fragment_state.entry_point_index,
            );
            (fragment_state, vm)
        });

        const PRIMITIVE_SIZE: usize = 3;

        for instance_index in instances {
            let vertices = draw_mode.vertex_indices().map(|vertex_index| {
                // create vertex output. this allocates buffers for the user-defined vertex
                // output.
                let mut vertex_output = VertexOutput {
                    position: Default::default(),
                    user_defined: pipeline_state.interstage_user_pool.allocate(),
                };

                // run vertex shaders
                vertex_vm.run_entry_point(
                    VertexInput {
                        vertex_index,
                        instance_index,
                        vertex_buffers: &vertex_buffers,
                        vertex_locations: &vertex_state.vertex_buffer_locations,
                    },
                    &mut vertex_output,
                );

                vertex_output
            });

            if let Some((fragment_state, fragment_vm)) = &mut fragment_state {
                for (primitive_index, vertex_outputs) in
                    vertices.array_chunks_::<PRIMITIVE_SIZE>().enumerate()
                {
                    // convert vertex positions to tri and make user-defined interstage buffers
                    // readonly for sharing
                    let (tri, vertex_outputs) = {
                        let tri = Tri(vertex_outputs
                            .each_ref()
                            .map(|output| output.position.into()));

                        let vertex_outputs = vertex_outputs
                            .map(|vertex_output| vertex_output.user_defined.read_only());
                        (tri, vertex_outputs)
                    };
                    //tracing::debug!(?primitive_index, ?tri);

                    for tri in self.clipper.clip(tri) {
                        let face = tri.front_face(pipeline.descriptor.primitive.front_face);
                        tracing::trace!(?tri, ?face);

                        if let Some(cull_face) = pipeline.descriptor.primitive.cull_mode
                            && face == cull_face
                        {
                            tracing::trace!(?tri, "culled");
                            continue;
                        }

                        for fragment in self.rasterizer.tri_fill(tri) {
                            // setup inputs and outputs
                            let sample_index = 0;
                            let sample_mask = !0;

                            let input = FragmentInput {
                                position: fragment.position,
                                front_facing: face == wgpu::Face::Front,
                                primitive_index: primitive_index as u32,
                                sample_index,
                                sample_mask,
                                barycentric: fragment.barycentric,
                                vertex_outputs: vertex_outputs.clone(),
                            };

                            let mut output = FragmentOutput {
                                position: fragment.raster,
                                frag_depth: fragment.position.z,
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
                // no fragment state bound. just consume the vertex interator so that the vertex
                // shaders run
                vertices.for_each(|_| ());
            }
        }
    }
}
