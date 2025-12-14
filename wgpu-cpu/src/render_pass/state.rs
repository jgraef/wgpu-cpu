use std::ops::Range;

use naga_interpreter::{
    Interpreter,
    bindings::UserDefinedIoLayout,
    memory::NullMemory,
};
use nalgebra::Vector2;

use crate::{
    pipeline::RenderPipeline,
    render_pass::{
        RenderPassDescriptor,
        clipper::Clipper,
        color_attachment::AcquiredColorAttachment,
        depth_stencil_attachment::AcquiredDepthStencilAttachment,
        primitive::Tri,
        rasterizer::Rasterizer,
    },
    shader::{
        UserDefinedIoBufferPool,
        fragment::{
            FragmentInput,
            FragmentOutput,
        },
        vertex::{
            VertexInput,
            VertexOutput,
        },
    },
};

#[derive(Debug)]
pub struct State<'pass> {
    pub color_attachments: Vec<Option<AcquiredColorAttachment<'pass>>>,
    pub depth_stencil_attachment: Option<AcquiredDepthStencilAttachment<'pass>>,
    pub clipper: Clipper,
    pub rasterizer: Rasterizer,
    pub pipeline: Option<RenderPipeline>,
}

impl<'pass> State<'pass> {
    pub fn new(descriptor: &'pass RenderPassDescriptor) -> Self {
        // todo: sort them in some canonical order to avoid deadlocks due to
        // interleaving locks
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
            color_attachments,
            depth_stencil_attachment,
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

        if let Some(depth_stencil_attachment) = &mut self.depth_stencil_attachment {
            depth_stencil_attachment.load();
        }
    }

    pub fn store(&mut self) {
        for color_attachment in &mut self.color_attachments {
            if let Some(color_attachment) = color_attachment {
                color_attachment.store();
            }
        }

        if let Some(depth_stencil_attachment) = &mut self.depth_stencil_attachment {
            depth_stencil_attachment.store();
        }
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        if let Some(pipeline) = &self.pipeline {
            assert!(
                pipeline.descriptor.primitive.topology == wgpu::PrimitiveTopology::TriangleList
            );

            let vertex_state = &pipeline.descriptor.vertex;
            let mut vertex_vm = Interpreter::new(
                vertex_state.module.clone(),
                NullMemory,
                vertex_state.entry_point_index,
            );

            let vertex_output_layout = match &vertex_state
                .module
                .as_ref()
                .user_defined_io_layout(vertex_state.entry_point_index)
            {
                UserDefinedIoLayout::Vertex { output } => output,
                _ => panic!("user defined io layouts for entry point are not for vertex stage"),
            };

            let vertex_output_pool = UserDefinedIoBufferPool::new(vertex_output_layout.clone());

            let mut fragment_state = pipeline.descriptor.fragment.as_ref().map(|fragment_state| {
                let vm = Interpreter::new(
                    fragment_state.module.clone(),
                    NullMemory,
                    fragment_state.entry_point_index,
                );
                (fragment_state, vm)
            });

            const PRIMITIVE_SIZE: usize = 3;
            let last_possible_primitive_start = vertices.end - PRIMITIVE_SIZE as u32;

            for instance_index in instances {
                let mut vertex_index = vertices.start;
                let mut primitive_index = 0;

                while vertex_index <= last_possible_primitive_start {
                    // create vertex outputs. this allocates buffers for the user-defined vertex
                    // outputs.
                    let mut vertex_outputs = std::array::from_fn::<_, PRIMITIVE_SIZE, _>(|_| {
                        VertexOutput {
                            position: Default::default(),
                            user_defined: vertex_output_pool.allocate(),
                        }
                    });

                    // run vertex shaders
                    for i in 0..PRIMITIVE_SIZE {
                        vertex_vm.run_entry_point(
                            &VertexInput {
                                vertex_index,
                                instance_index,
                                user_defined: NullMemory,
                            },
                            &mut vertex_outputs[i],
                        );

                        vertex_index += 1;
                    }

                    let (tri, vertex_outputs) = {
                        let tri = Tri(vertex_outputs
                            .each_ref()
                            .map(|output| output.position.into()));
                        tracing::debug!(?tri, "triangle!");

                        let vertex_outputs = vertex_outputs
                            .map(|vertex_output| vertex_output.user_defined.read_only());
                        (tri, vertex_outputs)
                    };

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
                                    depth_stencil_attachment: self
                                        .depth_stencil_attachment
                                        .as_mut(),
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

                    primitive_index += 1;
                }
            }
        }
    }
}
