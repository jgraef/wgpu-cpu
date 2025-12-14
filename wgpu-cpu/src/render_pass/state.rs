use std::ops::Range;

use naga_interpreter::{
    Interpreter,
    bindings::{
        FragmentInput,
        FragmentOutput,
        UserDefinedIoLayout,
        VertexInput,
        VertexOutput,
    },
    memory::NullMemory,
};
use nalgebra::{
    Point2,
    Vector2,
    Vector4,
};

use crate::{
    pipeline::RenderPipeline,
    render_pass::{
        ColorAttachment,
        clipper::Clipper,
        primitive::Tri,
        rasterizer::Rasterizer,
    },
    shader::UserDefinedIoBufferPool,
    texture::{
        TextureInfo,
        TextureWriteGuard,
    },
};

#[derive(Debug)]
pub struct State<'color> {
    pub color_attachments: Vec<Option<ColorAttachmentState<'color>>>,
    pub clipper: Clipper,
    pub rasterizer: Rasterizer,
    pub pipeline: Option<RenderPipeline>,
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
            let mut vertex_vm = Interpreter::new(vertex_state.module.clone(), NullMemory);

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
                let vm = Interpreter::new(fragment_state.module.clone(), NullMemory);
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
                            vertex_state.entry_point_index,
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
                                fragment_vm.run_entry_point(
                                    fragment_state.entry_point_index,
                                    &FragmentInput {
                                        position: fragment.position.into(),
                                        front_facing: face == wgpu::Face::Front,
                                        primitive_index: primitive_index as u32,
                                        sample_index: 0,
                                        sample_mask: !0,
                                        barycentric: fragment.barycentric,
                                        vertex_outputs: vertex_outputs.clone(),
                                    },
                                    &mut FragmentOutput {
                                        color_attachments: ColorAttachmentBinding {
                                            states: &mut *self.color_attachments,
                                        },
                                        raster: fragment.raster.into(),
                                    },
                                );
                            }
                        }
                    }

                    primitive_index += 1;
                }
            }
        }
    }
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
struct ColorAttachmentBinding<'a, 'color> {
    states: &'a mut [Option<ColorAttachmentState<'color>>],
}

impl<'a, 'color> naga_interpreter::bindings::ColorAttachments
    for ColorAttachmentBinding<'a, 'color>
{
    fn put_pixel(&mut self, location: u32, position: [u32; 2], color: [f32; 4]) {
        let color_attachment = self.states[location as usize].as_mut().unwrap();
        color_attachment.put_pixel(position.into(), color.into());
    }
}
