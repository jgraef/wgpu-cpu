use naga::{
    Binding,
    BuiltIn,
    ShaderStage,
    Type,
};
use naga_cranelift::{
    CompiledModule,
    bindings::{
        BindingLocation,
        ShaderInput,
        ShaderOutput,
    },
    product::{
        EntryPointError,
        EntryPointIndex,
    },
    runtime::DefaultRuntimeError,
};

use crate::{
    buffer::{
        BufferReadGuard,
        BufferSlice,
    },
    pipeline::RenderPipeline,
    render_pass::{
        clipper::ClipPosition,
        invalid_binding,
        state::RenderPipelineState,
    },
    shader::{
        Error,
        ShaderModule,
        UserDefinedInterStagePoolBuffer,
        memory::WriteMemory,
    },
};

#[derive(Debug)]
pub struct VertexState {
    pub module: CompiledModule,
    pub entry_point: EntryPointIndex,
    pub vertex_buffer_layouts: Vec<VertexBufferLayout>,
    pub vertex_buffer_locations: Vec<Option<VertexBufferLocation>>,
}

impl VertexState {
    pub fn new(vertex: &wgpu::VertexState) -> Result<Self, Error> {
        let module = vertex.module.as_custom::<ShaderModule>().unwrap().clone();

        let module = module.for_pipeline(&vertex.compilation_options)?;

        let entry_point = module
            .find_entry_point(vertex.entry_point.as_deref(), ShaderStage::Vertex)
            .unwrap();

        let mut vertex_buffer_locations = vec![];
        let vertex_buffer_layouts = vertex
            .buffers
            .iter()
            .enumerate()
            .map(|(buffer_index, buffer)| {
                for (attribute_index, attribute) in buffer.attributes.iter().enumerate() {
                    let location_index = attribute.shader_location as usize;
                    if location_index >= vertex_buffer_locations.len() {
                        vertex_buffer_locations.resize_with(location_index + 1, || None);
                    }

                    if vertex_buffer_locations[location_index].is_some() {
                        panic!("Duplicate vertex attribute at location {location_index}");
                    }

                    vertex_buffer_locations[location_index] = Some(VertexBufferLocation {
                        buffer_index,
                        attribute_offset: attribute.offset as usize,
                        attribute_size: attribute.format.size() as usize,
                    });
                }

                VertexBufferLayout::new(buffer)
            })
            .collect();

        Ok(Self {
            module,
            entry_point,
            vertex_buffer_layouts,
            vertex_buffer_locations,
        })
    }
}

#[derive(Debug)]
pub struct VertexBufferLayout {
    pub array_stride: usize,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn new(buffer: &wgpu::VertexBufferLayout) -> Self {
        Self {
            array_stride: buffer.array_stride as usize,
            step_mode: buffer.step_mode,
            attributes: buffer.attributes.to_vec(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexBufferLocation {
    pub buffer_index: usize,
    pub attribute_offset: usize,
    pub attribute_size: usize,
}

#[derive(Debug)]
pub struct VertexInput<'draw, 'state> {
    pub vertex_index: u32,
    pub instance_index: u32,
    pub vertex_buffers: &'draw [VertexBufferInput<'state>],
    pub vertex_locations: &'state [Option<VertexBufferLocation>],
}

impl<'draw, 'state> ShaderInput for VertexInput<'draw, 'state> {
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        let source = match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::VertexIndex => bytemuck::bytes_of(&self.vertex_index),
                    BuiltIn::InstanceIndex => bytemuck::bytes_of(&self.instance_index),
                    _ => invalid_binding(binding),
                }
            }
            Binding::Location {
                location,
                per_primitive,
                ..
            } => {
                if let Some(Some(buffer_location)) = self.vertex_locations.get(*location as usize) {
                    let buffer = &self.vertex_buffers[buffer_location.buffer_index];
                    &buffer.read(self.instance_index, self.vertex_index)
                        [buffer_location.attribute_offset..][..buffer_location.attribute_size]
                }
                else {
                    panic!("No vertex attribute at location {location}");
                }
            }
        };

        target.copy_from_slice(source);
    }
}

#[derive(Debug)]

pub struct VertexBufferInput<'state> {
    buffer_guard: BufferReadGuard<'state>,
    stride_instance: usize,
    stride_vertex: usize,
}

impl<'state> VertexBufferInput<'state> {
    pub fn new(
        buffer_guard: BufferReadGuard<'state>,
        array_stride: usize,
        step_mode: wgpu::VertexStepMode,
    ) -> Self {
        let mut stride_instance = 0;
        let mut stride_vertex = 0;

        match step_mode {
            wgpu::VertexStepMode::Vertex => {
                stride_vertex = array_stride;
            }
            wgpu::VertexStepMode::Instance => {
                stride_instance = array_stride;
            }
        }

        Self {
            buffer_guard,
            stride_instance,
            stride_vertex,
        }
    }

    pub fn read(&self, instance_index: u32, vertex_index: u32) -> &[u8] {
        let start = instance_index as usize * self.stride_instance
            + vertex_index as usize * self.stride_vertex;
        &self.buffer_guard[start..]
    }
}

#[derive(Clone, Debug, Default)]
pub struct VertexOutput<User> {
    pub clip_position: ClipPosition,
    pub inter_stage_variables: User,
}

impl<User> ShaderOutput for VertexOutput<User>
where
    User: WriteMemory<BindingLocation>,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        let target = match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::Position { invariant: _ } => {
                        self.clip_position = *bytemuck::from_bytes::<ClipPosition>(&source);
                    }
                    _ => invalid_binding(binding),
                }
            }
            Binding::Location { location, .. } => {
                self.inter_stage_variables
                    .write(BindingLocation::from(*location))
                    .copy_from_slice(source);
            }
        };
        target
    }
}

impl<User> AsRef<ClipPosition> for VertexOutput<User> {
    fn as_ref(&self) -> &ClipPosition {
        &self.clip_position
    }
}

impl<User> AsMut<ClipPosition> for VertexOutput<User> {
    fn as_mut(&mut self) -> &mut ClipPosition {
        &mut self.clip_position
    }
}

// todo: these can be made Clone if reacquire the lock in vertex_buffers (it's a
// read-only lock, so it can be shared). then we can either clone them, or just
// construct as many as we need to run this in parallel
#[derive(Debug)]
pub struct VertexProcessingState<'pipeline, 'state> {
    vertex_locations: &'pipeline [Option<VertexBufferLocation>],
    vertex_buffers: Vec<VertexBufferInput<'state>>,
    module: CompiledModule,
    entry_point: EntryPointIndex,
}

impl<'pipeline, 'state> VertexProcessingState<'pipeline, 'state> {
    pub fn new(
        pipeline: &'pipeline RenderPipeline,
        vertex_buffers: &'state [Option<BufferSlice>],
    ) -> Self {
        let vertex_state = &pipeline.descriptor.vertex;

        let vertex_buffers = vertex_state
            .vertex_buffer_layouts
            .iter()
            .enumerate()
            .map(|(buffer_index, layout)| {
                let Some(Some(buffer)) = vertex_buffers.get(buffer_index)
                else {
                    panic!("Buffer {buffer_index} not bound")
                };
                VertexBufferInput::new(buffer.read(), layout.array_stride, layout.step_mode)
            })
            .collect::<Vec<_>>();
        tracing::debug!(?vertex_buffers, vertex_locations = ?vertex_state.vertex_buffer_locations);

        Self {
            vertex_locations: &vertex_state.vertex_buffer_locations,
            vertex_buffers,
            module: vertex_state.module.clone(),
            entry_point: vertex_state.entry_point,
        }
    }

    pub fn process(
        &mut self,
        pipeline_state: &RenderPipelineState,
        instance_index: u32,
        vertex_index: u32,
    ) -> Result<VertexOutput<UserDefinedInterStagePoolBuffer>, VertexProcessingError> {
        // create vertex output. this allocates buffers for the user-defined vertex
        // output.
        let mut vertex_output = VertexOutput {
            clip_position: Default::default(),
            inter_stage_variables: pipeline_state.interstage_user_pool.allocate(),
        };

        // run vertex shaders
        let entry_point = self.module.entry_point(self.entry_point);
        entry_point.run(
            VertexInput {
                vertex_index,
                instance_index,
                vertex_buffers: &self.vertex_buffers,
                vertex_locations: &self.vertex_locations,
            },
            &mut vertex_output,
        )?;
        //tracing::debug!(?vertex_index, clip_position = ?vertex_output.clip_position);

        Ok(VertexOutput {
            clip_position: vertex_output.clip_position,
            inter_stage_variables: vertex_output.inter_stage_variables.read_only(),
        })
    }
}

#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub struct VertexProcessingError {
    #[from]
    pub inner: EntryPointError<DefaultRuntimeError>,
}
