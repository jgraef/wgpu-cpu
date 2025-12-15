use naga::{
    Binding,
    BuiltIn,
    ShaderStage,
    Type,
};
use naga_interpreter::{
    EntryPointIndex,
    bindings::{
        BindingLocation,
        ShaderInput,
        ShaderOutput,
    },
    memory::WriteMemory,
};
use nalgebra::Vector4;

use crate::{
    buffer::{
        BufferSlice,
        OwnedBufferReadGuard,
    },
    pipeline::PipelineCompilationOptions,
    render_pass::invalid_binding,
    shader::ShaderModule,
};

#[derive(Debug)]
pub struct VertexState {
    pub module: ShaderModule,
    pub entry_point_name: Option<String>,
    pub entry_point_index: EntryPointIndex,
    pub compilation_options: PipelineCompilationOptions,
    pub vertex_buffer_layouts: Vec<VertexBufferLayout>,
    pub vertex_buffer_locations: Vec<Option<VertexBufferLocation>>,
}

impl VertexState {
    pub fn new(vertex: &wgpu::VertexState) -> Self {
        let module = vertex.module.as_custom::<ShaderModule>().unwrap().clone();

        let entry_point_index = module
            .as_ref()
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

        Self {
            module,
            entry_point_name: vertex.entry_point.map(ToOwned::to_owned),
            entry_point_index,
            compilation_options: PipelineCompilationOptions::new(&vertex.compilation_options),
            vertex_buffer_layouts,
            vertex_buffer_locations,
        }
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
pub struct VertexInput<'draw, 'pass> {
    pub vertex_index: u32,
    pub instance_index: u32,
    pub vertex_buffers: &'draw [VertexBufferInput<'pass>],
    pub vertex_locations: &'pass [Option<VertexBufferLocation>],
}

impl<'draw, 'pass> ShaderInput for VertexInput<'draw, 'pass> {
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

pub struct VertexBufferInput<'pass> {
    buffer_guard: &'pass OwnedBufferReadGuard,
    stride_instance: usize,
    stride_vertex: usize,
}

impl<'pass> VertexBufferInput<'pass> {
    pub fn new(
        buffer_guard: &'pass OwnedBufferReadGuard,
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
    pub position: Vector4<f32>,
    pub user_defined: User,
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
                        self.position = *bytemuck::from_bytes::<Vector4<f32>>(&source);
                    }
                    _ => invalid_binding(binding),
                }
            }
            Binding::Location { location, .. } => {
                self.user_defined
                    .write(BindingLocation::from(*location))
                    .copy_from_slice(source);
            }
        };
        target
    }
}

#[derive(Debug)]
pub struct IndexBufferBinding {
    pub buffer_slice: BufferSlice,
    pub index_format: wgpu::IndexFormat,
}
