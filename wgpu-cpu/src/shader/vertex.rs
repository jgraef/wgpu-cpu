use naga::{
    Binding,
    BuiltIn,
    Type,
};
use naga_interpreter::{
    bindings::{
        BindingLocation,
        ShaderInput,
        ShaderOutput,
    },
    memory::{
        ReadMemory,
        WriteMemory,
    },
};
use nalgebra::Vector4;

use crate::shader::invalid_binding;

#[derive(Clone, Copy, Debug)]
pub struct VertexInput<User> {
    pub vertex_index: u32,
    pub instance_index: u32,
    pub user_defined: User,
}

impl<User> ShaderInput for VertexInput<User>
where
    User: ReadMemory<BindingLocation>,
{
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
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => todo!(),
        };
        target[..source.len()].copy_from_slice(source);
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
                        bytemuck::bytes_of_mut(&mut self.position)
                    }
                    _ => invalid_binding(binding),
                }
            }
            Binding::Location { location, .. } => {
                self.user_defined.write(BindingLocation::from(*location))
            }
        };
        target.copy_from_slice(&source[..target.len()]);
    }
}
