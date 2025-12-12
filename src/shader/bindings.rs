use naga::{
    Binding,
    BuiltIn,
    Handle,
    Type,
    TypeInner,
    UniqueArena,
};
use nalgebra::{
    Point2,
    Vector4,
};

use crate::{
    render_pass::ColorAttachmentState,
    shader::{
        ShaderModuleInner,
        interpreter::{
            Stack,
            Variable,
        },
    },
};

pub trait ApplyBinding {
    fn apply(&mut self, builtin: &Binding, ty: Handle<Type>, variable: Variable);
}

#[derive(Debug)]
pub struct ApplyBindings<'a, B> {
    pub types: &'a UniqueArena<Type>,
    pub apply_binding: B,
}

impl<'a, B> ApplyBindings<'a, B>
where
    B: ApplyBinding,
{
    pub fn apply(&mut self, binding: Option<&Binding>, ty: Handle<Type>, variable: Variable) {
        if let Some(binding) = binding {
            self.apply_binding.apply(binding, ty, variable);
        }
        else {
            let argument_ty = &self.types[ty];
            match &argument_ty.inner {
                TypeInner::Struct { members, span } => {
                    for member in members {
                        let variable = variable.member(member);
                        self.apply(member.binding.as_ref(), member.ty, variable);
                    }
                }
                _ => panic!("Invalid binding type: {:?}", argument_ty.inner),
            }
        }
    }
}

pub trait ShaderInput {
    fn write_binding(&self, binding: &Binding, variable: Variable, stack: &mut Stack);
}

#[derive(Debug)]
pub struct ApplyInput<'a, I> {
    pub stack: &'a mut Stack,
    pub input: &'a I,
}

impl<'a, I> ApplyBinding for ApplyInput<'a, I>
where
    I: ShaderInput,
{
    fn apply(&mut self, binding: &Binding, ty: Handle<Type>, variable: Variable) {
        self.input.write_binding(binding, variable, self.stack);
    }
}

pub trait ShaderOutput {
    fn read_binding(&mut self, binding: &Binding, variable: Variable, stack: &Stack);
}

#[derive(Debug)]
pub struct ApplyOutput<'a, O> {
    pub stack: &'a Stack,
    pub output: &'a mut O,
}

impl<'a, O> ApplyBinding for ApplyOutput<'a, O>
where
    O: ShaderOutput,
{
    fn apply(&mut self, binding: &Binding, ty: Handle<Type>, variable: Variable) {
        self.output.read_binding(binding, variable, self.stack);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrintBindingOutputs<'a> {
    pub module: &'a ShaderModuleInner,
    pub stack: &'a Stack,
}

impl<'a> ApplyBinding for PrintBindingOutputs<'a> {
    fn apply(&mut self, binding: &Binding, ty: Handle<Type>, variable: Variable) {
        tracing::debug!(?binding, ?ty, value = ?variable.debug(self.module, self.stack));
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexInput {
    pub vertex_index: u32,
    pub instance_index: u32,
    // todo: data from vertex buffers
}

impl ShaderInput for VertexInput {
    fn write_binding(&self, binding: &Binding, variable: Variable, stack: &mut Stack) {
        match binding {
            Binding::BuiltIn(builtin) => {
                macro_rules! builtin_inputs {
                    ($($variant:ident => $field:ident;)*) => {
                        match builtin {
                            $(
                                BuiltIn::$variant => {
                                    variable.write(stack, &self.$field);
                                }
                            )*
                            _ => {
                                tracing::warn!("Builtin input binding {builtin:?} not implemented");
                            }
                        }
                    };
                }

                builtin_inputs!(
                    VertexIndex => vertex_index;
                    InstanceIndex => instance_index;
                );
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => todo!(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct VertexOutput {
    pub position: Vector4<f32>,
}

impl ShaderOutput for VertexOutput {
    fn read_binding(&mut self, binding: &Binding, variable: Variable, stack: &Stack) {
        match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::Position { invariant } => {
                        self.position = *variable.read(stack);
                    }
                    _ => {
                        tracing::warn!("Builtin output binding {builtin:?} not implemented");
                    }
                }
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => todo!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput {
    pub position: Vector4<f32>,
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,
}

impl ShaderInput for FragmentInput {
    fn write_binding(&self, binding: &Binding, variable: Variable, stack: &mut Stack) {
        match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::Position { invariant } => variable.write(stack, &self.position),
                    BuiltIn::FrontFacing => variable.write(stack, &(self.front_facing as u32)),
                    BuiltIn::PrimitiveIndex => variable.write(stack, &self.primitive_index),
                    BuiltIn::SampleIndex => variable.write(stack, &self.sample_index),
                    BuiltIn::SampleMask => variable.write(stack, &self.sample_mask),
                    _ => panic!("Invalid fragment input binding: {binding:?}"),
                }
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct FragmentOutput<'a, 'color> {
    pub color_attachments: &'a mut [Option<ColorAttachmentState<'color>>],
    pub raster: Point2<u32>,
    pub t: f32,
}

impl<'a, 'color> ShaderOutput for FragmentOutput<'a, 'color> {
    fn read_binding(&mut self, binding: &Binding, variable: Variable, stack: &Stack) {
        match binding {
            Binding::BuiltIn(builtin) => todo!(),
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                let color_attachment = self.color_attachments[*location as usize].as_mut().unwrap();
                let color = *variable.read::<Vector4<f32>>(stack);
                tracing::debug!(?color, "color!");

                color_attachment.put_pixel(self.raster, color);
            }
        }
    }
}
