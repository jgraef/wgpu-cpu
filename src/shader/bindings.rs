use naga::{
    Binding,
    BuiltIn,
    FunctionArgument,
    FunctionResult,
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
        interpreter::Variable,
        memory::{
            ReadMemory,
            Slice,
            StackFrame,
            WriteMemory,
        },
    },
};

#[derive(Clone, Copy, Debug)]
pub struct BindingAddress {
    _todo: (),
}

#[derive(Clone, Copy, Debug)]
pub struct VertexInput {
    pub vertex_index: u32,
    pub instance_index: u32,
}

impl ReadMemory<Binding> for VertexInput {
    fn read(&self, address: Binding) -> &[u8] {
        match address {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::VertexIndex => bytemuck::bytes_of(&self.vertex_index),
                    BuiltIn::InstanceIndex => bytemuck::bytes_of(&self.instance_index),
                    _ => unsupported_binding(address),
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

#[derive(Clone, Copy, Debug, Default)]
pub struct VertexOutput {
    pub position: Vector4<f32>,
}

impl WriteMemory<Binding> for VertexOutput {
    fn write(&mut self, address: Binding, data: &[u8]) {
        match address {
            Binding::BuiltIn(builtin) => {
                let target = match builtin {
                    BuiltIn::Position { invariant: _ } => {
                        bytemuck::bytes_of_mut(&mut self.position)
                    }
                    _ => unsupported_binding(address),
                };

                target.copy_from_slice(&data[..target.len()]);
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

/*
#[derive(Clone, Copy, Debug)]
pub struct VertexBindings {
    pub input: VertexInput,
    pub output: VertexOutput,
}

impl ReadMemory<BindingAddress> for VertexBindings {
    fn read(&self, address: BindingAddress) -> &[u8] {
        match address {
            BindingAddress::InputBinding(binding) => self.input.read(binding),
            BindingAddress::OutputBinding(binding) => {
                panic!("Can't read from an output binding")
            }
        }
    }
}

impl WriteMemory<BindingAddress> for VertexBindings {
    fn write(&mut self, address: BindingAddress, data: &[u8]) {
        match address {
            BindingAddress::InputBinding(binding) => {
                panic!("Can't write to an input binidng");
            }
            BindingAddress::OutputBinding(binding) => {
                self.output.write(binding, data);
            }
        }
    }
}

impl ReadWriteMemory<BindingAddress> for VertexBindings {
    fn copy(&mut self, source: BindingAddress, target: BindingAddress) {
        todo!();
    }
} */

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput {
    pub position: Vector4<f32>,
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,
}

impl ReadMemory<Binding> for FragmentInput {
    fn read(&self, address: Binding) -> &[u8] {
        match address {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::Position { invariant } => bytemuck::bytes_of(&self.position),
                    BuiltIn::FrontFacing => bytes_of_bool_as_u32(self.front_facing),
                    BuiltIn::PrimitiveIndex => bytemuck::bytes_of(&self.primitive_index),
                    BuiltIn::SampleIndex => bytemuck::bytes_of(&self.sample_index),
                    BuiltIn::SampleMask => bytemuck::bytes_of(&self.sample_mask),
                    _ => unsupported_binding(address),
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
pub struct FragmentOutput<C> {
    pub color_attachments: C,
    pub raster: Point2<u32>,
}

impl<C> WriteMemory<Binding> for FragmentOutput<C>
where
    C: ColorAttachments,
{
    fn write(&mut self, address: Binding, data: &[u8]) {
        match address {
            Binding::BuiltIn(builtin) => {
                todo!()
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                let color: &Vector4<f32> = bytemuck::from_bytes(data);
                self.color_attachments
                    .put_pixel(location, self.raster, *color);
            }
        }
    }
}

/*
#[derive(Clone, Copy, Debug)]
pub struct FragmentBindings<C> {
    pub input: FragmentInput,
    pub output: FragmentOutput<C>,
}


impl<C> ReadMemory<BindingAddress> for FragmentBindings<C>
where
    C: ColorAttachments,
{
    fn read(&self, address: BindingAddress) -> &[u8] {
        match address {
            BindingAddress::InputBinding(binding) => self.input.read(binding),
            BindingAddress::OutputBinding(binding) => {
                panic!("Can't read from an output binding")
            }
        }
    }
}

impl<C> WriteMemory<BindingAddress> for FragmentBindings<C>
where
    C: ColorAttachments,
{
    fn write(&mut self, address: BindingAddress, data: &[u8]) {
        match address {
            BindingAddress::InputBinding(binding) => {
                panic!("Can't write to an input binidng");
            }
            BindingAddress::OutputBinding(binding) => {}
        }
    }
}

impl<C> ReadWriteMemory<BindingAddress> for FragmentBindings<C>
where
    C: ColorAttachments,
{
    fn copy(&mut self, source: BindingAddress, target: BindingAddress) {
        todo!();
    }
}
     */

pub trait ColorAttachments {
    fn put_pixel(&mut self, location: u32, position: Point2<u32>, color: Vector4<f32>);
}

impl<'a, 'color> ColorAttachments for &'a mut [Option<ColorAttachmentState<'color>>] {
    fn put_pixel(&mut self, location: u32, position: Point2<u32>, color: Vector4<f32>) {
        let color_attachment = self[location as usize].as_mut().unwrap();
        color_attachment.put_pixel(position, color);
    }
}

#[track_caller]
fn unsupported_binding(binding: Binding) -> ! {
    panic!("Binding not supported: {binding:?}");
}

fn bytes_of_bool_as_u32(b: bool) -> &'static [u8] {
    if b {
        bytemuck::bytes_of(&1u32)
    }
    else {
        bytemuck::bytes_of(&0u32)
    }
}

pub fn copy_shader_inputs_to_stack<'a, B, I>(
    stack_frame: &mut StackFrame<B>,
    module: &'a ShaderModuleInner,
    inputs: I,
    argument: &FunctionArgument,
) -> Variable<'a>
where
    B: WriteMemory<BindingAddress>,
    I: ReadMemory<Binding>,
{
    let variable = stack_frame.allocate_variable(argument.ty, module);

    ApplyBindings {
        types: &module.module.types,
        apply_binding: ApplyInputs {
            memory: &mut stack_frame.memory,
            inputs,
        },
    }
    .apply(argument.binding.as_ref(), argument.ty, variable);

    variable
}

pub fn copy_shader_outputs_from_stack<B, O>(
    stack_frame: &StackFrame<B>,
    module: &ShaderModuleInner,
    outputs: O,
    result: &FunctionResult,
    result_variable: Variable,
) where
    B: ReadMemory<BindingAddress>,
    O: WriteMemory<Binding>,
{
    ApplyBindings {
        types: &module.module.types,
        apply_binding: ApplyOutputs {
            memory: &stack_frame.memory,
            outputs,
        },
    }
    .apply(result.binding.as_ref(), result.ty, result_variable);
}

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

#[derive(Debug)]
pub struct ApplyInputs<M, I> {
    pub memory: M,
    pub inputs: I,
}

impl<M, I> ApplyBinding for ApplyInputs<M, I>
where
    M: WriteMemory<Slice>,
    I: ReadMemory<Binding>,
{
    fn apply(&mut self, binding: &Binding, ty: Handle<Type>, variable: Variable) {
        let value = self.inputs.read(*binding);
        self.memory.write(variable.slice, value);
    }
}

#[derive(Debug)]
pub struct ApplyOutputs<M, O> {
    pub memory: M,
    pub outputs: O,
}

impl<M, O> ApplyBinding for ApplyOutputs<M, O>
where
    M: ReadMemory<Slice>,
    O: WriteMemory<Binding>,
{
    fn apply(&mut self, binding: &Binding, ty: Handle<Type>, variable: Variable) {
        let value = self.memory.read(variable.slice);
        self.outputs.write(*binding, value);
    }
}
