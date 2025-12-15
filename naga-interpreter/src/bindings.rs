use std::{
    ops::{
        Index,
        Range,
    },
    sync::Arc,
};

use naga::{
    Binding,
    FunctionArgument,
    FunctionResult,
    Handle,
    Module,
    Type,
    TypeInner,
    UniqueArena,
    proc::Layouter,
};

use crate::{
    interpreter::Variable,
    memory::{
        ReadMemory,
        Slice,
        StackFrame,
        WriteMemory,
    },
    module::ShaderModule,
    util::SparseVec,
};

#[derive(Clone, Copy, Debug)]
pub struct BindingAddress {
    _todo: (),
}

pub trait ShaderInput {
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]);
}

impl<T> ShaderInput for &T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

impl<T> ShaderInput for &mut T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

pub trait ShaderOutput {
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]);
}

impl<T> ShaderOutput for &mut T
where
    T: ShaderOutput,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        T::read_from(self, binding, ty, source);
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From, derive_more::Into,
)]
pub struct BindingLocation(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct BindingLocationLayout {
    pub offset: u32,
    pub size: u32,
}

impl BindingLocationLayout {
    pub fn range(&self) -> Range<usize> {
        let start = self.offset as usize;
        let end = (self.offset + self.size) as usize;
        start..end
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageLayout {
    locations: Arc<[Option<BindingLocationLayout>]>,
    size: u32,
}

impl UserDefinedInterStageLayout {
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl FromIterator<Option<BindingLocationLayout>> for UserDefinedInterStageLayout {
    fn from_iter<T: IntoIterator<Item = Option<BindingLocationLayout>>>(iter: T) -> Self {
        let mut size = None;
        let locations = iter
            .into_iter()
            .inspect(|location| {
                if let Some(location) = location {
                    let update_size: &mut u32 = size.get_or_insert_default();
                    *update_size = (*update_size).max(location.offset + location.size);
                }
            })
            .collect();

        Self {
            locations,
            size: size.unwrap_or_default(),
        }
    }
}

impl Index<BindingLocation> for UserDefinedInterStageLayout {
    type Output = Option<BindingLocationLayout>;

    fn index(&self, index: BindingLocation) -> &Self::Output {
        let index: usize = index.0.try_into().unwrap();
        &self.locations[index]
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageBuffer {
    data: Vec<u8>,
    layout: UserDefinedInterStageLayout,
}

impl UserDefinedInterStageBuffer {
    pub fn new(layout: UserDefinedInterStageLayout) -> Self {
        let data = vec![0; layout.size as usize];
        Self { data, layout }
    }
}

impl ReadMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn read(&self, address: BindingLocation) -> &[u8] {
        self.layout[address]
            .as_ref()
            .map_or(&[], |layout| &self.data[layout.range()])
    }
}

impl WriteMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn write(&mut self, address: BindingLocation) -> &mut [u8] {
        self.layout[address]
            .as_ref()
            .map_or(&mut [], |layout| &mut self.data[layout.range()])
    }
}

pub fn copy_shader_inputs_to_stack<'a, B, I>(
    stack_frame: &mut StackFrame<B>,
    module: &'a ShaderModule,
    inputs: I,
    argument: &FunctionArgument,
) -> Variable<'a, 'a>
where
    B: WriteMemory<BindingAddress>,
    I: ShaderInput,
{
    let variable = Variable::allocate(argument.ty, module, stack_frame);

    IoBindingVisitor {
        types: &module.module.types,
        visit: CopyInputsToMemory {
            slice: variable.slice(),
            memory: &mut stack_frame.memory,
            inputs,
            layouter: &module.layouter,
        },
    }
    .visit_function_argument(argument, 0);

    variable
}

pub fn copy_shader_outputs_from_stack<B, O>(
    stack_frame: &StackFrame<B>,
    module: &ShaderModule,
    outputs: O,
    result: &FunctionResult,
    variable: Variable,
) where
    B: ReadMemory<BindingAddress>,
    O: ShaderOutput,
{
    IoBindingVisitor {
        types: &module.module.types,
        visit: CopyOutputsFromMemory {
            slice: variable.slice(),
            memory: &stack_frame.memory,
            outputs,
            layouter: &module.layouter,
        },
    }
    .visit_function_result(result, 0);
}

pub trait VisitIoBindings {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    );
}

impl<T> VisitIoBindings for &mut T
where
    T: VisitIoBindings,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        T::visit(self, binding, ty_handle, ty, offset, name);
    }
}

#[derive(Debug)]
pub struct IoBindingVisitor<'a, B> {
    pub types: &'a UniqueArena<Type>,
    pub visit: B,
}

impl<'a, B> IoBindingVisitor<'a, B>
where
    B: VisitIoBindings,
{
    pub fn visit(
        &mut self,
        binding: Option<&Binding>,
        ty_handle: Handle<Type>,
        offset: u32,
        name: Option<&str>,
    ) {
        let ty = &self.types[ty_handle];
        if let Some(binding) = binding {
            self.visit.visit(binding, ty_handle, ty, offset, name);
        }
        else {
            match &ty.inner {
                TypeInner::Struct { members, span } => {
                    for member in members {
                        self.visit(
                            member.binding.as_ref(),
                            member.ty,
                            offset + member.offset,
                            member.name.as_deref(),
                        );
                    }
                }
                _ => panic!("Invalid binding type: {:?}", ty),
            }
        }
    }

    pub fn visit_function_argument(&mut self, argument: &FunctionArgument, offset: u32) {
        self.visit(
            argument.binding.as_ref(),
            argument.ty,
            offset,
            argument.name.as_deref(),
        );
    }

    pub fn visit_function_result(&mut self, result: &FunctionResult, offset: u32) {
        self.visit(result.binding.as_ref(), result.ty, offset, None);
    }
}

#[derive(Debug)]
pub struct CopyInputsToMemory<'layouter, M, I> {
    pub slice: Slice,
    pub memory: M,
    pub inputs: I,
    pub layouter: &'layouter Layouter,
}

impl<'layouter, M, I> VisitIoBindings for CopyInputsToMemory<'layouter, M, I>
where
    M: WriteMemory<Slice>,
    I: ShaderInput,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        let type_layout = &self.layouter[ty_handle];
        let target = self
            .memory
            .write(self.slice.slice(offset..offset + type_layout.size));
        self.inputs.write_into(binding, ty, target);
    }
}

#[derive(Debug)]
pub struct CopyOutputsFromMemory<'layouter, M, O> {
    pub slice: Slice,
    pub memory: M,
    pub outputs: O,
    pub layouter: &'layouter Layouter,
}

impl<'layouter, M, O> VisitIoBindings for CopyOutputsFromMemory<'layouter, M, O>
where
    M: ReadMemory<Slice>,
    O: ShaderOutput,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        let type_layout = &self.layouter[ty_handle];
        let source = self
            .memory
            .read(self.slice.slice(offset..offset + type_layout.size));
        self.outputs.read_from(binding, ty, source);
    }
}

#[derive(Clone, Debug)]
pub struct CollectUserDefinedInterStageLayout<'module> {
    pub layouter: &'module Layouter,
    pub buffer_offset: u32,
    pub locations: SparseVec<BindingLocationLayout>,
}

impl<'module> VisitIoBindings for CollectUserDefinedInterStageLayout<'module> {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        // this is the offset in the struct that contains this inter-stage location
        // binding. we don't care about this, since we can layout our
        // inter-stage buffer as we want. in particular the layout of the vertex
        // output and fragment input might not even match.
        let _ = offset;

        match binding {
            Binding::BuiltIn(_builtin) => {
                // nop
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                let type_layout = self.layouter[ty_handle];
                let offset = type_layout.alignment.round_up(self.buffer_offset);
                let size = type_layout.size;
                self.buffer_offset = offset + size;

                let index = *location as usize;
                self.locations
                    .insert(index, BindingLocationLayout { offset, size });
            }
        }
    }
}

pub fn collect_user_defined_inter_stage_layout_from_function_arguments<'a>(
    module: &Module,
    layouter: &Layouter,
    arguments: impl IntoIterator<Item = &'a FunctionArgument>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::default(),
    };

    for argument in arguments {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_argument(argument, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

pub fn collect_user_defined_inter_stage_layout_from_function_result<'a>(
    module: &Module,
    layouter: &Layouter,
    result: impl Into<Option<&'a FunctionResult>>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::new(),
    };

    if let Some(result) = result.into() {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_result(result, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

#[derive(Clone, Debug)]
pub enum UserDefinedIoLayout {
    Vertex {
        // todo: input
        output: UserDefinedInterStageLayout,
    },
    Fragment {
        input: UserDefinedInterStageLayout,
        // output: don't need it, but would be handy for verification
    },
}
