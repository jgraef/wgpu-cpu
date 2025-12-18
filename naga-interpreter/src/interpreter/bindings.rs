use std::ops::Index;

use naga::{
    Binding,
    FunctionArgument,
    FunctionResult,
    Handle,
    Module,
    Type,
    proc::Layouter,
};

use crate::{
    bindings::{
        BindingAddress,
        BindingLocationLayout,
        IoBindingVisitor,
        ShaderInput,
        ShaderOutput,
        UserDefinedInterStageLayout,
        VisitIoBindings,
    },
    interpreter::{
        EntryPointIndex,
        ShaderModule,
        memory::{
            Slice,
            StackFrame,
        },
        variable::Variable,
    },
    memory::{
        ReadMemory,
        WriteMemory,
    },
    util::SparseVec,
};

#[derive(Clone, Debug)]
pub struct UserDefinedIoLayouts {
    pub(super) inner: SparseVec<UserDefinedIoLayout>,
}

impl Index<EntryPointIndex> for UserDefinedIoLayouts {
    type Output = UserDefinedIoLayout;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.inner[index.0]
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
