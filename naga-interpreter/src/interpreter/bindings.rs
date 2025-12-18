use naga::{
    Binding,
    FunctionArgument,
    FunctionResult,
    Handle,
    Type,
    proc::Layouter,
};

use crate::{
    bindings::{
        BindingAddress,
        IoBindingVisitor,
        ShaderInput,
        ShaderOutput,
        VisitIoBindings,
    },
    interpreter::{
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
};

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
