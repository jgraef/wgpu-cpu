use std::{
    fmt::Debug,
    ops::{
        BitAnd,
        BitOr,
        ControlFlow,
    },
};

use bytemuck::{
    Pod,
    Zeroable,
};
use half::f16;
use naga::{
    BinaryOperator,
    Block,
    Expression,
    Function,
    Handle,
    Literal,
    LocalVariable,
    Range,
    Scalar,
    ScalarKind,
    Statement,
    StructMember,
    Type,
    TypeInner,
    front::Typifier,
    proc::TypeResolution,
};

use crate::{
    bindings::{
        BindingAddress,
        ShaderInput,
        ShaderOutput,
        copy_shader_inputs_to_stack,
        copy_shader_outputs_from_stack,
    },
    memory::{
        Memory,
        Pointer,
        ReadMemory,
        ReadWriteMemory,
        Slice,
        Stack,
        StackFrame,
        WriteMemory,
    },
    module::{
        EntryPointIndex,
        ShaderModule,
    },
    util::{
        CoArena,
        SparseCoArena,
    },
};

#[derive(Debug)]
pub struct Interpreter<Module, Bindings> {
    pub module: Module,
    pub memory: Memory<Bindings>,
}

impl<Module, Bindings> Interpreter<Module, Bindings> {
    pub fn new(module: Module, bindings: Bindings) -> Self {
        Self {
            module,
            memory: Memory {
                stack: Stack::new(0x1000),
                bindings,
            },
        }
    }
}

impl<Module, Bindings> Interpreter<Module, Bindings>
where
    Module: AsRef<ShaderModule>,
{
    pub fn run_entry_point<I, O>(
        &mut self,
        entry_point_index: EntryPointIndex,
        inputs: I,
        outputs: O,
    ) where
        I: ShaderInput,
        O: ShaderOutput,
        Bindings: ReadWriteMemory<BindingAddress>,
    {
        let module = self.module.as_ref();
        let entry_point = &module[entry_point_index];

        {
            let mut outer_frame = self.memory.stack_frame();

            let result_variable =
                entry_point.function.result.as_ref().map(|function_result| {
                    outer_frame.allocate_variable(function_result.ty, module)
                });

            let argument_variables = entry_point
                .function
                .arguments
                .iter()
                .map(|argument| {
                    copy_shader_inputs_to_stack(&mut outer_frame, module, &inputs, argument)
                })
                .collect::<Vec<_>>();

            // drop inputs now. if they're a pooled resource it can be freed early
            drop(inputs);

            let local_variables = CoArena::from_arena(
                &entry_point.function.local_variables,
                |handle, local_variable: &LocalVariable| {
                    outer_frame.allocate_variable(local_variable.ty, module)
                },
            );

            let mut function_context = FunctionContext {
                module,
                function: &entry_point.function,
                typifier: &module.expression_types[entry_point_index],
                emitted_expression: SparseCoArena::default(),
                argument_variables,
                local_variables,
            };

            RunFunction {
                stack_frame: outer_frame.frame(),
                context: &mut function_context,
            }
            .run(result_variable);

            if let Some(result) = &entry_point.function.result {
                copy_shader_outputs_from_stack(
                    &outer_frame,
                    module,
                    outputs,
                    &result,
                    result_variable.unwrap(),
                );
            }
        }

        tracing::trace!(stack_size = self.memory.stack.allocated());
    }
}

#[derive(Debug)]
pub struct FunctionContext<'module> {
    module: &'module ShaderModule,
    function: &'module Function,
    typifier: &'module Typifier,
    emitted_expression: SparseCoArena<Expression, Variable<'module>>,
    argument_variables: Vec<Variable<'module>>,
    local_variables: CoArena<LocalVariable, Variable<'module>>,
}

#[derive(Debug)]
pub struct RunFunction<'module, 'memory, 'function, B> {
    stack_frame: StackFrame<'memory, B>,
    context: &'function mut FunctionContext<'module>,
}

impl<'module, 'memory, 'function, B> RunFunction<'module, 'memory, 'function, B>
where
    B: ReadWriteMemory<BindingAddress>,
{
    pub fn frame(&mut self) -> RunFunction<'module, '_, '_, B> {
        RunFunction {
            stack_frame: self.stack_frame.frame(),
            context: &mut self.context,
        }
    }

    pub fn run(&mut self, output: Option<Variable>) {
        let result = match self.run_block(&self.context.function.body) {
            ControlFlow::Continue(()) => None,
            ControlFlow::Break(Break::Return(return_value)) => return_value,
        };

        if let Some(result) = result {
            self.evaluate(result, output.unwrap());
        };
    }

    pub fn run_block(&mut self, block: &Block) -> ControlFlow<Break> {
        for statement in block {
            self.run_statement(statement)?;
        }
        ControlFlow::Continue(())
    }

    pub fn run_statement(&mut self, statement: &Statement) -> ControlFlow<Break> {
        match statement {
            Statement::Emit(range) => {
                self.run_emit(range.clone());
            }
            Statement::Block(block) => self.run_block(block)?,
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                self.run_if(*condition, accept, reject)?;
            }
            Statement::Switch { selector, cases } => todo!(),
            Statement::Loop {
                body,
                continuing,
                break_if,
            } => todo!(),
            Statement::Break => todo!(),
            Statement::Continue => todo!(),
            Statement::Return { value } => return ControlFlow::Break(Break::Return(*value)),
            Statement::Kill => todo!(),
            Statement::ControlBarrier(barrier) => todo!(),
            Statement::MemoryBarrier(barrier) => todo!(),
            Statement::Store { pointer, value } => todo!(),
            Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => todo!(),
            Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => todo!(),
            Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => todo!(),
            Statement::WorkGroupUniformLoad { pointer, result } => todo!(),
            Statement::Call {
                function,
                arguments,
                result,
            } => todo!(),
            Statement::RayQuery { query, fun } => todo!(),
            Statement::SubgroupBallot { result, predicate } => todo!(),
            Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => todo!(),
            Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => todo!(),
        }

        ControlFlow::Continue(())
    }

    pub fn run_emit(&mut self, range: Range<Expression>) {
        self.context.emitted_expression.reserve_range(&range);

        for handle in range {
            if self.context.emitted_expression.contains(handle) {
                let variable = self
                    .stack_frame
                    .allocate_variable(&self.context.typifier[handle], &self.context.module);

                self.evaluate(handle, variable);

                self.context.emitted_expression.insert(handle, variable);
            }
        }
    }

    pub fn run_if(
        &mut self,
        condition: Handle<Expression>,
        accept: &Block,
        reject: &Block,
    ) -> ControlFlow<Break> {
        let condition = {
            let mut inner = self.frame();

            let condition_ty = TypeResolution::Value(TypeInner::Scalar(Scalar::BOOL));
            let condition_output = inner
                .stack_frame
                .allocate_variable(&condition_ty, &inner.context.module);

            inner.evaluate(condition, condition_output);

            *condition_output.read::<u32, _>(&inner.stack_frame.memory)
        };

        if condition != 0 {
            self.run_block(accept)?;
        }
        else {
            self.run_block(reject)?;
        }

        ControlFlow::Continue(())
    }

    pub fn evaluate(&mut self, expression: Handle<Expression>, output: Variable) {
        if let Some(variable) = self.context.emitted_expression.get(expression) {
            // expression was emitted before, so we just need to copy from there
            output.copy_from(*variable, &mut self.stack_frame.memory);
        }
        else {
            let expression = &self.context.function.expressions[expression];

            // this always creates a new stack frame and cleans it up at the end of the
            // evaluation
            let mut inner = self.frame();
            inner.evaluate_inner(expression, output);
        }
    }

    fn evaluate_inner(&mut self, expression: &Expression, output: Variable) {
        tracing::trace!(?expression, "evaluate");

        match expression {
            Expression::Literal(literal) => {
                write_literal(literal, output, &mut self.stack_frame.memory);
            }
            Expression::Constant(handle) => todo!(),
            Expression::Override(handle) => todo!(),
            Expression::ZeroValue(handle) => todo!(),
            Expression::Compose { ty, components } => {
                self.evaluate_compose(*ty, &components, output);
            }
            Expression::Access { base, index } => todo!(),
            Expression::AccessIndex { base, index } => {
                tracing::trace!(?base, ?index, ?output, "access index");

                let base_ty = &self.context.typifier[*base];
                let mut base_variable = self
                    .stack_frame
                    .allocate_variable(base_ty, &self.context.module);

                tracing::trace!(?base_ty, ?base_variable, "evaluating base");
                self.evaluate(*base, base_variable);

                let mut produce_pointer = false;
                if let Some(base_deref) =
                    base_variable.try_deref(&self.stack_frame.memory, &self.context.module)
                {
                    tracing::trace!(?base_variable, "deref");
                    base_variable = base_deref;
                    produce_pointer = true;
                }

                let component_variable =
                    base_variable.component(*index as usize, output.ty, &self.context.module);
                tracing::trace!(?component_variable, "index");

                if produce_pointer {
                    let pointer = component_variable.pointer();
                    *output.write(&mut self.stack_frame.memory) = pointer;
                }
                else {
                    output.copy_from(component_variable, &mut self.stack_frame.memory);
                }
            }
            Expression::Splat { size, value } => todo!(),
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => todo!(),
            Expression::FunctionArgument(index) => {
                let index = *index as usize;
                let input_variable = self.context.argument_variables[index];
                let function_argument = &self.context.function.arguments[index];
                output.copy_from(input_variable, &mut self.stack_frame.memory);
            }
            Expression::GlobalVariable(handle) => todo!(),
            Expression::LocalVariable(handle) => {
                let local_variable = &self.context.local_variables[*handle];
                *output.write(self.stack_frame.memory) = local_variable.pointer();
            }
            Expression::Load { pointer } => todo!(),
            Expression::ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge,
            } => todo!(),
            Expression::ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => todo!(),
            Expression::ImageQuery { image, query } => todo!(),
            Expression::Unary { op, expr } => todo!(),
            Expression::Binary { op, left, right } => {
                self.evaluate_binary(*op, *left, *right, output);
            }
            Expression::Select {
                condition,
                accept,
                reject,
            } => todo!(),
            Expression::Derivative { axis, ctrl, expr } => todo!(),
            Expression::Relational { fun, argument } => todo!(),
            Expression::Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => todo!(),
            Expression::As {
                expr,
                kind,
                convert,
            } => {
                self.evaluate_as(*expr, *kind, *convert, output);
            }
            Expression::CallResult(handle) => todo!(),
            Expression::AtomicResult { ty, comparison } => todo!(),
            Expression::WorkGroupUniformLoadResult { ty } => todo!(),
            Expression::ArrayLength(handle) => todo!(),
            Expression::RayQueryVertexPositions { query, committed } => todo!(),
            Expression::RayQueryProceedResult => todo!(),
            Expression::RayQueryGetIntersection { query, committed } => todo!(),
            Expression::SubgroupBallotResult => todo!(),
            Expression::SubgroupOperationResult { ty } => todo!(),
        }
    }

    pub fn evaluate_compose(
        &mut self,
        ty: Handle<Type>,
        components: &[Handle<Expression>],
        output: Variable,
    ) {
        for (i, component_handle) in components.into_iter().enumerate() {
            let component_ty = &self.context.typifier[*component_handle];
            let variable = output.component(i, component_ty, self.context.module);
            self.evaluate(*component_handle, variable);
        }
    }

    pub fn evaluate_as(
        &mut self,
        expression: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<u8>,
        output: Variable,
    ) {
        let output_variable = output;
        let input_ty = &self.context.typifier[expression];

        let input_variable = if let Some(new_width) = convert {
            // allocate some temporary variable for this
            self.stack_frame
                .allocate_variable(input_ty, self.context.module)
        }
        else {
            // this is a bitcast, so we can evaluate to the same output variable
            output_variable.cast(input_ty)
        };

        self.evaluate(expression, input_variable);

        let input_ty_inner = input_variable.ty.inner_with(&self.context.module);
        let output_ty_inner = output_variable.ty.inner_with(&self.context.module);

        match (input_ty_inner, output_ty_inner) {
            (TypeInner::Scalar(input_scalar), TypeInner::Scalar(output_scalar)) => {
                convert_scalar(
                    input_variable,
                    *input_scalar,
                    output_variable,
                    *output_scalar,
                    &mut self.stack_frame.memory,
                );
            }
            (
                TypeInner::Vector { size, scalar: left },
                TypeInner::Vector {
                    size: size_right,
                    scalar: right,
                },
            ) => {
                assert_eq!(size, size_right);
                todo!();

                // todo
            }
            _ => panic!("Invalid cast from {input_ty_inner:?} to {output_ty_inner:?}"),
        }
    }

    pub fn evaluate_binary(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
        output: Variable,
    ) {
        let left_ty = &self.context.typifier[left];
        let left_variable = self
            .stack_frame
            .allocate_variable(left_ty, self.context.module);
        self.evaluate(left, left_variable);

        let right_ty = &self.context.typifier[right];
        let right_variable = self
            .stack_frame
            .allocate_variable(right_ty, self.context.module);
        self.evaluate(right, right_variable);

        let left_ty_inner = left_ty.inner_with(&self.context.module.module.types);
        let right_ty_inner = left_ty.inner_with(&self.context.module.module.types);

        match (left_ty_inner, right_ty_inner) {
            (TypeInner::Scalar(left), TypeInner::Scalar(right)) if left == right => {
                scalar_binary_op(
                    op,
                    *left,
                    left_variable,
                    right_variable,
                    output,
                    &mut self.stack_frame.memory,
                );
            }

            (
                TypeInner::Vector {
                    size: left_size,
                    scalar: left_scalar,
                },
                TypeInner::Vector {
                    size: right_size,
                    scalar: right_scalar,
                },
            ) if left_size == right_size && left_scalar == right_scalar => {
                todo!("vector binary operation")
            }

            (
                TypeInner::Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                TypeInner::Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if (op == BinaryOperator::Add || op == BinaryOperator::Subtract)
                && left_columns == right_columns
                && left_rows == right_rows
                && left_scalar == right_scalar =>
            {
                todo!("matrix add/subtract")
            }

            (
                TypeInner::Scalar(scalar),
                TypeInner::Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
            )
            | (
                TypeInner::Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
                TypeInner::Scalar(scalar),
            ) if scalar == vector_scalar && op == BinaryOperator::Multiply => {
                todo!("vector * scalar")
            }

            (
                TypeInner::Scalar(scalar),
                TypeInner::Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
            )
            | (
                TypeInner::Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
                TypeInner::Scalar(scalar),
            ) if scalar == matrix_scalar && op == BinaryOperator::Multiply => {
                todo!("vector * scalar")
            }

            (
                TypeInner::Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                TypeInner::Vector {
                    size: right_size,
                    scalar: right_scalar,
                },
            ) if op == BinaryOperator::Multiply
                && left_scalar == right_scalar
                && right_size == left_columns =>
            {
                todo!("matrix * vector");
            }

            (
                TypeInner::Vector {
                    size: left_size,
                    scalar: left_scalar,
                },
                TypeInner::Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if op == BinaryOperator::Multiply
                && left_scalar == right_scalar
                && left_size == right_rows =>
            {
                todo!("vector * matrix")
            }

            _ => {
                panic!("Invalid binary op {op:?} between {left_ty_inner:?} and {right_ty_inner:?}")
            }
        }
    }
}

#[derive(Debug)]
pub enum Break {
    Return(Option<Handle<Expression>>),
}

// todo: this could always contain a `&'a TypeInner` because we have the
// life-time anyway and we can make the lookup at construction
#[derive(Clone, Copy, Debug)]
pub enum VariableType<'a> {
    Handle(Handle<Type>),
    Inner(&'a TypeInner),
}

impl<'a> VariableType<'a> {
    pub fn inner_with(&self, module: &'a ShaderModule) -> &'a TypeInner {
        match self {
            VariableType::Handle(handle) => &module.module.types[*handle].inner,
            VariableType::Inner(type_inner) => *type_inner,
        }
    }
}

impl From<Handle<Type>> for VariableType<'static> {
    fn from(value: Handle<Type>) -> Self {
        Self::Handle(value)
    }
}

impl<'a> From<&'a Handle<Type>> for VariableType<'static> {
    fn from(value: &'a Handle<Type>) -> Self {
        Self::from(*value)
    }
}

impl<'a> From<&'a TypeInner> for VariableType<'a> {
    fn from(value: &'a TypeInner) -> Self {
        Self::Inner(value)
    }
}

impl<'a> From<&'a TypeResolution> for VariableType<'a> {
    fn from(value: &'a TypeResolution) -> Self {
        match value {
            TypeResolution::Handle(handle) => Self::Handle(*handle),
            TypeResolution::Value(type_inner) => Self::Inner(type_inner),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Variable<'a> {
    pub ty: VariableType<'a>,
    pub slice: Slice,
}

impl<'a> Variable<'a> {
    pub fn read<'m, T, M>(&self, memory: &'m M) -> &'m T
    where
        T: Pod,
        M: ReadMemory<Slice>,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes(&memory.read(self.slice.slice(..n)))
    }

    pub fn write<'m, T, M>(&self, memory: &'m mut M) -> &'m mut T
    where
        T: Pod,
        M: WriteMemory<Slice>,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes_mut(memory.write(self.slice.slice(..n)))
    }

    pub fn copy_from<M>(&self, source: Variable, memory: &mut M)
    where
        M: ReadWriteMemory<Slice>,
    {
        memory.copy(source.slice, self.slice);
    }

    pub fn cast(&self, ty: impl Into<VariableType<'a>>) -> Self {
        Self {
            ty: ty.into(),
            slice: self.slice,
        }
    }

    pub fn debug<M>(&self, module: &'a ShaderModule, memory: &'a M) -> VariableDebug<'a, M>
    where
        M: ReadMemory<Slice>,
    {
        VariableDebug {
            variable: *self,
            module,
            memory,
        }
    }

    pub fn component(
        &self,
        index: usize,
        component_ty: impl Into<VariableType<'a>>,
        module: &ShaderModule,
    ) -> Variable<'a> {
        let component_ty = component_ty.into();
        let offset = module.offset_of(self.ty, component_ty, index);
        Variable {
            ty: component_ty,
            slice: self.slice.slice(offset..),
        }
    }

    pub fn member(&self, member: &StructMember) -> Variable<'a> {
        Variable {
            ty: member.ty.into(),
            slice: self.slice.slice(member.offset..),
        }
    }

    pub fn pointer(&self) -> Pointer {
        Pointer::from(self.slice)
    }

    pub fn try_deref<M>(&self, memory: &M, module: &ShaderModule) -> Option<Variable<'a>>
    where
        M: ReadMemory<Slice>,
    {
        let ty_inner = self.ty.inner_with(module);
        match ty_inner {
            TypeInner::Pointer { base, space } => {
                let ty = base.into();
                let pointer = self.read::<Pointer, M>(memory);
                let size = module.size_of(ty);
                let slice = pointer.deref(*space, size);

                Some(Variable { ty, slice })
            }
            TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => {
                /*let ty = scalar.into();
                let pointer = self.read::<Pointer, M>(memory);
                let size = module.size_of(ty);
                let slice = pointer.deref(*space, size);

                Some(Variable { ty, slice })*/
                todo!("pain!");
            }
            _ => None,
        }
    }
}

fn scalar_binary_op<M>(
    op: BinaryOperator,
    scalar_ty: Scalar,
    left: Variable,
    right: Variable,
    output: Variable,
    memory: &mut M,
) where
    M: ReadWriteMemory<Slice>,
{
    use std::ops::*;

    macro_rules! binary_ops {
        (@emit_for_ty(($scalar:ident, $width:pat, $ty:ty), [$(($op:ident, $func:path)),*])) => {
            match (scalar_ty.kind, scalar_ty.width, op) {
                $(
                    (ScalarKind::$scalar, $width, BinaryOperator::$op) => {
                        let left = left.read::<$ty, M>(memory);
                        let right = right.read::<$ty, M>(memory);
                        let result = $func(left, right);
                        *output.write::<_, M>(memory) = result;
                        return;
                    },
                )*
                _ => {}
            }

        };
        (@emit_for_many([$($arg:tt),*], $ops:tt)) => {
            $(
                binary_ops!(@emit_for_ty($arg, $ops));
            )*
        };
        ($(
            [$($scalar:ident @ $width:pat => $ty:ty),*]: [$($op:ident => $func:path),*];
        )*) => {
            $(
                binary_ops!(@emit_for_many([$(($scalar, $width, $ty)),*], [$(($op, $func)),*]));
            )*

            panic!("Invalid binary op: {left:?} {op:?} {right:?} ({scalar_ty:?})");
        };
    }

    macro_rules! impl_comparisions {
        ($($func:ident => $trait:ident :: $method:ident,)*) => {
            $(
                fn $func<L, R>(left: &L, right: &R) -> Bool
                where
                    L: $trait<R>
                {
                    Bool::from($trait::$method(left, right))
                }
            )*
        };
    }

    impl_comparisions!(
        equal => PartialEq::eq,
        not_equal => PartialEq::ne,
        less => PartialEq::eq,
        less_equal => PartialEq::eq,
        greater => PartialEq::eq,
        greater_equal => PartialEq::eq,
    );

    binary_ops!(
        [Bool@1 => Bool]: [LogicalAnd => BitAnd::bitand, LogicalOr => BitOr::bitor];
        [
            Sint@4 => i32,
            Uint@4 => u32,
            Float@2 => f16,
            Float@4 => f32
        ]: [
            Add => Add::add,
            Subtract => Sub::sub,
            Multiply => Mul::mul,
            Divide => Div::div,
            Modulo => Rem::rem
        ];
        [
            Sint@4 => i32,
            Uint@4 => u32
        ]: [
            And => BitAnd::bitand,
            ExclusiveOr => BitOr::bitor,
            InclusiveOr => BitXor::bitxor,
            ShiftLeft => Shl::shl,
            ShiftRight => Shr::shr
        ];
        [
            Bool@1 => Bool,
            Sint@4 => i32,
            Uint@4 => u32,
            Float@2 => f16,
            Float@4 => f32
        ]: [
            Equal => equal,
            NotEqual => not_equal
        ];
        [
            Sint@4 => i32,
            Uint@4 => u32,
            Float@2 => f16,
            Float@4 => f32
        ]: [
            Less => less,
            LessEqual => less_equal,
            Greater => greater,
            GreaterEqual => greater_equal
        ];
    );
}

/// # FIXME
///
/// Naga makes bools 1 byte wide, although WebGPU specifies them to be 4 byte.
/// This might be a bug, or not. bools don't specify an internal layout though,
/// so we can just use an `u8`. See [`super::tests::naga_bool_width_is_32bit`].
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
pub struct Bool(u8);

impl From<bool> for Bool {
    fn from(value: bool) -> Self {
        if value { Self(1) } else { Self(0) }
    }
}

impl From<Bool> for bool {
    fn from(value: Bool) -> Self {
        value.0 != 0
    }
}

impl BitAnd for Bool {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::from(bool::from(self) && bool::from(rhs))
    }
}

impl BitOr for Bool {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::from(bool::from(self) || bool::from(rhs))
    }
}

impl BitAnd for &Bool {
    type Output = Bool;

    fn bitand(self, rhs: Self) -> Self::Output {
        Bool::from(bool::from(*self) && bool::from(*rhs))
    }
}

impl BitOr for &Bool {
    type Output = Bool;

    fn bitor(self, rhs: Self) -> Self::Output {
        Bool::from(bool::from(*self) || bool::from(*rhs))
    }
}

impl Eq for Bool {}

impl PartialEq for Bool {
    fn eq(&self, other: &Self) -> bool {
        bool::from(*self) == bool::from(*other)
    }
}

fn convert_scalar<M>(
    input_variable: Variable,
    input_scalar: Scalar,
    output_variable: Variable,
    output_scalar: Scalar,
    memory: &mut M,
) where
    M: ReadWriteMemory<Slice>,
{
    macro_rules! convert_scalar {
        ($(($scalar_in:ident @ $width_in:pat) as ($scalar_out:ident @ $width_out:pat) => $func:expr;)*) => {
            match (
                input_scalar.kind,
                input_scalar.width,
                output_scalar.kind,
                output_scalar.width,
            ) {
                $(
                    (ScalarKind::$scalar_in, $width_in, ScalarKind::$scalar_out, $width_out) => {
                        let input = input_variable.read(memory);
                        let output = $func(*input);
                        *output_variable.write(memory) = output;
                    }
                )*
                _ => panic!("Unsupported cast from {input_scalar:?} to {output_scalar:?}"),
            }
        };
    }

    fn switch<T>(x: Bool, accept: T, reject: T) -> T {
        if bool::from(x) { accept } else { reject }
    }

    // these are specified here: https://gpuweb.github.io/gpuweb/wgsl/#value-constructor-builtin-function
    convert_scalar!(
        (Sint@4) as (Uint@4) => (|x: i32| x as u32);
        (Uint@4) as (Sint@4) => (|x: u32| x as i32);
        (Sint@4) as (Float@4) => (|x: i32| x as f32);
        (Uint@4) as (Float@4) => (|x: u32| x as f32);
        (Bool@1) as (Uint@4) => (|x: Bool| switch(x, 1u32, 0u32));
        (Bool@1) as (Sint@4) => (|x: Bool| switch(x, 1i32, 0i32));
        (Bool@1) as (Float@4) => (|x: Bool| switch(x, 1.0f32, 0.0f32));
    );
}

fn write_literal<M>(literal: &Literal, output: Variable, memory: &mut M)
where
    M: WriteMemory<Slice>,
{
    macro_rules! write_literal {
        ($($variant:ident),*) => {
            match literal {
                $(
                    Literal::$variant(value) => {
                        *output.write(memory) = *value;
                    }
                )*
                _ => panic!("Unsupported literal: {literal:?}"),
            }
        };
    }

    write_literal!(F32, F16, U32, I32);
}

#[derive(Clone, Copy)]
pub struct VariableDebug<'a, M> {
    variable: Variable<'a>,
    module: &'a ShaderModule,
    memory: &'a M,
}

impl<'a, M> VariableDebug<'a, M>
where
    M: ReadMemory<Slice>,
{
    fn write_scalar(
        &self,
        variable: Variable,
        scalar: &Scalar,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        macro_rules! write_scalar {
                ($(($scalar:ident @ $width:pat) as $ty:ty;)*) => {
                    match (
                        scalar.kind,
                        scalar.width,
                    ) {
                        $(
                            (ScalarKind::$scalar, $width) => {
                                let input = variable.read::<$ty, M>(self.memory);
                                write!(f, "{input:?}")?;
                            }
                        )*
                        _ => {
                            write!(f, "(?{scalar:?})")?;
                        },
                    }
                };
            }

        write_scalar!(
            (Sint@4) as i32;
            (Uint@4) as u32;
            (Float@4) as f32;
            (Float@2) as f16;
        );

        Ok(())
    }
}

impl<'a, M> Debug for VariableDebug<'a, M>
where
    M: ReadMemory<Slice>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ty = self.variable.ty.inner_with(&self.module);

        match ty {
            TypeInner::Scalar(scalar) => self.write_scalar(self.variable, scalar, f)?,
            TypeInner::Vector { size, scalar } => {
                let component_ty = TypeInner::Scalar(*scalar);
                write!(f, "[")?;
                for i in 0..(*size as u8) {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    let component = self
                        .variable
                        .component(i as usize, &component_ty, self.module);

                    self.write_scalar(component, scalar, f)?;
                }
                write!(f, "]")?;
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => todo!(),
            TypeInner::Atomic(scalar) => todo!(),
            TypeInner::Pointer { base, space } => todo!(),
            TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => todo!(),
            TypeInner::Array { base, size, stride } => todo!(),
            TypeInner::Struct { members, span } => todo!(),
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => todo!(),
            TypeInner::Sampler { comparison } => todo!(),
            TypeInner::AccelerationStructure { vertex_return } => todo!(),
            TypeInner::RayQuery { vertex_return } => todo!(),
            TypeInner::BindingArray { base, size } => todo!(),
        }

        Ok(())
    }
}
