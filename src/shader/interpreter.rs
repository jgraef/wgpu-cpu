use std::{
    fmt::Debug,
    ops::{
        Add,
        ControlFlow,
    },
};

use bytemuck::Pod;
use half::f16;
use naga::{
    BinaryOperator,
    Block,
    Expression,
    Function,
    Handle,
    Literal,
    Scalar,
    ScalarKind,
    Statement,
    StructMember,
    Type,
    TypeInner,
    front::Typifier,
    proc::TypeResolution,
};

use crate::shader::{
    EntryPointIndex,
    ShaderModule,
    ShaderModuleInner,
    bindings::{
        ApplyBindings,
        ApplyInput,
        ApplyOutput,
        ShaderInput,
        ShaderOutput,
    },
};

#[derive(Debug)]
pub struct VirtualMachine {
    module: ShaderModule,
    stack: Stack,
}

impl<'a> VirtualMachine {
    pub fn new(module: ShaderModule) -> Self {
        Self {
            module,
            stack: Stack::new(0x1000),
        }
    }

    pub fn run_entry_point<I, O>(
        &mut self,
        entry_point_index: EntryPointIndex,
        input: &I,
        output: &mut O,
    ) where
        I: ShaderInput,
        O: ShaderOutput,
    {
        let entry_point = &self.module[entry_point_index];

        {
            let mut outer_frame = self.stack.frame();

            let result_variable = entry_point.function.result.as_ref().map(|function_result| {
                outer_frame.allocate_variable(function_result.ty, &self.module.inner)
            });

            let argument_variables = entry_point
                .function
                .arguments
                .iter()
                .map(|argument| {
                    let variable = outer_frame.allocate_variable(argument.ty, &self.module.inner);
                    ApplyBindings {
                        types: &self.module.module.types,
                        apply_binding: ApplyInput {
                            stack: &mut outer_frame.stack,
                            input,
                        },
                    }
                    .apply(argument.binding.as_ref(), argument.ty, variable);
                    variable
                })
                .collect::<Vec<_>>();

            RunFunction {
                module: &self.module.inner,
                function: &entry_point.function,
                typifier: &self.module.inner.expression_types[entry_point_index],
                frame: outer_frame.frame(),
                arguments: &argument_variables,
            }
            .run(result_variable);

            if let Some(result) = &entry_point.function.result {
                ApplyBindings {
                    types: &self.module.module.types,
                    apply_binding: ApplyOutput {
                        stack: &outer_frame.stack,
                        output,
                    },
                }
                .apply(
                    result.binding.as_ref(),
                    result.ty,
                    result_variable.unwrap(),
                );
            }
        }

        tracing::debug!(stack_size = self.stack.allocated());
    }
}

#[derive(derive_more::Debug)]
pub struct RunFunction<'a> {
    module: &'a ShaderModuleInner,
    function: &'a Function,
    typifier: &'a Typifier,
    frame: StackFrame<'a>,
    arguments: &'a [Variable<'a>],
}

impl<'a> RunFunction<'a> {
    pub fn frame(&mut self) -> RunFunction<'_> {
        RunFunction {
            module: &self.module,
            function: &self.function,
            typifier: &self.typifier,
            frame: self.frame.frame(),
            arguments: self.arguments,
        }
    }

    pub fn with_frame<R>(&mut self, f: impl FnOnce(&mut RunFunction) -> R) -> R {
        let mut inner = self.frame();
        f(&mut inner)
    }

    pub fn run(&mut self, output: Option<Variable>) {
        if let Some(result) = match self.run_block(&self.function.body) {
            ControlFlow::Continue(()) => None,
            ControlFlow::Break(Break::Return(return_value)) => return_value,
        } {
            self.evaluate(&self.function.expressions[result], output.unwrap());
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
                // nop?
            }
            Statement::Block(block) => self.run_block(block)?,
            Statement::If {
                condition,
                accept,
                reject,
            } => {
                let condition = self.with_frame(|inner| {
                    let condition = &inner.function.expressions[*condition];
                    let condition_ty = TypeResolution::Value(TypeInner::Scalar(Scalar::BOOL));
                    let condition_output =
                        inner.frame.allocate_variable(&condition_ty, &inner.module);

                    inner.evaluate(condition, condition_output);
                    *condition_output.read::<u32>(&inner.frame.stack)
                });

                if condition != 0 {
                    self.run_block(accept)?;
                }
                else {
                    self.run_block(reject)?;
                }
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

    pub fn evaluate(&mut self, expression: &Expression, output: Variable) {
        tracing::debug!(?expression, "evaluate");

        let mut inner = self.frame();

        match expression {
            Expression::Literal(literal) => {
                write_literal(literal, output, &mut inner.frame.stack);
            }
            Expression::Constant(handle) => todo!(),
            Expression::Override(handle) => todo!(),
            Expression::ZeroValue(handle) => todo!(),
            Expression::Compose { ty, components } => {
                inner.evaluate_compose(*ty, &components, output);
            }
            Expression::Access { base, index } => todo!(),
            Expression::AccessIndex { base, index } => todo!(),
            Expression::Splat { size, value } => todo!(),
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => todo!(),
            Expression::FunctionArgument(index) => {
                let index = *index as usize;
                let input_variable = inner.arguments[index];
                let function_argument = &inner.function.arguments[index];
                output.copy_from(input_variable, &mut inner.frame.stack, &inner.module);
            }
            Expression::GlobalVariable(handle) => todo!(),
            Expression::LocalVariable(handle) => todo!(),
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
                inner.evaluate_binary(*op, *left, *right, output);
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
                inner.evaluate_as(*expr, *kind, *convert, output);
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
            let component_expr = &self.function.expressions[*component_handle];
            let component_ty = &self.typifier[*component_handle];
            let variable = output.component(i, component_ty, self.module);
            self.evaluate(component_expr, variable);
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
        let input_ty = &self.typifier[expression];

        let input_variable = if let Some(new_width) = convert {
            // allocate some temporary variable for this
            self.frame.allocate_variable(input_ty, self.module)
        }
        else {
            // this is a bitcast, so we can evaluate to the same output variable
            output_variable.cast(input_ty)
        };

        self.evaluate(&self.function.expressions[expression], input_variable);

        let input_ty_inner = input_variable.ty.inner_with(&self.module);
        let output_ty_inner = output_variable.ty.inner_with(&self.module);

        match (input_ty_inner, output_ty_inner) {
            (TypeInner::Scalar(input_scalar), TypeInner::Scalar(output_scalar)) => {
                convert_scalar(
                    input_variable,
                    *input_scalar,
                    output_variable,
                    *output_scalar,
                    &mut self.frame.stack,
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
        let left_ty = &self.typifier[left];
        let left_expression = &self.function.expressions[left];
        let left_variable = self.frame.allocate_variable(left_ty, self.module);
        self.evaluate(left_expression, left_variable);

        let right_ty = &self.typifier[right];
        let right_expression = &self.function.expressions[right];
        let right_variable = self.frame.allocate_variable(right_ty, self.module);
        self.evaluate(right_expression, right_variable);

        let left_ty_inner = left_ty.inner_with(&self.module.module.types);
        let right_ty_inner = left_ty.inner_with(&self.module.module.types);

        match (left_ty_inner, right_ty_inner) {
            (TypeInner::Scalar(left), TypeInner::Scalar(right)) if left == right => {
                scalar_binary_op(
                    op,
                    *left,
                    left_variable,
                    right_variable,
                    output,
                    &mut self.frame.stack,
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

#[derive(Clone, Copy, Debug)]
pub enum Value {
    F64(f64),
    F32(f32),
    F16(f16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    Bool(bool),
    Stack {
        stack_pointer: StackPointer,
        ty: Handle<Type>,
    },
}

#[derive(derive_more::Debug)]
pub struct Stack {
    #[debug("[... {} bytes]", self.stack.len())]
    stack: Vec<u8>,
    limit: usize,
}

impl Stack {
    pub fn new(limit: u32) -> Self {
        Self {
            stack: vec![],
            limit: limit.try_into().unwrap(),
        }
    }

    pub fn frame(&mut self) -> StackFrame<'_> {
        let start = self.stack.len();
        StackFrame { stack: self, start }
    }

    pub fn allocate(&mut self, size: u32) -> StackPointer {
        let top = self.stack.len();
        self.stack.resize(top + size as usize, 0);
        StackPointer(top)
    }

    pub fn read<T>(&self, pointer: StackPointer) -> &T
    where
        T: Pod,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes(&self.stack[pointer.0..][..n])
    }

    pub fn write<T>(&mut self, pointer: StackPointer, value: &T)
    where
        T: Pod,
    {
        let n = std::mem::size_of::<T>();
        self.stack[pointer.0..][..n].copy_from_slice(bytemuck::bytes_of(value));
    }

    pub fn copy(&mut self, target: StackPointer, source: StackPointer, len: u32) {
        let source_end = source + len;
        self.stack.copy_within(source.0..source_end.0, target.0);
    }

    pub fn size(&self) -> usize {
        self.stack.len()
    }

    pub fn allocated(&self) -> usize {
        self.stack.capacity()
    }
}

#[derive(Debug)]
pub struct StackFrame<'a> {
    stack: &'a mut Stack,
    start: usize,
}

impl<'a> StackFrame<'a> {
    pub fn frame(&mut self) -> StackFrame<'_> {
        self.stack.frame()
    }

    pub fn allocate(&mut self, size: u32) -> StackPointer {
        self.stack.allocate(size)
    }

    pub fn allocate_variable<'ty>(
        &mut self,
        ty: impl Into<VariableType<'ty>>,
        module: &ShaderModuleInner,
    ) -> Variable<'ty> {
        let ty = ty.into();
        let inner = ty.inner_with(module);
        let size = inner.size(module.module.to_ctx());
        let stack_address = self.allocate(size);
        Variable { ty, stack_address }
    }
}

impl<'a> Drop for StackFrame<'a> {
    fn drop(&mut self) {
        assert!(self.start <= self.stack.stack.len());
        self.stack.stack.resize(self.start, 0);
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VariableType<'a> {
    Handle(Handle<Type>),
    Inner(&'a TypeInner),
}

impl<'a> VariableType<'a> {
    pub fn inner_with(&self, module: &'a ShaderModuleInner) -> &'a TypeInner {
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
    ty: VariableType<'a>,
    stack_address: StackPointer,
}

impl<'a> Variable<'a> {
    pub fn read<'b, T>(&self, stack: &'b Stack) -> &'b T
    where
        T: Pod,
    {
        stack.read::<T>(self.stack_address)
    }

    pub fn write<T>(&self, stack: &mut Stack, value: &T)
    where
        T: Pod,
    {
        stack.write::<T>(self.stack_address, value);
    }

    pub fn copy_from(&self, source: Variable, stack: &mut Stack, module: &ShaderModuleInner) {
        let target_size = self.ty.inner_with(module).size(module.module.to_ctx());
        let source_size = source.ty.inner_with(module).size(module.module.to_ctx());
        assert_eq!(target_size, source_size);

        stack.copy(self.stack_address, source.stack_address, target_size);
    }

    pub fn cast(&self, ty: impl Into<VariableType<'a>>) -> Self {
        Self {
            ty: ty.into(),
            stack_address: self.stack_address,
        }
    }

    pub fn debug(&self, module: &'a ShaderModuleInner, stack: &'a Stack) -> VariableDebug<'a> {
        VariableDebug {
            variable: *self,
            module,
            stack,
        }
    }

    pub fn component(
        &self,
        index: usize,
        component_ty: impl Into<VariableType<'a>>,
        module: &ShaderModuleInner,
    ) -> Variable<'a> {
        let component_ty = component_ty.into();
        let offset = offset_of(self.ty, component_ty, index, module);
        Variable {
            ty: component_ty,
            stack_address: self.stack_address + offset,
        }
    }

    pub fn member(&self, member: &StructMember) -> Variable<'a> {
        Variable {
            ty: member.ty.into(),
            stack_address: self.stack_address + member.offset,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StackPointer(usize);

impl Add<u32> for StackPointer {
    type Output = Self;

    fn add(self, rhs: u32) -> Self::Output {
        Self(self.0 + rhs as usize)
    }
}

fn offset_of(
    outer_ty: VariableType,
    inner_ty: VariableType,
    index: usize,
    module: &ShaderModuleInner,
) -> u32 {
    let outer_ty = outer_ty.inner_with(module);
    match outer_ty {
        TypeInner::Vector { size, scalar } => {
            let inner_ty_layout = module.type_layout(inner_ty);
            let inner_stride = inner_ty_layout.to_stride();
            let offset = inner_stride * index as u32;
            //tracing::debug!(?outer_ty, ?inner_ty, ?inner_ty_layout, ?inner_stride,
            // ?index, ?offset);
            offset
        }
        TypeInner::Matrix {
            columns,
            rows,
            scalar,
        } => todo!(),
        TypeInner::Array { base, size, stride } => todo!(),
        TypeInner::Struct { members, span } => todo!(),
        _ => panic!("Can't produce offset into {outer_ty:?}"),
    }
}

fn scalar_binary_op(
    op: BinaryOperator,
    scalar_ty: Scalar,
    left: Variable,
    right: Variable,
    output: Variable,
    stack: &mut Stack,
) {
    use std::ops::*;

    macro_rules! binary_ops {
        (@emit_for_ty(($scalar:ident, $width:pat, $ty:ty), [$(($op:ident, $func:path)),*])) => {
            match (scalar_ty.kind, scalar_ty.width, op) {
                $(
                    (ScalarKind::$scalar, $width, BinaryOperator::$op) => {
                        let left = left.read::<$ty>(stack);
                        let right = right.read::<$ty>(stack);
                        let result = $func(left, right);
                        output.write::<$ty>(stack, &result);
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

    binary_ops!(
        [Bool@_ => u32]: [LogicalAnd => BitAnd::bitand, LogicalOr => BitOr::bitor];
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
    );
}

fn convert_scalar(
    input_variable: Variable,
    input_scalar: Scalar,
    output_variable: Variable,
    output_scalar: Scalar,
    stack: &mut Stack,
) {
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
                        let input = input_variable.read(stack);
                        let output = ($func)(*input);
                        output_variable.write(stack, &output);
                    }
                )*
                _ => panic!("Unsupported cast from {input_scalar:?} to {output_scalar:?}"),
            }
        };
    }

    convert_scalar!(
        (Sint@4) as (Uint@4) => (|x: i32| x as u32);
        (Uint@4) as (Sint@4) => (|x: u32| x as i32);
        (Sint@4) as (Float@4) => (|x: i32| x as f32);
        (Uint@4) as (Float@4) => (|x: u32| x as f32);
    );
}

fn write_literal(literal: &Literal, output: Variable, stack: &mut Stack) {
    macro_rules! write_literal {
        ($($variant:ident),*) => {
            match literal {
                $(
                    Literal::$variant(value) => {
                        output.write(stack, value);
                    }
                )*
                _ => panic!("Unsupported literal: {literal:?}"),
            }
        };
    }

    write_literal!(F32, F16, U32, I32);
}

#[derive(Clone, Copy)]
pub struct VariableDebug<'a> {
    variable: Variable<'a>,
    module: &'a ShaderModuleInner,
    stack: &'a Stack,
}

impl<'a> VariableDebug<'a> {
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
                                let input = variable.read::<$ty>(self.stack);
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

impl<'a> Debug for VariableDebug<'a> {
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
