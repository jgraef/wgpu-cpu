use std::ops::ControlFlow;

use bytemuck::Pod;
use half::f16;
use naga::{
    Block,
    Expression,
    Function,
    Handle,
    Scalar,
    ScalarKind,
    ShaderStage,
    Statement,
    Type,
    TypeInner,
    front::Typifier,
    proc::TypeResolution,
};

use crate::{
    pipeline::VertexState,
    shader::{
        ShaderModule,
        ShaderModuleInner,
    },
};

pub fn run_vertex_shader(vertex: &VertexState, instance_id: u32, vertex_id: u32) {
    let module = &vertex.module.inner.module;

    let mut virtual_machine = VirtualMachine::new(vertex.module.clone());
    virtual_machine.run_entry_point(vertex.entry_point.as_deref(), ShaderStage::Vertex);
}

#[derive(derive_more::Debug)]
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

    pub fn run_entry_point(&mut self, name: Option<&str>, shader_stage: ShaderStage) {
        let (index, entry_point) = self
            .module
            .entry_point(name, shader_stage)
            .unwrap_or_else(|| panic!("Vertex shader entry point with name {:?} not found", name));

        let mut outer_frame = self.stack.frame();

        let result_variable = entry_point.function.result.as_ref().map(|result| {
            outer_frame.allocate_variable(TypeResolution::Handle(result.ty), &self.module.inner)
        });

        RunFunction {
            module: &self.module.inner,
            function: &entry_point.function,
            typifier: &self.module.inner.expression_types[index],
            stack_frame: outer_frame.frame(),
        }
        .run(result_variable.as_ref());
    }
}

#[derive(derive_more::Debug)]
pub struct RunFunction<'a> {
    module: &'a ShaderModuleInner,
    function: &'a Function,
    typifier: &'a Typifier,
    stack_frame: StackFrame<'a>,
}

impl<'a> RunFunction<'a> {
    pub fn run(&mut self, output: Option<&Variable>) {
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
                let condition = &self.function.expressions[*condition];

                let condition_output = self.stack_frame.allocate_variable(
                    TypeResolution::Value(TypeInner::Scalar(Scalar::BOOL)),
                    &self.module,
                );

                self.evaluate(condition, &condition_output);

                if *self
                    .stack_frame
                    .stack
                    .read::<u32>(condition_output.stack_address)
                    != 0
                {
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

    pub fn evaluate(&mut self, expression: &Expression, output: &Variable) {
        tracing::debug!(?expression, "evaluate");

        match expression {
            Expression::Literal(literal) => todo!(),
            Expression::Constant(handle) => todo!(),
            Expression::Override(handle) => todo!(),
            Expression::ZeroValue(handle) => todo!(),
            Expression::Compose { ty, components } => {
                self.evaluate_compose(*ty, &components, output)
            }
            Expression::Access { base, index } => todo!(),
            Expression::AccessIndex { base, index } => todo!(),
            Expression::Splat { size, value } => todo!(),
            Expression::Swizzle {
                size,
                vector,
                pattern,
            } => todo!(),
            Expression::FunctionArgument(_) => todo!(),
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
            Expression::Binary { op, left, right } => todo!(),
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
            } => self.evaluate_as(*expr, *kind, *convert),
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
        output: &Variable,
    ) {
        for (i, component_handle) in components.into_iter().enumerate() {
            let component_expr = &self.function.expressions[*component_handle];
            let component_ty = &self.typifier[*component_handle];

            let composite_ty = &self.module.module.types[ty];

            let offset = offset_of(composite_ty, component_ty, i, self.module);

            todo!();

            //self.evaluate(expression, &variable);
        }

        todo!();
    }

    pub fn evaluate_as(
        &self,
        expression: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<u8>,
    ) {
        todo!();
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
struct Stack {
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
        let stack_pointer = self.stack.len();
        self.stack.resize(stack_pointer + size as usize, 0);
        StackPointer(stack_pointer)
    }

    pub fn read<T>(&self, pointer: StackPointer) -> &T
    where
        T: Pod,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes(&self.stack[pointer.0..][..n])
    }
}

#[derive(Debug)]
struct StackFrame<'a> {
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

    pub fn allocate_variable(
        &mut self,
        ty: TypeResolution,
        module: &ShaderModuleInner,
    ) -> Variable {
        let inner = ty.inner_with(&module.module.types);
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

#[derive(Clone, Debug)]
pub struct Variable {
    pub ty: TypeResolution,
    pub stack_address: StackPointer,
}

#[derive(Clone, Copy, Debug)]
pub struct StackPointer(usize);

fn offset_of(
    outer_ty: &Type,
    inner_ty: &TypeResolution,
    index: usize,
    module: &ShaderModuleInner,
) -> u32 {
    match &outer_ty.inner {
        TypeInner::Vector { size, scalar } => {
            tracing::debug!(?outer_ty, ?inner_ty, ?index);
            todo!();
            //let type_layout = module.layouter[]
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
