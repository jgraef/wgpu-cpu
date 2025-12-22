use std::{
    fmt::{
        Debug,
        Display,
    },
    ops::Range,
};

use cranelift_codegen::ir::{
    self,
    AbiParam,
    InstBuilder as _,
};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{
    FuncId,
    Linkage,
    Module,
};

use crate::{
    compiler::{
        Error,
        compiler::{
            Context,
            State,
        },
        expression::{
            CompileAdd,
            CompileAs,
            CompileBitAnd,
            CompileBitNot,
            CompileBitOr,
            CompileBitXor,
            CompileCompose,
            CompileDiv,
            CompileEq,
            CompileGe,
            CompileGt,
            CompileLe,
            CompileLiteral,
            CompileLogAnd,
            CompileLogNot,
            CompileLogOr,
            CompileLt,
            CompileMod,
            CompileMul,
            CompileNeg,
            CompileNeq,
            CompileShl,
            CompileShr,
            CompileSub,
            CompileZero,
        },
        simd::SimdImmediates,
        types::{
            AsIrTypes,
            CastTo,
            InvalidType,
            MatrixType,
            PointerType,
            ScalarType,
            Type,
            VectorType,
        },
        util::alignment_log2,
        value::{
            AsIrValue,
            AsIrValues,
            FromIrValues,
            PointerValue,
            ScalarValue,
            StackLocation,
            Store,
            Value,
        },
    },
    util::{
        CoArena,
        SparseCoArena,
        typifier_from_function,
    },
};

#[derive(Debug)]
pub struct FunctionContext<'source, 'compiler> {
    pub compiler_context: &'compiler Context<'source>,
    pub function: &'source naga::Function,
    pub function_name: String,
    pub typifier: naga::front::Typifier,
    pub entry_block: ir::Block,
    pub function_arguments: Vec<FunctionArgument>,
    pub local_variables: CoArena<naga::LocalVariable, LocalVariable<'source>>,
    pub simd_immediates: SimdImmediates,
}

impl<'source, 'compiler> FunctionContext<'source, 'compiler> {
    pub fn new(
        compiler_context: &'compiler Context<'source>,
        function: &'source naga::Function,
        function_name: String,
        entry_block: ir::Block,
        function_arguments: Vec<FunctionArgument>,
        local_variables: CoArena<naga::LocalVariable, LocalVariable<'source>>,
        simd_immediates: SimdImmediates,
    ) -> Self {
        let typifier = typifier_from_function(&compiler_context.source, function);

        Self {
            compiler_context,
            function,
            function_name,
            typifier,
            entry_block,
            function_arguments,
            local_variables,
            simd_immediates,
        }
    }

    pub fn expression_type(
        &self,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<Type, InvalidType> {
        let type_resolution = &self.typifier[expression];
        let output = match type_resolution {
            naga::proc::TypeResolution::Handle(handle) => self.compiler_context.types[*handle],
            naga::proc::TypeResolution::Value(type_inner) => {
                match type_inner {
                    naga::TypeInner::Scalar(scalar) => ScalarType::from_naga(*scalar)?.into(),
                    naga::TypeInner::Vector { size, scalar } => {
                        VectorType::from_naga(*scalar, *size)?.into()
                    }
                    naga::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => MatrixType::from_naga(*scalar, *columns, *rows)?.into(),
                    naga::TypeInner::Pointer { base, space } => {
                        PointerType::from_naga(*base, *space)?.into()
                    }
                    naga::TypeInner::ValuePointer {
                        size,
                        scalar,
                        space,
                    } => PointerType::from_naga_value(*scalar, *size, *space)?.into(),
                    _ => unreachable!("Invalid inner type returned by typifier: {type_inner:?}"),
                }
            }
        };
        Ok(output)
    }
}

#[derive(Clone, Debug)]
pub struct FunctionArgument {
    pub block_inputs: Range<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalVariable<'source> {
    pub name: Option<&'source str>,
    pub ty: Type,
    pub pointer_type: PointerType,
    pub stack_slot: ir::StackSlot,
}

#[derive(derive_more::Debug)]
pub struct FunctionCompiler<'source, 'compiler> {
    pub context: FunctionContext<'source, 'compiler>,

    #[debug(skip)]
    pub function_builder: FunctionBuilder<'compiler>,

    emitted_expression: SparseCoArena<naga::Expression, Value>,
}

impl<'source, 'compiler> FunctionCompiler<'source, 'compiler> {
    pub fn new(
        compiler_context: &'compiler Context<'source>,
        state: &'compiler mut State,
        function: &'source naga::Function,
    ) -> Result<Self, Error> {
        let function_name = function
            .name
            .clone()
            .map_or_else(|| state.anonymous_function_name(), FunctionName::Named);

        let simd_immediates = compiler_context.simd_context.simd_immediates(state);

        // function result
        if let Some(result) = &function.result {
            state.cl_context.func.signature.returns.extend(
                compiler_context.types[result.ty]
                    .as_ir_types(compiler_context)
                    .map(AbiParam::new),
            );
        }

        // function arguments
        let mut function_arguments = Vec::with_capacity(function.arguments.len());
        assert!(state.cl_context.func.signature.params.is_empty());
        for argument in &function.arguments {
            let start = state.cl_context.func.signature.params.len();

            state.cl_context.func.signature.params.extend(
                compiler_context.types[argument.ty]
                    .as_ir_types(compiler_context)
                    .map(AbiParam::new),
            );

            let end = state.cl_context.func.signature.params.len();
            function_arguments.push(FunctionArgument {
                block_inputs: start..end,
            })
        }

        let mut function_builder =
            FunctionBuilder::new(&mut state.cl_context.func, &mut state.fb_context);

        let entry_block = function_builder.create_block();
        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);

        // local variables
        let local_variables =
            CoArena::try_from_arena(&function.local_variables, |handle, variable| {
                let type_layout = compiler_context.layouter[variable.ty];
                let stack_slot_key = ir::StackSlotKey::new(handle.index().try_into().unwrap());

                let stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
                    kind: ir::StackSlotKind::ExplicitSlot,
                    size: type_layout.size,
                    align_shift: alignment_log2(type_layout.alignment),
                    key: Some(stack_slot_key),
                });

                let pointer_type =
                    PointerType::from_naga(variable.ty, naga::AddressSpace::Function)?;

                Ok::<_, Error>(LocalVariable {
                    name: variable.name.as_deref(),
                    ty: compiler_context.types[variable.ty],
                    pointer_type,
                    stack_slot,
                })
            })?;

        Ok(Self {
            context: FunctionContext::new(
                compiler_context,
                function,
                function_name.to_string(),
                entry_block,
                function_arguments,
                local_variables,
                simd_immediates,
            ),
            function_builder,
            emitted_expression: Default::default(),
        })
    }

    pub fn declare<M>(&self, module: &mut M) -> Result<FuncId, Error>
    where
        M: Module,
    {
        let function_id = module.declare_function(
            &self.context.function_name,
            Linkage::Local,
            &self.function_builder.func.signature,
        )?;
        Ok(function_id)
    }

    pub fn initialize_local_variables(&mut self) -> Result<(), Error> {
        for (handle, variable) in self.context.function.local_variables.iter() {
            if let Some(init) = variable.init {
                let variable = self.context.local_variables[handle];
                let value = self.compile_expression(init)?;
                value.store(
                    self.context.compiler_context,
                    &mut self.function_builder,
                    StackLocation::from(variable.stack_slot),
                )?;
            }
        }

        Ok(())
    }

    pub fn finish(mut self) {
        self.function_builder.seal_block(self.context.entry_block);
        self.function_builder.finalize();
    }

    pub fn compile_block(&mut self, naga_block: &naga::ir::Block) -> Result<(), Error> {
        for statement in naga_block {
            self.compile_statement(statement)?;
        }
        Ok(())
    }

    pub fn compile_statement(&mut self, statement: &naga::Statement) -> Result<(), Error> {
        #![allow(unused_variables)]

        match statement {
            naga::Statement::Emit(range) => {
                self.compile_emit(range.clone())?;
            }
            naga::Statement::Block(naga_block) => {
                // I don't think we actually have to emit a block in cranelift IR.
                // We only have to emit blocks, if we want to jump to them from multiple other
                // blocks, or as an entry point for functions.

                /*let current_cl_block = self.function_builder.current_block().unwrap();
                let new_cl_block = self.function_builder.create_block();

                self.function_builder.switch_to_block(new_cl_block);
                self.compile_block(naga_block);
                self.function_builder.switch_to_block(current_cl_block);

                self.function_builder.ins().jump(new_cl_block, []);
                self.function_builder.seal_block(new_cl_block);*/

                self.compile_block(naga_block)?;
            }
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => self.compile_if(*condition, accept, reject)?,
            naga::Statement::Switch { selector, cases } => todo!(),
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => todo!(),
            naga::Statement::Break => todo!(),
            naga::Statement::Continue => todo!(),
            naga::Statement::Return { value } => {
                self.compile_return(*value)?;
            }
            naga::Statement::Kill => todo!(),
            naga::Statement::ControlBarrier(barrier) => todo!(),
            naga::Statement::MemoryBarrier(barrier) => todo!(),
            naga::Statement::Store { pointer, value } => self.compile_store(*pointer, *value)?,
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => todo!(),
            naga::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => todo!(),
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => todo!(),
            naga::Statement::WorkGroupUniformLoad { pointer, result } => todo!(),
            naga::Statement::Call {
                function,
                arguments,
                result,
            } => todo!(),
            naga::Statement::RayQuery { query, fun } => todo!(),
            naga::Statement::SubgroupBallot { result, predicate } => todo!(),
            naga::Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => todo!(),
            naga::Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => todo!(),
        }

        Ok(())
    }

    pub fn compile_emit(
        &mut self,
        expressions: naga::Range<naga::Expression>,
    ) -> Result<(), Error> {
        for expression in expressions {
            self.compile_expression(expression)?;
        }
        Ok(())
    }

    pub fn compile_return(
        &mut self,
        expression: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let mut return_values = vec![];

        if let Some(expression) = expression {
            let value = self.compile_expression(expression)?;
            return_values.extend(value.as_ir_values());
        }

        self.function_builder.ins().return_(&return_values);

        // fixme: return ControlFlow to stop compiling this block. this is a bit tricky
        // because we also return Results for now we'll just switch to a new
        // block for the rest. this block will not be jumped to, but we still do the
        // work compiling it.
        let void_block = self.function_builder.create_block();
        self.function_builder.seal_block(void_block);
        self.function_builder.switch_to_block(void_block);

        Ok(())
    }

    pub fn compile_expression(
        &mut self,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        #![allow(unused_variables)]

        let value = if let Some(value) = self.emitted_expression.get(expression) {
            value.clone()
        }
        else {
            let output_type = self.context.expression_type(expression)?;
            let expression = &self.context.function.expressions[expression];

            match expression {
                naga::Expression::Literal(literal) => self.compile_literal(*literal)?,
                naga::Expression::Constant(handle) => todo!(),
                naga::Expression::Override(handle) => todo!(),
                naga::Expression::ZeroValue(handle) => self.compile_zero(*handle)?,
                naga::Expression::Compose { ty, components } => {
                    self.compile_compose(*ty, components)?
                }
                naga::Expression::Access { base, index } => {
                    //self.compile_access(*base, *index, output_type)?
                    todo!();
                }
                naga::Expression::AccessIndex { base, index } => {
                    //self.compile_access_index(*base, *index, output_type)?
                    todo!();
                }
                naga::Expression::Splat { size, value } => todo!(),
                naga::Expression::Swizzle {
                    size,
                    vector,
                    pattern,
                } => todo!(),
                naga::Expression::FunctionArgument(function_argument) => {
                    self.compile_function_argument(*function_argument, output_type)?
                }
                naga::Expression::GlobalVariable(handle) => todo!(),
                naga::Expression::LocalVariable(handle) => self.compile_local_variable(*handle)?,
                naga::Expression::Load { pointer } => self.compile_load(*pointer)?,
                naga::Expression::ImageSample {
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
                naga::Expression::ImageLoad {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                } => todo!(),
                naga::Expression::ImageQuery { image, query } => todo!(),
                naga::Expression::Unary { op, expr } => self.compile_unary_operator(*op, *expr)?,
                naga::Expression::Binary { op, left, right } => {
                    self.compile_binary_operator(*op, *left, *right)?
                }
                naga::Expression::Select {
                    condition,
                    accept,
                    reject,
                } => todo!(),
                naga::Expression::Derivative { axis, ctrl, expr } => todo!(),
                naga::Expression::Relational { fun, argument } => todo!(),
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                } => todo!(),
                naga::Expression::As {
                    expr,
                    kind,
                    convert,
                } => self.compile_as(*expr, *kind, *convert)?,
                naga::Expression::CallResult(handle) => todo!(),
                naga::Expression::AtomicResult { ty, comparison } => todo!(),
                naga::Expression::WorkGroupUniformLoadResult { ty } => todo!(),
                naga::Expression::ArrayLength(handle) => todo!(),
                naga::Expression::RayQueryVertexPositions { query, committed } => todo!(),
                naga::Expression::RayQueryProceedResult => todo!(),
                naga::Expression::RayQueryGetIntersection { query, committed } => todo!(),
                naga::Expression::SubgroupBallotResult => todo!(),
                naga::Expression::SubgroupOperationResult { ty } => todo!(),
            }
        };

        Ok(value)
    }

    pub fn compile_local_variable(
        &mut self,
        variable: naga::Handle<naga::LocalVariable>,
    ) -> Result<Value, Error> {
        let local_variable = self.context.local_variables[variable];

        let value =
            PointerValue::from_stack_slot(local_variable.pointer_type, local_variable.stack_slot);

        Ok(value.into())
    }

    pub fn compile_function_argument(
        &mut self,
        index: u32,
        output_type: Type,
    ) -> Result<Value, Error> {
        let argument = &self.context.function_arguments[index as usize];
        let block_params = self.function_builder.block_params(self.context.entry_block);
        let block_params = block_params[argument.block_inputs.clone()].iter().copied();

        Ok(Value::from_ir_values_iter(
            &self.context.compiler_context,
            output_type,
            block_params,
        ))
    }

    pub fn compile_literal(&mut self, literal: naga::Literal) -> Result<Value, Error> {
        Value::compile_literal(literal, self)
    }

    pub fn compile_zero(&mut self, ty: naga::Handle<naga::Type>) -> Result<Value, Error> {
        let ty = self.context.compiler_context.types[ty];
        Value::compile_zero(ty, self)
    }

    pub fn compile_load(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let pointer: PointerValue = self.compile_expression(pointer)?.try_into()?;
        pointer.deref_load(self.context.compiler_context, &mut self.function_builder)
    }

    pub fn compile_store(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let pointer: PointerValue = self.compile_expression(pointer)?.try_into()?;
        let value: Value = self.compile_expression(expression)?;
        pointer.deref_store(
            self.context.compiler_context,
            &mut self.function_builder,
            &value,
        )
    }

    pub fn compile_compose(
        &mut self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
    ) -> Result<Value, Error> {
        let ty = self.context.compiler_context.types[ty];
        let components = components
            .into_iter()
            .map(|expression| self.compile_expression(*expression))
            .collect::<Result<Vec<_>, Error>>()?;

        Value::compile_compose(ty, components, self)
    }

    pub fn compile_as(
        &mut self,
        input_expression: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<u8>,
    ) -> Result<Value, Error> {
        let cast_to = CastTo::from_naga(kind, convert);
        let input_value = self.compile_expression(input_expression)?;
        input_value.compile_as(cast_to, self)
    }

    pub fn compile_unary_operator(
        &mut self,
        operator: naga::UnaryOperator,
        input_expression: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        use naga::UnaryOperator::*;
        let input_value = self.compile_expression(input_expression)?;

        let output = match operator {
            Negate => input_value.compile_neg(self)?.into(),
            LogicalNot => input_value.compile_log_not(self)?.into(),
            BitwiseNot => input_value.compile_bit_not(self)?.into(),
        };

        Ok(output)
    }

    pub fn compile_binary_operator(
        &mut self,
        operator: naga::BinaryOperator,
        left_expression: naga::Handle<naga::Expression>,
        right_expression: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        use naga::BinaryOperator::*;
        let left_value = self.compile_expression(left_expression)?;
        let right_value = self.compile_expression(right_expression)?;

        let output = match operator {
            Add => left_value.compile_add(&right_value, self)?.into(),
            Subtract => left_value.compile_sub(&right_value, self)?.into(),
            Multiply => left_value.compile_mul(&right_value, self)?.into(),
            Divide => left_value.compile_div(&right_value, self)?.into(),
            Modulo => left_value.compile_mod(&right_value, self)?.into(),
            Equal => left_value.compile_eq(&right_value, self)?.into(),
            NotEqual => left_value.compile_neq(&right_value, self)?.into(),
            Less => left_value.compile_lt(&right_value, self)?.into(),
            LessEqual => left_value.compile_le(&right_value, self)?.into(),
            Greater => left_value.compile_gt(&right_value, self)?.into(),
            GreaterEqual => left_value.compile_ge(&right_value, self)?.into(),
            And => left_value.compile_bit_and(&right_value, self)?.into(),
            ExclusiveOr => left_value.compile_bit_xor(&right_value, self)?.into(),
            InclusiveOr => left_value.compile_bit_or(&right_value, self)?.into(),
            LogicalAnd => left_value.compile_log_and(&right_value, self)?.into(),
            LogicalOr => left_value.compile_log_or(&right_value, self)?.into(),
            ShiftLeft => left_value.compile_shl(&right_value, self)?.into(),
            ShiftRight => left_value.compile_shr(&right_value, self)?.into(),
        };

        Ok(output)
    }

    pub fn compile_if(
        &mut self,
        condition: naga::Handle<naga::Expression>,
        accept: &naga::Block,
        reject: &naga::Block,
    ) -> Result<(), Error> {
        let condition_value: ScalarValue = self.compile_expression(condition)?.try_into()?;
        let condition_value = condition_value.as_ir_value();

        let accept_block = self.function_builder.create_block();
        let reject_block = self.function_builder.create_block();
        let continue_block = self.function_builder.create_block();

        self.function_builder
            .ins()
            .brif(condition_value, accept_block, [], reject_block, []);

        self.function_builder.seal_block(accept_block);
        self.function_builder.seal_block(reject_block);

        self.function_builder.switch_to_block(accept_block);
        self.compile_block(accept)?;
        self.function_builder.ins().jump(continue_block, []);

        self.function_builder.switch_to_block(reject_block);
        self.compile_block(reject)?;
        self.function_builder.ins().jump(continue_block, []);

        self.function_builder.seal_block(continue_block);
        self.function_builder.switch_to_block(continue_block);

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum FunctionName {
    Anonymous(usize),
    Named(String),
}

impl Display for FunctionName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionName::Anonymous(id) => {
                write!(f, "__naga_interpreter_anonymous_{id}")
            }
            FunctionName::Named(name) => {
                write!(f, "{name}")
            }
        }
    }
}
