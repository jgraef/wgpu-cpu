use std::fmt::{
    Debug,
    Display,
};

use cranelift_codegen::ir::{
    Block,
    Immediate,
    InstBuilder,
    MemFlags,
    StackSlot,
    StackSlotData,
    StackSlotKind,
    Value,
    condcodes::{
        FloatCC,
        IntCC,
    },
    immediates::{
        Imm64,
        V128Imm,
    },
    types,
};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{
    FuncId,
    Linkage,
    Module,
};
use half::f16;

use crate::{
    compiler::{
        Error,
        compiler::CodegenState,
        context::Context,
        util::{
            MatrixLanes,
            alignment_log2,
            ieee16_from_f16,
            make_transpose_shuffle_mask,
        },
        value::ValueExt,
    },
    util::SparseCoArena,
};

#[derive(derive_more::Debug)]
pub struct FunctionCompiler<'source, 'compiler> {
    context: &'compiler Context<'source>,
    function: &'source naga::Function,
    function_name: String,
    typifier: &'compiler naga::front::Typifier,
    #[debug(skip)]
    function_builder: FunctionBuilder<'compiler>,
    entry_block: Block,
    emitted_expression: SparseCoArena<naga::Expression, ValueExt>,
    //local_variables: CoArena<naga::LocalVariable, Variable>,
    vector_reduce_shuffle_masks: [Immediate; 2],
    matrix_transpose_shuffle_masks: [[Immediate; 3]; 3],
}

impl<'source, 'compiler> FunctionCompiler<'source, 'compiler> {
    pub fn new(
        context: &'compiler Context<'source>,
        state: &'compiler mut CodegenState,
        typifier: &'compiler naga::front::Typifier,
        function: &'source naga::Function,
    ) -> Result<Self, Error> {
        let function_name = function
            .name
            .clone()
            .map_or_else(|| state.anonymous_function_name(), FunctionName::Named);

        // some immediates that we might use
        const VECTOR_REDUCE_SHUFFLE_MASKS: [V128Imm; 2] = [
            V128Imm([1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            V128Imm([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let vector_reduce_shuffle_masks = VECTOR_REDUCE_SHUFFLE_MASKS
            .map(|mask| state.cl_context.func.dfg.immediates.push(mask.into()));
        let matrix_transpose_shuffle_masks = {
            let vector_sizes = [2, 3, 4];
            vector_sizes.map(|columns| {
                vector_sizes.map(|rows| {
                    let mask = make_transpose_shuffle_mask(columns, rows);
                    state.cl_context.func.dfg.immediates.push(mask.into())
                })
            })
        };

        // function result
        if let Some(result) = &function.result {
            let ty = &context.source.types[result.ty];

            state
                .cl_context
                .func
                .signature
                .returns
                .push(context.abi_param(&ty.inner)?);
        }

        // function arguments
        for argument in &function.arguments {
            let ty = &context.source.types[argument.ty];

            state
                .cl_context
                .func
                .signature
                .params
                .push(context.abi_param(&ty.inner)?);
        }

        let mut function_builder =
            FunctionBuilder::new(&mut state.cl_context.func, &mut state.fb_context);

        let entry_block = function_builder.create_block();

        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);

        Ok(Self {
            context,
            function,
            function_name: function_name.to_string(),
            typifier,
            function_builder,
            entry_block,
            emitted_expression: Default::default(),
            //local_variables,
            vector_reduce_shuffle_masks,
            matrix_transpose_shuffle_masks,
        })
    }

    pub fn declare<M>(&self, module: &mut M) -> Result<FuncId, Error>
    where
        M: Module,
    {
        let function_id = module.declare_function(
            &self.function_name,
            Linkage::Local,
            &self.function_builder.func.signature,
        )?;
        Ok(function_id)
    }

    pub fn finish(mut self) {
        self.function_builder.seal_block(self.entry_block);
        self.function_builder.finalize();
    }

    pub fn expression_ty(&self, expression: naga::Handle<naga::Expression>) -> &naga::TypeInner {
        self.typifier[expression].inner_with(&self.context.source.types)
    }

    pub fn compile_block(&mut self, naga_block: &naga::Block) -> Result<(), Error> {
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
            } => todo!(),
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
            naga::Statement::Store { pointer, value } => todo!(),
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
        let return_value = expression
            .map(|expression| {
                self.compile_expression(expression)
                    .map(|value| value.as_abi(&self.context, &mut self.function_builder))
            })
            .transpose()?;
        self.function_builder.ins().return_(return_value.as_slice());
        Ok(())
    }

    pub fn compile_expression(
        &mut self,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<ValueExt, Error> {
        #![allow(unused_variables)]

        let value = if let Some(value) = self.emitted_expression.get(expression) {
            *value
        }
        else {
            let output_type = self.typifier[expression].inner_with(&self.context.source.types);
            let expression = &self.function.expressions[expression];

            match expression {
                naga::Expression::Literal(literal) => {
                    self.compile_literal(*literal, output_type)?.into()
                }
                naga::Expression::Constant(handle) => todo!(),
                naga::Expression::Override(handle) => todo!(),
                naga::Expression::ZeroValue(handle) => self.compile_zero(output_type)?,
                naga::Expression::Compose { ty, components } => {
                    self.compile_compose(*ty, components, output_type)?
                }
                naga::Expression::Access { base, index } => {
                    self.compile_access(*base, *index, output_type)?
                }
                naga::Expression::AccessIndex { base, index } => {
                    self.compile_access_index(*base, *index, output_type)?
                }
                naga::Expression::Splat { size, value } => todo!(),
                naga::Expression::Swizzle {
                    size,
                    vector,
                    pattern,
                } => todo!(),
                naga::Expression::FunctionArgument(function_argument) => {
                    self.compile_function_argument(*function_argument)?
                }
                naga::Expression::GlobalVariable(handle) => todo!(),
                naga::Expression::LocalVariable(handle) => todo!(),
                naga::Expression::Load { pointer } => todo!(),
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
                naga::Expression::Unary { op, expr } => {
                    self.compile_unary_operator(*op, *expr, output_type)?
                }
                naga::Expression::Binary { op, left, right } => {
                    self.compile_binary_operator(*op, *left, *right, output_type)?
                        .into()
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
                } => self.compile_as(*expr, *kind, *convert, output_type)?,
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

    pub fn compile_function_argument(&mut self, index: u32) -> Result<ValueExt, Error> {
        Ok(self.function_builder.block_params(self.entry_block)[index as usize].into())
    }

    pub fn compile_as(
        &mut self,
        input_expression: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<u8>,
        output_ty: &naga::TypeInner,
    ) -> Result<ValueExt, Error> {
        use naga::TypeInner::*;

        // these are not used actually
        let _ = (kind, convert);

        let input_value = self.compile_expression(input_expression)?;
        let input_ty = self.expression_ty(input_expression);

        match (input_ty, output_ty) {
            (naga::TypeInner::Scalar(input_scalar), naga::TypeInner::Scalar(output_scalar)) => {
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            (
                Vector {
                    size: input_size,
                    scalar: input_scalar,
                },
                Vector {
                    size: output_size,
                    scalar: output_scalar,
                },
            ) => {
                assert_eq!(input_size, output_size);
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            (
                Matrix {
                    columns: input_columns,
                    rows: input_rows,
                    scalar: input_scalar,
                },
                Matrix {
                    columns: output_columns,
                    rows: output_rows,
                    scalar: output_scalar,
                },
            ) => {
                assert_eq!(input_columns, output_columns);
                assert_eq!(input_rows, output_rows);
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            _ => panic!("Invalid cast from {input_ty:?} to {output_ty:?}"),
        }
    }

    pub fn compile_as_scalar(
        &mut self,
        input_value: ValueExt,
        input_scalar: naga::Scalar,
        output_scalar: naga::Scalar,
    ) -> Result<ValueExt, Error> {
        use naga::ScalarKind::*;

        // https://gpuweb.github.io/gpuweb/wgsl/#value-constructor-builtin-function

        let input_value = input_value.value();
        let output_ty = self.context.scalar_type(output_scalar)?;

        let output_value = match (input_scalar.kind, output_scalar.kind) {
            (Sint, Uint) | (Uint, Sint) => input_value,
            (Sint, Bool) | (Uint, Bool) => {
                self.function_builder
                    .ins()
                    .icmp_imm(IntCC::NotEqual, input_value, 0)
            }
            (Sint, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_sint(output_ty, input_value)
            }
            (Uint, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_uint(output_ty, input_value)
            }
            (Float, Sint) => {
                self.function_builder
                    .ins()
                    .fcvt_to_sint(output_ty, input_value)
            }
            (Float, Uint) => {
                self.function_builder
                    .ins()
                    .fcvt_to_uint(output_ty, input_value)
            }
            (Float, Bool) => {
                let zero = match input_scalar.width {
                    2 => {
                        self.function_builder
                            .ins()
                            .f16const(ieee16_from_f16(f16::NEG_ZERO))
                    }
                    4 => self.function_builder.ins().f32const(0.0),
                    _ => panic!("invalid float width: {}", input_scalar.width),
                };
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, input_value, zero)
            }
            (Float, Float) if input_scalar.width < output_scalar.width => {
                self.function_builder.ins().fpromote(output_ty, input_value)
            }
            (Float, Float) if input_scalar.width > output_scalar.width => {
                self.function_builder.ins().fdemote(output_ty, input_value)
            }
            (Bool, Sint) | (Bool, Uint) => {
                self.function_builder.ins().uextend(output_ty, input_value)
            }
            (Bool, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_uint(output_ty, input_value)
            }
            _ => panic!("unhandled type conversion: {input_scalar:?} to {output_scalar:?}"),
        };

        Ok(output_value.into())
    }

    pub fn compile_unary_operator(
        &mut self,
        operator: naga::UnaryOperator,
        input_expression: naga::Handle<naga::Expression>,
        output_ty: &naga::TypeInner,
    ) -> Result<ValueExt, Error> {
        use naga::TypeInner::*;

        let _ = output_ty;

        let input_value = self.compile_expression(input_expression)?;
        let input_ty = self.expression_ty(input_expression);

        match input_ty {
            Scalar(scalar) => self.compile_unary_scalar(operator, input_value, *scalar),
            Vector { size: _, scalar } => self.compile_unary_scalar(operator, input_value, *scalar),
            Matrix {
                columns: _,
                rows: _,
                scalar,
            } => self.compile_unary_scalar(operator, input_value, *scalar),
            _ => panic!("invalid unary operator {operator:?} on {input_ty:?}"),
        }
    }

    pub fn compile_unary_scalar(
        &mut self,
        operator: naga::UnaryOperator,
        input_value: ValueExt,
        scalar: naga::Scalar,
    ) -> Result<ValueExt, Error> {
        use naga::{
            ScalarKind::{
                Bool,
                Float,
                Sint,
                Uint,
            },
            UnaryOperator::*,
        };

        let input_value = input_value.value();

        let output_value = match (scalar.kind, operator) {
            (Sint, Negate) => self.function_builder.ins().ineg(input_value),
            (Sint, BitwiseNot) => self.function_builder.ins().bnot(input_value),
            (Uint, BitwiseNot) => self.function_builder.ins().bnot(input_value),
            (Float, Negate) => self.function_builder.ins().fneg(input_value),
            (Bool, LogicalNot) => {
                self.function_builder
                    .ins()
                    .icmp_imm(IntCC::Equal, input_value, 0)
            }
            _ => panic!("invalid unary operator {operator:?} on {scalar:?}"),
        };

        Ok(output_value.into())
    }

    pub fn compile_binary_operator(
        &mut self,
        operator: naga::BinaryOperator,
        left_expression: naga::Handle<naga::Expression>,
        right_expression: naga::Handle<naga::Expression>,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        use naga::{
            BinaryOperator::*,
            TypeInner::*,
        };

        let _ = output_ty;

        let left_value = self.compile_expression(left_expression)?.value();
        let right_value = self.compile_expression(right_expression)?.value();
        let left_ty = self.typifier[left_expression].inner_with(&self.context.source.types);
        let right_ty = self.typifier[right_expression].inner_with(&self.context.source.types);

        let output_value = match (left_ty, right_ty) {
            (Scalar(left_scalar), Scalar(right_scalar)) => {
                assert_eq!(left_scalar, right_scalar);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Vector {
                    size: left_size,
                    scalar: left_scalar,
                },
                Vector {
                    size: right_size,
                    scalar: right_scalar,
                },
            ) => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_size, right_size);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if operator == Add || operator == Subtract => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_columns, right_columns);
                assert_eq!(left_rows, right_rows);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
                Scalar(scalar),
            ) if operator == Multiply => {
                let scalar_splat = self.function_builder.ins().splat(
                    self.context
                        .matrix_type(*scalar, *matrix_columns, *matrix_rows)?,
                    left_value,
                );
                self.compile_scalar_multiply(left_value, scalar_splat, *scalar)?
            }
            (
                Scalar(scalar),
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, scalar);
                let scalar_splat = self.function_builder.ins().splat(
                    self.context
                        .matrix_type(*scalar, *matrix_columns, *matrix_rows)?,
                    left_value,
                );
                self.compile_scalar_multiply(scalar_splat, right_value, *scalar)?
            }
            (
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
                Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, vector_scalar);
                assert_eq!(matrix_rows, vector_size);

                /*
                // transpose matrix with shuffle
                let matrix =
                    self.compile_matrix_transpose(left_value, *matrix_columns, *matrix_rows)?;
                // then vector * matrix
                self.compile_vector_matrix_multiply(
                    right_value,
                    matrix,
                    *matrix_scalar,
                    *matrix_rows,
                    *matrix_columns,
                )?
                 */

                // todo: the above code is wrong. it would need a transpose at the end, right?
                // not ideal anyway. there's probably a better way
                todo!();
            }
            (
                Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, vector_scalar);
                assert_eq!(matrix_columns, vector_size);

                self.compile_vector_matrix_multiply(
                    left_value,
                    right_value,
                    *matrix_scalar,
                    *matrix_rows,
                    *matrix_columns,
                )?
            }
            (
                Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_rows, right_columns);
                todo!("matrix * matrix");
            }
            _ => panic!("invalid unary operator {operator:?} on {left_ty:?} and {right_ty:?}"),
        };

        Ok(output_value)
    }

    pub fn compile_scalar_binary_operator(
        &mut self,
        operator: naga::BinaryOperator,
        left_value: Value,
        right_value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::{
            BinaryOperator::*,
            ScalarKind::{
                Bool,
                Float,
                Sint,
                Uint,
            },
        };

        let output = match (scalar.kind, operator) {
            (Sint, Add) | (Uint, Add) => self.function_builder.ins().iadd(left_value, right_value),
            (Float, Add) => self.function_builder.ins().fadd(left_value, right_value),
            (Sint, Subtract) | (Uint, Subtract) => {
                self.function_builder.ins().isub(left_value, right_value)
            }
            (Float, Subtract) => self.function_builder.ins().fsub(left_value, right_value),
            (Sint, Multiply) | (Uint, Multiply) => {
                self.function_builder.ins().imul(left_value, right_value)
            }
            (Float, Multiply) => self.function_builder.ins().fmul(left_value, right_value),
            (Sint, Divide) => self.function_builder.ins().sdiv(left_value, right_value),
            (Uint, Divide) => self.function_builder.ins().udiv(left_value, right_value),
            (Float, Divide) => self.function_builder.ins().fdiv(left_value, right_value),
            (Sint, Modulo) => self.function_builder.ins().srem(left_value, right_value),
            (Uint, Modulo) => self.function_builder.ins().urem(left_value, right_value),
            (Float, Modulo) => {
                // https://www.w3.org/TR/WGSL/#arithmetic-expr
                // > If T is a floating point type, the result is equal to:
                // > e1 - e2 * trunc(e1 / e2).
                let x = self.function_builder.ins().fdiv(left_value, right_value);
                let x = self.function_builder.ins().trunc(x);
                let x = self.function_builder.ins().fmul(right_value, x);
                self.function_builder.ins().fsub(left_value, x)
            }
            (Uint, Equal) | (Sint, Equal) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::Equal, left_value, right_value)
            }
            (Float, Equal) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::Equal, left_value, right_value)
            }
            (Uint, NotEqual) | (Sint, NotEqual) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::NotEqual, left_value, right_value)
            }
            (Float, NotEqual) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, left_value, right_value)
            }
            (Uint, Less) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, left_value, right_value)
            }
            (Sint, Less) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, left_value, right_value)
            }
            (Float, Less) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::LessThan, left_value, right_value)
            }
            (Uint, LessEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedLessThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Sint, LessEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::SignedLessThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Float, LessEqual) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, left_value, right_value)
            }
            (Uint, Greater) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedGreaterThan,
                    left_value,
                    right_value,
                )
            }
            (Sint, Greater) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThan, left_value, right_value)
            }
            (Float, Greater) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::GreaterThan, left_value, right_value)
            }
            (Uint, GreaterEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedGreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Sint, GreaterEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::SignedGreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Float, GreaterEqual) => {
                self.function_builder.ins().fcmp(
                    FloatCC::GreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Uint, And) | (Sint, And) | (Bool, LogicalAnd) => {
                self.function_builder.ins().band(left_value, right_value)
            }
            (Uint, ExclusiveOr) | (Sint, ExclusiveOr) => {
                self.function_builder.ins().bxor(left_value, right_value)
            }
            (Uint, InclusiveOr) | (Sint, InclusiveOr) | (Bool, LogicalOr) => {
                self.function_builder.ins().bor(left_value, right_value)
            }
            (Uint, ShiftLeft) | (Sint, ShiftLeft) => {
                self.function_builder.ins().ishl(left_value, right_value)
            }
            (Uint, ShiftRight) => self.function_builder.ins().ushr(left_value, right_value),
            (Sint, ShiftRight) => self.function_builder.ins().sshr(left_value, right_value),
            _ => panic!("invalid unary operator {operator:?} on {scalar:?}"),
        };

        Ok(output.into())
    }

    pub fn compile_scalar_multiply(
        &mut self,
        left_value: Value,
        right_value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::ScalarKind::{
            Float,
            Sint,
            Uint,
        };

        let output = match scalar.kind {
            Sint | Uint => self.function_builder.ins().imul(left_value, right_value),
            Float => self.function_builder.ins().fmul(left_value, right_value),
            _ => panic!("Invalid scalar for multiplication: {scalar:?}"),
        };

        Ok(output)
    }

    pub fn compile_vector_matrix_multiply(
        &mut self,
        vector: Value,
        matrix: Value,
        scalar: naga::Scalar,
        rows: naga::VectorSize,
        columns: naga::VectorSize,
    ) -> Result<Value, Error> {
        let mut output = self.compile_vector_zero(scalar, rows)?;

        for column in 0..u8::from(columns) {
            let column_vector = self
                .function_builder
                .ins()
                .extractlane(matrix, column)
                .into();
            let column_mul = self.compile_scalar_multiply(vector, column_vector, scalar)?;
            let column_sum = self.compile_vector_sum(column_mul, scalar)?;
            output = self
                .function_builder
                .ins()
                .insertlane(output, column_sum, column);
        }

        Ok(output)
    }

    pub fn compile_vector_sum(
        &mut self,
        value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::ScalarKind::{
            Float,
            Sint,
            Uint,
        };

        let reduce_output = match scalar.kind {
            Sint | Uint => {
                self.compile_vector_reduce(value, |function_builder, accu, shuffled| {
                    Ok(function_builder.ins().iadd(accu, shuffled))
                })?
            }
            Float => {
                self.compile_vector_reduce(value, |function_builder, accu, shuffled| {
                    Ok(function_builder.ins().fadd(accu, shuffled))
                })?
            }
            _ => panic!("Invalid scalar for multiplication: {scalar:?}"),
        };

        let output = self.function_builder.ins().extractlane(reduce_output, 0);
        Ok(output)
    }

    pub fn compile_vector_reduce(
        &mut self,
        mut value: Value,
        mut op: impl FnMut(&mut FunctionBuilder, Value, Value) -> Result<Value, Error>,
    ) -> Result<Value, Error> {
        // (sum example)
        //               v0      v1      v2      v3
        //
        // shuffle:       1       0       3       2
        // add:     (v0+v1) (v1+v0) (v2+v3) (v3+v2)
        // shuffle:       2       3       0       1
        // add:     (v0+v1+v2+v2), ...

        for i in 0..1 {
            let shuffled = self.function_builder.ins().shuffle(
                value,
                value,
                self.vector_reduce_shuffle_masks[i],
            );
            value = op(&mut self.function_builder, value, shuffled)?;
        }

        Ok(value)
    }

    pub fn compile_literal(
        &mut self,
        literal: naga::Literal,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        let _ = output_ty;

        let output = match literal {
            naga::Literal::F64(value) => self.function_builder.ins().f64const(value),
            naga::Literal::F32(value) => self.function_builder.ins().f32const(value),
            naga::Literal::F16(value) => {
                self.function_builder.ins().f16const(ieee16_from_f16(value))
            }
            naga::Literal::U32(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I32, Imm64::new(value.into()))
            }
            naga::Literal::I32(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I32, Imm64::new(value.into()))
            }
            naga::Literal::U64(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I64, Imm64::new(value as i64))
            }
            naga::Literal::I64(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I64, Imm64::new(value))
            }
            naga::Literal::Bool(value) => {
                self.function_builder.ins().iconst(types::I8, value as i64)
            }
            _ => panic!("abstract literal: {literal:?}"),
        };

        Ok(output)
    }

    pub fn compile_zero(&mut self, output_type: &naga::TypeInner) -> Result<ValueExt, Error> {
        use naga::TypeInner::*;

        let output = match output_type {
            Scalar(scalar) => self.compile_scalar_zero(*scalar)?.into(),
            Vector { size, scalar } => self.compile_vector_zero(*scalar, *size)?.into(),
            Matrix {
                columns,
                rows,
                scalar,
            } => self.compile_matrix_zero(*scalar, *columns, *rows)?.into(),
            Atomic(scalar) => self.compile_scalar_zero(*scalar)?.into(),
            Array {
                base,
                size: _,
                stride: _,
            } => {
                self.compile_stack_zero(naga::proc::TypeLayout {
                    size: output_type.size(self.context.source.to_ctx()),
                    alignment: self.context.layouter[*base].alignment,
                })?
                .into()
            }
            Struct { members, span } => {
                self.compile_stack_zero(naga::proc::TypeLayout {
                    size: *span,
                    alignment: self.context.layouter[members[0].ty].alignment,
                })?
                .into()
            }
            _ => panic!("type can't be zeroed: {output_type:?}"),
        };

        Ok(output)
    }

    pub fn compile_scalar_zero(&mut self, scalar: naga::Scalar) -> Result<Value, Error> {
        use naga::ScalarKind::*;

        let output = match (scalar.kind, scalar.width) {
            (Sint, 4) | (Uint, 4) => self.function_builder.ins().iconst(types::I64, 0),
            (Float, 2) => {
                self.function_builder
                    .ins()
                    .f16const(ieee16_from_f16(f16::ZERO))
            }
            (Float, 4) => self.function_builder.ins().f32const(0.0),
            (Bool, 1) => self.function_builder.ins().iconst(types::I8, 0),
            _ => panic!("Invalid scalar type: {scalar:?}"),
        };

        Ok(output)
    }

    pub fn compile_vector_zero(
        &mut self,
        scalar: naga::Scalar,
        size: naga::VectorSize,
    ) -> Result<Value, Error> {
        // todo: use vconst
        let zero = self.compile_scalar_zero(scalar)?;
        let vector_ty = self.context.vector_type(scalar, size)?;
        let output = self.function_builder.ins().splat(vector_ty, zero);
        Ok(output)
    }

    pub fn compile_matrix_zero(
        &mut self,
        scalar: naga::Scalar,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Value, Error> {
        // todo: use vconst
        let zero = self.compile_scalar_zero(scalar)?;
        let vector_ty = self.context.matrix_type(scalar, columns, rows)?;
        let output = self.function_builder.ins().splat(vector_ty, zero);
        Ok(output)
    }

    pub fn compile_stack_zero(
        &mut self,
        type_layout: naga::proc::TypeLayout,
    ) -> Result<StackSlot, Error> {
        let alignment = alignment_log2(type_layout.alignment);
        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment,
                key: None,
            });

        let pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        self.function_builder.emit_small_memset(
            self.context.target_config,
            pointer,
            0,
            type_layout.size.into(),
            alignment,
            MemFlags::new(),
        );

        Ok(stack_slot)
    }

    pub fn compile_matrix_transpose(
        &mut self,
        value: Value,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Value, Error> {
        let output = self.function_builder.ins().shuffle(
            value,
            value,
            self.matrix_transpose_shuffle_masks[columns as u8 as usize - 1]
                [rows as u8 as usize - 1],
        );

        Ok(output)
    }

    pub fn compile_compose(
        &mut self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
        output_ty: &naga::TypeInner,
    ) -> Result<ValueExt, Error> {
        #![allow(unused_variables)]

        let _ = output_ty;
        let ty = &self.context.source.types[ty];

        let components = components
            .into_iter()
            .copied()
            .map(|expression| {
                let value = self.compile_expression(expression)?;
                let ty = self.typifier[expression].inner_with(&self.context.source.types);
                Ok::<_, Error>((value, ty))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let output = match &ty.inner {
            naga::TypeInner::Vector { size, scalar } => {
                let mut output = self.compile_vector_zero(*scalar, *size)?;

                for (i, (value, _ty)) in components.into_iter().enumerate() {
                    output = self
                        .function_builder
                        .ins()
                        .insertlane(output, value.value(), i as u8);
                }

                output.into()
            }
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar: _,
            } => {
                match &components[0].1 {
                    naga::TypeInner::Scalar(scalar) => {
                        let mut output = self.compile_matrix_zero(*scalar, *columns, *rows)?;
                        let lanes = MatrixLanes::new(*columns, *rows);

                        for (i, (value, _ty)) in components.into_iter().enumerate() {
                            output = self.function_builder.ins().insertlane(
                                output,
                                value.value(),
                                lanes.lane_flat(i as u8),
                            );
                        }

                        output.into()
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        assert_eq!(*size, *rows);

                        let mut output = self.compile_matrix_zero(*scalar, *columns, *rows)?;
                        let lanes = MatrixLanes::new(*columns, *rows);

                        lanes.for_each(|lane, column, row| {
                            if let Some((value, _ty)) = components.get(usize::from(column)) {
                                let value =
                                    self.function_builder.ins().extractlane(value.value(), row);
                                output =
                                    self.function_builder.ins().insertlane(output, value, lane);
                            }
                        });

                        output.into()
                    }
                    _ => panic!("Invalid compose: {ty:?} from {:?}", components[0].1),
                }
            }
            naga::TypeInner::Array { base, size, stride } => {
                /*let type_layout = self.context.layouter[*base];
                let array = self.compile_stack_zero(type_layout)?;

                let offset = 0;
                for (value, ty) in components {

                    self.function_builder.

                    offset += *stride;
                    todo!();
                }

                array*/
                todo!();
            }
            naga::TypeInner::Struct { members, span } => {
                todo!();
            }
            _ => panic!("Invalid compose: {ty:?}"),
        };

        Ok(output)
    }

    pub fn compile_access(
        &mut self,
        _base: naga::Handle<naga::Expression>,
        _index: naga::Handle<naga::Expression>,
        _output_ty: &naga::TypeInner,
    ) -> Result<ValueExt, Error> {
        /*let base_value = self.compile_expression(base)?;
        let base_type = self.typifier[base].inner_with(&self.context.module.types);
        let index_value = self.compile_expression(index)?;
        let invalid = || -> ! {
            panic!("Invalid index into {base_type:?}");
        };

        let output: ValueExt = match base_type {
            naga::TypeInner::Vector { size: _, scalar: _ } => {
                self.function_builder
                    .ins()
                    .extractlane(
                        base_value.value(),
                        u8::try_from(index).expect("vector index overflow"),
                    )
                    .into()
            }
            naga::TypeInner::Matrix {
                columns: _,
                rows: _,
                scalar: _,
            } => {
                todo!("access index matrix");
            }
            naga::TypeInner::Pointer { base, space: _ } => {
                let pointer_base_type = &self.context.module.types[*base];
                let offset = match &pointer_base_type.inner {
                    naga::TypeInner::Vector { size: _, scalar } => index * u32::from(scalar.width),
                    naga::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        let lanes = MatrixLanes::new(*columns, *rows);
                        lanes.column_offset(
                            index.try_into().expect("matrix column index overflow"),
                            scalar.width,
                        )
                    }
                    naga::TypeInner::Array {
                        base: _,
                        size: _,
                        stride,
                    } => index * *stride,

                    _ => invalid(),
                };
                self.function_builder
                    .ins()
                    .iadd_imm(base_value.value(), i64::from(offset))
                    .into()
            }
            naga::TypeInner::ValuePointer {
                size: Some(_),
                scalar,
                space: _,
            } => {
                let offset = index * u32::from(scalar.width);
                self.function_builder
                    .ins()
                    .iadd_imm(base_value.value(), i64::from(offset))
                    .into()
            }
            naga::TypeInner::Array {
                base: _,
                size: _,
                stride,
            } => {
                let offset = index * *stride;
                base_value.with_offset(&mut self.function_builder, offset)
            }
            _ => invalid(),
        };

        Ok(output)*/
        todo!();
    }

    pub fn compile_access_index(
        &mut self,
        base: naga::Handle<naga::Expression>,
        index: u32,
        _output_ty: &naga::TypeInner,
    ) -> Result<ValueExt, Error> {
        let base_value = self.compile_expression(base)?;
        let base_type = self.typifier[base].inner_with(&self.context.source.types);
        let invalid = || -> ! {
            panic!("Invalid index into {base_type:?}");
        };

        let output: ValueExt = match base_type {
            naga::TypeInner::Vector { size: _, scalar: _ } => {
                self.function_builder
                    .ins()
                    .extractlane(
                        base_value.value(),
                        u8::try_from(index).expect("vector index overflow"),
                    )
                    .into()
            }
            naga::TypeInner::Matrix {
                columns: _,
                rows: _,
                scalar: _,
            } => {
                todo!("access index matrix");
            }
            naga::TypeInner::Pointer { base, space: _ } => {
                let pointer_base_type = &self.context.source.types[*base];
                let offset = match &pointer_base_type.inner {
                    naga::TypeInner::Vector { size: _, scalar } => index * u32::from(scalar.width),
                    naga::TypeInner::Matrix {
                        columns,
                        rows,
                        scalar,
                    } => {
                        let lanes = MatrixLanes::new(*columns, *rows);
                        lanes.column_offset(
                            index.try_into().expect("matrix column index overflow"),
                            scalar.width,
                        )
                    }
                    naga::TypeInner::Array {
                        base: _,
                        size: _,
                        stride,
                    } => index * *stride,
                    naga::TypeInner::Struct { members, span: _ } => {
                        members[usize::try_from(index).expect("struct index overflow")].offset
                    }
                    _ => invalid(),
                };
                self.function_builder
                    .ins()
                    .iadd_imm(base_value.value(), i64::from(offset))
                    .into()
            }
            naga::TypeInner::ValuePointer {
                size: Some(_),
                scalar,
                space: _,
            } => {
                let offset = index * u32::from(scalar.width);
                self.function_builder
                    .ins()
                    .iadd_imm(base_value.value(), i64::from(offset))
                    .into()
            }
            naga::TypeInner::Array {
                base: _,
                size: _,
                stride,
            } => {
                let offset = index * *stride;
                base_value.with_offset(&mut self.function_builder, offset)
            }
            naga::TypeInner::Struct { members, span: _ } => {
                let offset = members[usize::try_from(index).expect("struct index overflow")].offset;
                base_value.with_offset(&mut self.function_builder, offset)
            }
            _ => invalid(),
        };

        Ok(output)
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
