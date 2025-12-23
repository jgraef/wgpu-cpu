#![allow(unused_variables)]

mod access;
mod array_length;
mod r#as;
mod atomic_result;
mod binary;
mod call_result;
mod compose;
mod constant;
mod derivative;
mod function_argument;
mod image;
mod literal;
mod load;
mod math;
mod r#override;
mod ray_query;
mod relational;
mod select;
mod splat;
mod subgroup;
mod swizzle;
mod unary;
mod variable;
mod workgroup;
mod zero_value;

pub use access::*;
pub use array_length::*;
pub use r#as::*;
pub use atomic_result::*;
pub use binary::*;
pub use call_result::*;
pub use compose::*;
pub use constant::*;
use cranelift_codegen::{
    entity::EntityRef,
    ir,
};
pub use derivative::*;
pub use function_argument::*;
pub use image::*;
pub use literal::*;
pub use load::*;
pub use math::*;
pub use r#override::*;
pub use ray_query::*;
pub use relational::*;
pub use select::*;
pub use splat::*;
pub use subgroup::*;
pub use swizzle::*;
pub use unary::*;
pub use variable::*;
pub use workgroup::*;
pub use zero_value::*;

use crate::compiler::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    function::FunctionCompiler,
    types::CastTo,
    util::math_args_to_array_vec,
    value::{
        AsIrValues,
        Value,
    },
};

macro_rules! define_expression {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Debug)]
        pub enum Expression {
            $($variant($ty),)*
        }

        impl CompileExpression for Expression {
            fn compile_expression(
                &self,
                compiler: &mut FunctionCompiler,
            ) -> Result<Value, Error> {
                let value = match self {
                    $(Self::$variant(expression) => CompileExpression::compile_expression(expression, compiler)?.into(),)*
                };
                Ok(value)
            }
        }

        $(
            impl From<$ty> for Expression {
                fn from(expression: $ty) -> Self {
                    Self::$variant(expression)
                }
            }

            impl TryFrom<Expression> for $ty {
                type Error = ();

                fn try_from(expression: Expression) -> Result<Self, Self::Error> {
                    match expression {
                        Expression::$variant(expression) => Ok(expression),
                        _ => Err(())
                    }
                }
            }
        )*
    };
}

define_expression!(
    Literal(LiteralExpression),
    Constant(ConstantUseExpression),
    Override(OverrideExpression),
    ZeroValue(ZeroValueExpression),
    Compose(ComposeExpression),
    Access(AccessExpression),
    AccessIndex(AccessIndexExpression),
    Splat(SplatExpression),
    Swizzle(SwizzleExpression),
    FunctionArgument(FunctionArgumentExpression),
    GlobalVariable(GlobalVariableExpression),
    LocalVariable(LocalVariableExpression),
    Load(LoadExpression),
    ImageSample(ImageSampleExpression),
    ImageLoad(ImageLoadExpression),
    ImageQuery(ImageQueryExpression),
    Unary(UnaryExpression),
    Binary(BinaryExpression),
    Select(SelectExpression),
    Derivative(DerivativeExpression),
    Relational(RelationalExpression),
    Math(MathExpression),
    As(AsExpression),
    CallResult(CallResultExpression),
    AtomicResult(AtomicResultExpression),
    WorkGroupUniformLoadResult(WorkGroupUniformLoadResultExpression),
    ArrayLength(ArrayLengthExpression),
    RayQueryVertexPositions(RayQueryVertexPositionsExpression),
    RayQueryProceedResult(RayQueryProceedResultExpression),
    RayQueryGetIntersection(RayQueryGetIntersectionExpression),
    SubgroupBallotResult(SubgroupBallotResultExpression),
    SubgroupOperationResult(SubgroupOperationResultExpression),
);

impl From<naga::Expression> for Expression {
    fn from(expression: naga::Expression) -> Self {
        use naga::Expression::*;

        match expression {
            Literal(literal) => Self::Literal(LiteralExpression { literal }),
            Constant(handle) => Self::Constant(ConstantUseExpression { handle }),
            Override(handle) => Self::Override(OverrideExpression { handle }),
            ZeroValue(handle) => Self::ZeroValue(ZeroValueExpression { ty: handle }),
            Compose { ty, components } => Self::Compose(ComposeExpression { ty, components }),
            Access { base, index } => Self::Access(AccessExpression { base, index }),
            AccessIndex { base, index } => Self::AccessIndex(AccessIndexExpression { base, index }),
            Splat { size, value } => Self::Splat(SplatExpression { size, value }),
            Swizzle {
                size,
                vector,
                pattern,
            } => {
                Self::Swizzle(SwizzleExpression {
                    size,
                    vector,
                    pattern,
                })
            }
            FunctionArgument(index) => Self::FunctionArgument(FunctionArgumentExpression { index }),
            GlobalVariable(handle) => Self::GlobalVariable(GlobalVariableExpression { handle }),
            LocalVariable(handle) => Self::LocalVariable(LocalVariableExpression { handle }),
            Load { pointer } => Self::Load(LoadExpression { pointer }),
            ImageSample {
                image,
                sampler,
                gather,
                coordinate,
                array_index,
                offset,
                level,
                depth_ref,
                clamp_to_edge,
            } => {
                Self::ImageSample(ImageSampleExpression {
                    image,
                    sampler,
                    gather,
                    coordinate,
                    array_index,
                    offset,
                    level,
                    depth_ref,
                    clamp_to_edge,
                })
            }
            ImageLoad {
                image,
                coordinate,
                array_index,
                sample,
                level,
            } => {
                Self::ImageLoad(ImageLoadExpression {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                })
            }
            ImageQuery { image, query } => Self::ImageQuery(ImageQueryExpression { image, query }),
            Unary { op, expr } => {
                Self::Unary(UnaryExpression {
                    operator: op,
                    operand: expr,
                })
            }
            Binary { op, left, right } => {
                Self::Binary(BinaryExpression {
                    operator: op,
                    left_operand: left,
                    right_operand: right,
                })
            }
            Select {
                condition,
                accept,
                reject,
            } => {
                Self::Select(SelectExpression {
                    condition,
                    accept,
                    reject,
                })
            }
            Derivative { axis, ctrl, expr } => {
                Self::Derivative(DerivativeExpression {
                    axis,
                    control: ctrl,
                    expression: expr,
                })
            }
            Relational { fun, argument } => {
                Self::Relational(RelationalExpression {
                    function: fun,
                    argument,
                })
            }
            Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                Self::Math(MathExpression {
                    function: fun,
                    arguments: math_args_to_array_vec(arg, arg1, arg2, arg3),
                })
            }
            As {
                expr,
                kind,
                convert,
            } => {
                Self::As(AsExpression {
                    expression: expr,
                    target: CastTo::from_naga(kind, convert),
                })
            }
            CallResult(handle) => Self::CallResult(CallResultExpression { function: handle }),
            AtomicResult { ty, comparison } => {
                Self::AtomicResult(AtomicResultExpression { ty, comparison })
            }
            WorkGroupUniformLoadResult { ty } => {
                Self::WorkGroupUniformLoadResult(WorkGroupUniformLoadResultExpression { ty })
            }
            ArrayLength(handle) => Self::ArrayLength(ArrayLengthExpression { array: handle }),
            RayQueryVertexPositions { query, committed } => {
                Self::RayQueryVertexPositions(RayQueryVertexPositionsExpression {
                    query,
                    committed,
                })
            }
            RayQueryProceedResult => {
                Self::RayQueryProceedResult(RayQueryProceedResultExpression {})
            }
            RayQueryGetIntersection { query, committed } => {
                Self::RayQueryGetIntersection(RayQueryGetIntersectionExpression {
                    query,
                    committed,
                })
            }
            SubgroupBallotResult => Self::SubgroupBallotResult(SubgroupBallotResultExpression {}),
            SubgroupOperationResult { ty } => {
                Self::SubgroupOperationResult(SubgroupOperationResultExpression { ty })
            }
        }
    }
}

pub trait CompileExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error>;
}

impl CompileExpression for naga::Handle<naga::Expression> {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let value = if let Some(value) = compiler.emitted_expression.get(*self) {
            value.clone()
        }
        else {
            let expression = &compiler.function.expressions[*self];
            // todo: do that here or in the constructor?
            let expression: Expression = expression.clone().into();

            let span = &compiler.function.expressions.get_span(*self);
            compiler.set_source_span(*span);

            let value = expression.compile_expression(compiler)?;

            if compiler.context.config.collect_debug_info
                && compiler.function.named_expressions.contains_key(&*self)
            {
                value.as_ir_values().for_each(|ir_value| {
                    compiler
                        .function_builder
                        .set_val_label(ir_value, ir::ValueLabel::new(self.index()));
                });
            }

            compiler.emitted_expression.insert(*self, value.clone());

            value
        };

        Ok(value)
    }
}

macro_rules! define_constant_expression {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Debug)]
        pub enum ConstantExpression {
            $($variant($ty),)*
        }

        impl EvaluateExpression for ConstantExpression {
            fn evaluate_expression(
                &self,
                context: &Context,
            ) -> Result<ConstantValue, Error> {
                let value = match self {
                    $(Self::$variant(expression) => EvaluateExpression::evaluate_expression(expression, context)?.into(),)*
                };
                Ok(value)
            }
        }

        $(
            impl From<$ty> for ConstantExpression {
                fn from(expression: $ty) -> Self {
                    Self::$variant(expression)
                }
            }

            impl TryFrom<ConstantExpression> for $ty {
                type Error = ();

                fn try_from(expression: ConstantExpression) -> Result<Self, Self::Error> {
                    match expression {
                        ConstantExpression::$variant(expression) => Ok(expression),
                        _ => Err(())
                    }
                }
            }
        )*
    };
}

define_constant_expression!(
    Literal(LiteralExpression),
    Constant(ConstantUseExpression),
    ZeroValue(ZeroValueExpression),
    Compose(ComposeExpression),
    Access(AccessExpression),
    AccessIndex(AccessIndexExpression),
    Splat(SplatExpression),
    Swizzle(SwizzleExpression),
    Unary(UnaryExpression),
    Binary(BinaryExpression),
    Select(SelectExpression),
    Relational(RelationalExpression),
    Math(MathExpression),
    As(AsExpression),
);

#[derive(Clone, Debug, thiserror::Error)]
#[error("Expression is not constant: {expression:?}")]
pub struct ExpressionNotConstant {
    pub expression: naga::Expression,
}

impl TryFrom<naga::Expression> for ConstantExpression {
    type Error = ExpressionNotConstant;

    fn try_from(expression: naga::Expression) -> Result<Self, ExpressionNotConstant> {
        use naga::Expression::*;

        let expression = match expression {
            Literal(literal) => Self::Literal(LiteralExpression { literal }),
            Constant(handle) => Self::Constant(ConstantUseExpression { handle }),
            ZeroValue(handle) => Self::ZeroValue(ZeroValueExpression { ty: handle }),
            Compose { ty, components } => Self::Compose(ComposeExpression { ty, components }),
            Access { base, index } => Self::Access(AccessExpression { base, index }),
            AccessIndex { base, index } => Self::AccessIndex(AccessIndexExpression { base, index }),
            Splat { size, value } => Self::Splat(SplatExpression { size, value }),
            Swizzle {
                size,
                vector,
                pattern,
            } => {
                Self::Swizzle(SwizzleExpression {
                    size,
                    vector,
                    pattern,
                })
            }
            Unary { op, expr } => {
                Self::Unary(UnaryExpression {
                    operator: op,
                    operand: expr,
                })
            }
            Binary { op, left, right } => {
                Self::Binary(BinaryExpression {
                    operator: op,
                    left_operand: left,
                    right_operand: right,
                })
            }
            Select {
                condition,
                accept,
                reject,
            } => {
                Self::Select(SelectExpression {
                    condition,
                    accept,
                    reject,
                })
            }
            Relational { fun, argument } => {
                Self::Relational(RelationalExpression {
                    function: fun,
                    argument,
                })
            }
            Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => {
                Self::Math(MathExpression {
                    function: fun,
                    arguments: math_args_to_array_vec(arg, arg1, arg2, arg3),
                })
            }
            As {
                expr,
                kind,
                convert,
            } => {
                Self::As(AsExpression {
                    expression: expr,
                    target: CastTo::from_naga(kind, convert),
                })
            }
            _ => return Err(ExpressionNotConstant { expression }),
        };

        Ok(expression)
    }
}

pub trait EvaluateExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error>;
}

impl EvaluateExpression for naga::Handle<naga::Expression> {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        let expression = &context.source.global_expressions[*self];
        let expression: ConstantExpression = expression.clone().try_into()?;
        expression.evaluate_expression(context)
    }
}

/*
fn compile_vector_reduce(
    function_compiler: &mut FunctionCompiler,
    mut value: ir::Value,
    mut op: impl FnMut(&mut FunctionBuilder, ir::Value, ir::Value) -> Result<ir::Value, Error>,
) -> Result<ir::Value, Error> {
    // (sum example)
    //               v0      v1      v2      v3
    //
    // shuffle:       1       0       3       2
    // add:     (v0+v1) (v1+v0) (v2+v3) (v3+v2)
    // shuffle:       2       3       0       1
    // add:     (v0+v1+v2+v2), ...

    for i in 0..1 {
        let shuffled = function_compiler.function_builder.ins().shuffle(
            value,
            value,
            function_compiler
                .context
                .simd_immediates
                .vector_reduce_shuffle_masks[i],
        );
        value = op(&mut function_compiler.function_builder, value, shuffled)?;
    }

    Ok(value)
}

fn compile_matrix_transpose(
    function_compiler: &mut FunctionCompiler,
    value: ir::Value,
    columns: naga::VectorSize,
    rows: naga::VectorSize,
) -> Result<ir::Value, Error> {
    let output = function_compiler.function_builder.ins().shuffle(
        value,
        value,
        function_compiler
            .context
            .simd_immediates
            .matrix_transpose_shuffle_masks[columns as u8 as usize - 1][rows as u8 as usize - 1],
    );

    Ok(output)
}
 */
