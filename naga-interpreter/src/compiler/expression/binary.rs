use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    types::{
        ScalarType,
        Signedness,
    },
    value::{
        MatrixValue,
        ScalarValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct BinaryExpression {
    pub operator: naga::BinaryOperator,
    pub left_operand: naga::Handle<naga::Expression>,
    pub right_operand: naga::Handle<naga::Expression>,
}

impl CompileExpression for BinaryExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        use naga::BinaryOperator::*;
        let left_value = compiler.compile_expression(self.left_operand)?;
        let right_value = compiler.compile_expression(self.right_operand)?;

        let output = match self.operator {
            Add => left_value.compile_add(compiler, &right_value)?.into(),
            Subtract => left_value.compile_sub(compiler, &right_value)?.into(),
            Multiply => left_value.compile_mul(compiler, &right_value)?.into(),
            Divide => left_value.compile_div(compiler, &right_value)?.into(),
            Modulo => left_value.compile_mod(compiler, &right_value)?.into(),
            Equal => left_value.compile_eq(compiler, &right_value)?.into(),
            NotEqual => left_value.compile_neq(compiler, &right_value)?.into(),
            Less => left_value.compile_lt(compiler, &right_value)?.into(),
            LessEqual => left_value.compile_le(compiler, &right_value)?.into(),
            Greater => left_value.compile_gt(compiler, &right_value)?.into(),
            GreaterEqual => left_value.compile_ge(compiler, &right_value)?.into(),
            And => left_value.compile_bit_and(compiler, &right_value)?.into(),
            ExclusiveOr => left_value.compile_bit_xor(compiler, &right_value)?.into(),
            InclusiveOr => left_value.compile_bit_or(compiler, &right_value)?.into(),
            LogicalAnd => left_value.compile_log_and(compiler, &right_value)?.into(),
            LogicalOr => left_value.compile_log_or(compiler, &right_value)?.into(),
            ShiftLeft => left_value.compile_shl(compiler, &right_value)?.into(),
            ShiftRight => left_value.compile_shr(compiler, &right_value)?.into(),
        };

        Ok(output)
    }
}

impl EvaluateExpression for BinaryExpression {
    type Output = ConstantValue;

    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

macro_rules! define_binary_traits {
    {$($trait:ident :: $method:ident $(auto $auto_kind:ident $auto_args:tt)*;)*} => {
        $(
            pub trait $trait<Other = Self> {
                type Output: Sized;

                fn $method(
                    &self,
                    compiler: &mut FunctionCompiler,
                    other: &Other,
                ) -> Result<Self::Output, Error>;
            }

            $(
                define_binary_traits!(@impl_auto($trait, $method, $auto_kind, $auto_args));
            )?
        )*
    };
    (@impl_auto($trait:ident, $method:ident, Value, [$($variant:ident),*])) => {
        impl $trait for Value {
            type Output = Self;

            fn $method(&self, compiler: &mut FunctionCompiler, other: &Self) -> Result<Self::Output, Error> {
                let output = match (self, other) {
                    $(
                        (Self::$variant(left), Self::$variant(right)) => $trait::$method(left, compiler, right)?.into(),
                    )*
                    _ => panic!("{} not implemented for {:?}", stringify!($trait), self.type_of()),
                };
                Ok(output)
            }
        }
    };
    (@impl_auto($trait:ident, $method:ident, map, [$($ty:ident),*])) => {
        $(
            impl $trait for $ty {
                type Output = Self;

                fn $method(&self, compiler: &mut FunctionCompiler, other: &Self) -> Result<Self::Output, Error> {
                    self.try_zip_map_as_scalars(other, |left, right| $trait::$method(&left, compiler, &right))
                }
            }
        )*
    };
}

define_binary_traits! {
    CompileAdd::compile_add
        auto Value [Scalar, Vector, Matrix]
        auto map [VectorValue, MatrixValue];

    CompileSub::compile_sub
        auto Value [Scalar, Vector, Matrix]
        auto map [VectorValue, MatrixValue];

    CompileMul::compile_mul
        auto Value [Scalar, Vector]
        auto map [VectorValue];

    CompileDiv::compile_div
        auto Value [Scalar, Vector]
        auto map [VectorValue];

    CompileMod::compile_mod
        auto Value [Scalar, Vector]
        auto map [VectorValue];

    CompileEq::compile_eq
        auto Value [Scalar];

    CompileNeq::compile_neq
        auto Value [Scalar];

    CompileLt::compile_lt
        auto Value [Scalar];

    CompileLe::compile_le
        auto Value [Scalar];

    CompileGt::compile_gt
        auto Value [Scalar];

    CompileGe::compile_ge
        auto Value [Scalar];

    CompileBitAnd::compile_bit_and
        auto Value [Scalar];

    CompileBitXor::compile_bit_xor
        auto Value [Scalar];

    CompileBitOr::compile_bit_or
        auto Value [Scalar];

    CompileLogOr::compile_log_or
        auto Value [Scalar];

    CompileLogAnd::compile_log_and
        auto Value [Scalar];

    CompileShl::compile_shl
        auto Value [Scalar];

    CompileShr::compile_shr
        auto Value [Scalar];
}

macro_rules! impl_binary_scalar {
    (@impl($trait:ident, $method:ident, _, $builder:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $invalid()
    };
    (@impl($trait:ident, $method:ident, $instr:ident, $builder:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $builder.ins().$instr($left, $right)
    };
    (@impl($trait:ident, $method:ident, {$function:ident}, $builder:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $function($builder, $left, $right)
    };
    {$($trait:ident :: $method:ident => [$bool:tt, $unsigned:tt, $signed:tt, $float:tt];)*} => {
        $(
            impl $trait for ScalarValue {
                type Output = Self;

                fn $method(
                    &self,
                    compiler: &mut FunctionCompiler,
                    other: &Self,
                ) -> Result<Self::Output, Error> {
                    let invalid  = || -> ! { panic!("{} is not valid for {:?}", stringify!($trait), self.ty) };
                    let value = match self.ty {
                        ScalarType::Bool => {
                            impl_binary_scalar!(@impl($trait, $method, $bool, &mut compiler.function_builder, self.value, other.value, invalid))
                        }
                        ScalarType::Int(Signedness::Unsigned, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $unsigned, &mut compiler.function_builder, self.value, other.value, invalid))
                        },
                        ScalarType::Int(Signedness::Signed, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $signed, &mut compiler.function_builder, self.value, other.value, invalid))
                        },
                        ScalarType::Float(_float_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $float, &mut compiler.function_builder, self.value, other.value, invalid))
                        },
                    };

                    Ok(self.with_ir_value(value))
                }
            }
        )*
    };
}

impl_binary_scalar! {
    CompileAdd::compile_add => [_, iadd, iadd, fadd];
    CompileSub::compile_sub => [_, isub, isub, fsub];
    CompileMul::compile_mul => [_, imul, imul, fmul];
    CompileDiv::compile_div => [_, udiv, sdiv, fdiv];
    CompileMod::compile_mod => [_, urem, srem, {custom_frem}];
    CompileBitAnd::compile_bit_and => [_, band, band, _];
    CompileBitXor::compile_bit_xor => [_, bxor, bxor, _];
    CompileBitOr::compile_bit_or => [_, bor, bor, _];
    CompileLogOr::compile_log_or => [bor , _, _, _];
    CompileLogAnd::compile_log_and => [band , _, _, _];
    CompileShl::compile_shl => [_, ishl, ishl, _];
    CompileShr::compile_shr => [_, ushr, sshr, _];
}

fn custom_frem(builder: &mut FunctionBuilder, left: ir::Value, right: ir::Value) -> ir::Value {
    // https://www.w3.org/TR/WGSL/#arithmetic-expr
    // > If T is a floating point type, the result is equal to:
    // > e1 - e2 * trunc(e1 / e2).

    let x = builder.ins().fdiv(left, right);
    let x = builder.ins().trunc(x);
    let x = builder.ins().fmul(right, x);
    builder.ins().fsub(left, x)
}

macro_rules! impl_comparisions {
    {$($trait:ident :: $method:ident => [$unsigned:tt, $signed:tt, $float:tt];)*} => {
        $(
            impl $trait for ScalarValue {
                type Output = Self;

                fn $method(
                    &self,
                    compiler: &mut FunctionCompiler,
                    other: &Self,
                ) -> Result<Self::Output, Error> {
                    let value = match self.ty {
                        ScalarType::Int(Signedness::Unsigned, _int_width) => {
                            compiler.function_builder
                                .ins()
                                .icmp(ir::condcodes::IntCC::$unsigned, self.value, other.value)
                        },
                        ScalarType::Int(Signedness::Signed, _int_width) => {
                            compiler.function_builder
                                .ins()
                                .icmp(ir::condcodes::IntCC::$signed, self.value, other.value)
                        },
                        ScalarType::Float(_float_width) => {
                            compiler.function_builder
                                .ins()
                                .fcmp(ir::condcodes::FloatCC::$float, self.value, other.value)
                        },
                        _ => panic!("{} is not valid for {:?}", stringify!($trait), self.ty)
                    };

                    Ok(Self { ty: ScalarType::Bool, value })
                }
            }
        )*
    };
}

impl_comparisions! {
    CompileEq::compile_eq => [Equal, Equal, Equal];
    CompileNeq::compile_neq => [NotEqual, NotEqual, NotEqual];
    CompileLt::compile_lt => [UnsignedLessThan, SignedLessThan, LessThan];
    CompileLe::compile_le => [UnsignedLessThanOrEqual, SignedLessThanOrEqual, LessThanOrEqual];
    CompileGt::compile_gt => [UnsignedGreaterThan, SignedGreaterThan, GreaterThan];
    CompileGe::compile_ge => [UnsignedGreaterThanOrEqual, SignedGreaterThanOrEqual, GreaterThanOrEqual];
}
