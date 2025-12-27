use cranelift_codegen::ir::{
    self,
    BlockArg,
    InstBuilder,
};

use crate::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::{
        ABORT_CODE_TYPE,
        AbortCode,
        FunctionCompiler,
    },
    simd::{
        MatrixIrType,
        VectorIrType,
    },
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
        let left_value = self.left_operand.compile_expression(compiler)?;
        let right_value = self.right_operand.compile_expression(compiler)?;

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
    (@impl_auto($trait:ident, $method:ident, Value, [$(($left:ident, $right:ident)),*])) => {
        impl $trait for Value {
            type Output = Self;

            fn $method(&self, compiler: &mut FunctionCompiler, other: &Self) -> Result<Self::Output, Error> {
                let output = match (self, other) {
                    $(
                        (Self::$left(left), Self::$right(right)) => $trait::$method(left, compiler, right)?.into(),
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
        auto Value [
            (Scalar, Scalar),
            (Vector, Vector),
            (Matrix, Matrix)
        ]
        auto map [VectorValue, MatrixValue];

    CompileSub::compile_sub
        auto Value [
            (Scalar, Scalar),
            (Vector, Vector),
            (Matrix, Matrix)
        ]
        auto map [VectorValue, MatrixValue];

    CompileMul::compile_mul
        auto Value [
            (Scalar, Scalar),
            (Vector, Vector),
            (Matrix, Matrix),
            (Scalar, Vector),
            (Vector, Scalar),
            (Scalar, Matrix),
            (Matrix, Scalar),
            (Vector, Matrix),
            (Matrix, Vector)
        ]
        auto map [VectorValue];

    CompileDiv::compile_div
        auto Value [
            (Scalar, Scalar),
            (Vector, Vector)
        ]
        auto map [VectorValue];

    CompileMod::compile_mod
        auto Value [
            (Scalar, Scalar),
            (Vector, Vector)
        ]
        auto map [VectorValue];

    CompileEq::compile_eq
        auto Value [(Scalar, Scalar)];

    CompileNeq::compile_neq
        auto Value [(Scalar, Scalar)];

    CompileLt::compile_lt
        auto Value [(Scalar, Scalar)];

    CompileLe::compile_le
        auto Value [(Scalar, Scalar)];

    CompileGt::compile_gt
        auto Value [(Scalar, Scalar)];

    CompileGe::compile_ge
        auto Value [(Scalar, Scalar)];

    CompileBitAnd::compile_bit_and
        auto Value [(Scalar, Scalar)];

    CompileBitXor::compile_bit_xor
        auto Value [(Scalar, Scalar)];

    CompileBitOr::compile_bit_or
        auto Value [(Scalar, Scalar)];

    CompileLogOr::compile_log_or
        auto Value [(Scalar, Scalar)];

    CompileLogAnd::compile_log_and
        auto Value [(Scalar, Scalar)];

    CompileShl::compile_shl
        auto Value [(Scalar, Scalar)];

    CompileShr::compile_shr
        auto Value [(Scalar, Scalar)];
}

impl CompileMul<MatrixValue> for ScalarValue {
    type Output = MatrixValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &MatrixValue,
    ) -> Result<Self::Output, Error> {
        other.compile_mul(compiler, self)
    }
}

impl CompileMul<ScalarValue> for MatrixValue {
    type Output = MatrixValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &ScalarValue,
    ) -> Result<Self::Output, Error> {
        let scalar_value = match compiler.context.simd_context[self.ty] {
            MatrixIrType::Plain { ty } => *other,
            MatrixIrType::ColumnVector { ty } | MatrixIrType::FullVector { ty } => {
                ScalarValue {
                    ty: other.ty,
                    value: compiler.function_builder.ins().splat(ty, other.value),
                }
            }
        };

        self.try_map_as_scalars(|vector_value| vector_value.compile_mul(compiler, &scalar_value))
    }
}

impl CompileMul<MatrixValue> for MatrixValue {
    type Output = MatrixValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &MatrixValue,
    ) -> Result<Self::Output, Error> {
        todo!("matrix * matrix")
    }
}

impl CompileMul<MatrixValue> for VectorValue {
    type Output = VectorValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &MatrixValue,
    ) -> Result<Self::Output, Error> {
        todo!("vector * matrix")
    }
}

impl CompileMul<VectorValue> for MatrixValue {
    type Output = VectorValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &VectorValue,
    ) -> Result<Self::Output, Error> {
        let matrix_vectorization = compiler.context.simd_context[self.ty];
        let vector_vectorization = compiler.context.simd_context[other.ty];

        let values = match (matrix_vectorization, vector_vectorization) {
            (MatrixIrType::Plain { ty: matrix_type }, VectorIrType::Plain { ty: vector_type }) => {
                todo!()
            }
            (
                MatrixIrType::ColumnVector {
                    ty: matrix_column_type,
                },
                VectorIrType::Vector { ty: vector_type },
            ) => {
                assert_eq!(self.ty.columns, other.ty.size);

                let columns = u8::from(self.ty.columns);
                assert_eq!(self.values.len(), columns.into());
                assert_eq!(other.values.len(), 1);

                let mut column_sum = None;

                for i in 0..columns {
                    // todo: shuffle instead?
                    let v_i = compiler
                        .function_builder
                        .ins()
                        .extractlane(other.values[0], i);
                    let v_i = compiler.function_builder.ins().splat(vector_type, v_i);

                    // wgsl apparently only allows float matrices
                    assert!(vector_type.lane_of().is_float());

                    let x_i = compiler
                        .function_builder
                        .ins()
                        .fmul(v_i, self.values[usize::from(i)]);

                    if let Some(column_sum) = &mut column_sum {
                        *column_sum = compiler.function_builder.ins().fadd(*column_sum, x_i);
                    }
                    else {
                        column_sum = Some(x_i);
                    }
                }

                [column_sum.unwrap()].into_iter().collect()
            }
            (
                MatrixIrType::FullVector { ty: matrix_type },
                VectorIrType::Vector { ty: vector_type },
            ) => todo!(),
            _ => {
                panic!(
                    "bug: how is this vectorization possible: matrix={matrix_vectorization:?}, vector={vector_vectorization:?}"
                )
            }
        };

        Ok(VectorValue {
            ty: other.ty,
            values,
        })
    }
}

impl CompileMul<VectorValue> for ScalarValue {
    type Output = VectorValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &VectorValue,
    ) -> Result<Self::Output, Error> {
        other.compile_mul(compiler, self)
    }
}

impl CompileMul<ScalarValue> for VectorValue {
    type Output = VectorValue;

    fn compile_mul(
        &self,
        compiler: &mut FunctionCompiler,
        other: &ScalarValue,
    ) -> Result<Self::Output, Error> {
        let scalar_value = match compiler.context.simd_context[self.ty] {
            VectorIrType::Plain { ty } => *other,
            VectorIrType::Vector { ty } => {
                ScalarValue {
                    ty: other.ty,
                    value: compiler.function_builder.ins().splat(ty, other.value),
                }
            }
        };

        self.try_map_as_scalars(|vector_value| vector_value.compile_mul(compiler, &scalar_value))
    }
}

macro_rules! impl_binary_scalar {
    (@impl($trait:ident, $method:ident, _, $compiler:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $invalid()
    };
    (@impl($trait:ident, $method:ident, $instr:ident, $compiler:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $compiler.function_builder.ins().$instr($left, $right)
    };
    (@impl($trait:ident, $method:ident, {$function:ident}, $compiler:expr, $left:expr, $right:expr, $invalid:expr)) => {
        $function($compiler, $left, $right)?
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
                            impl_binary_scalar!(@impl($trait, $method, $bool, compiler, self.value, other.value, invalid))
                        }
                        ScalarType::Int(Signedness::Unsigned, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $unsigned, compiler, self.value, other.value, invalid))
                        },
                        ScalarType::Int(Signedness::Signed, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $signed, compiler, self.value, other.value, invalid))
                        },
                        ScalarType::Float(_float_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $float, compiler, self.value, other.value, invalid))
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
    CompileDiv::compile_div => [_, {checked_udiv}, {checked_sdiv}, fdiv];
    CompileMod::compile_mod => [_, {checked_urem}, {checked_srem}, {custom_frem}];
    CompileBitAnd::compile_bit_and => [_, band, band, _];
    CompileBitXor::compile_bit_xor => [_, bxor, bxor, _];
    CompileBitOr::compile_bit_or => [_, bor, bor, _];
    CompileLogOr::compile_log_or => [bor , _, _, _];
    CompileLogAnd::compile_log_and => [band , _, _, _];
    CompileShl::compile_shl => [_, ishl, ishl, _];
    CompileShr::compile_shr => [_, ushr, sshr, _];
}

fn custom_frem(
    compiler: &mut FunctionCompiler,
    left: ir::Value,
    right: ir::Value,
) -> Result<ir::Value, Error> {
    // https://www.w3.org/TR/WGSL/#arithmetic-expr
    // > If T is a floating point type, the result is equal to:
    // > e1 - e2 * trunc(e1 / e2).

    let x = compiler.function_builder.ins().fdiv(left, right);
    let x = compiler.function_builder.ins().trunc(x);
    let x = compiler.function_builder.ins().fmul(right, x);
    let value = compiler.function_builder.ins().fsub(left, x);
    Ok(value)
}

macro_rules! impl_checked_if_right_is_zero {
    ($name:ident => $inst:ident) => {
        fn $name(
            compiler: &mut FunctionCompiler,
            left: ir::Value,
            right: ir::Value,
        ) -> Result<ir::Value, Error> {
            let invalid =
                compiler
                    .function_builder
                    .ins()
                    .icmp_imm(ir::condcodes::IntCC::Equal, right, 0);

            let continue_block = compiler.function_builder.create_block();
            let abort_code = compiler
                .function_builder
                .ins()
                .iconst(ABORT_CODE_TYPE, AbortCode::DivisionByZero);
            compiler.function_builder.ins().brif(
                right,
                continue_block,
                [],
                compiler.abort_block,
                [&BlockArg::Value(abort_code)],
            );
            compiler.function_builder.seal_block(continue_block);
            compiler.function_builder.switch_to_block(continue_block);

            let value = compiler.function_builder.ins().$inst(left, right);
            Ok(value)
        }
    };
}

impl_checked_if_right_is_zero!(checked_udiv => udiv);
impl_checked_if_right_is_zero!(checked_urem => urem);
impl_checked_if_right_is_zero!(checked_srem => srem);

fn checked_sdiv(
    compiler: &mut FunctionCompiler,
    left: ir::Value,
    right: ir::Value,
) -> Result<ir::Value, Error> {
    {
        // https://docs.rs/cranelift-codegen/latest/cranelift_codegen/ir/trait.InstBuilder.html#method.sdiv
        // jumps to abort block if left = -2^(B-1) and right = -1
        let ty = compiler.function_builder.func.dfg.value_type(right);
        let b = ty.bits();

        let continue_block = compiler.function_builder.create_block();

        let invalid_left = compiler.function_builder.ins().icmp_imm(
            ir::condcodes::IntCC::Equal,
            left,
            -(1 << (i64::from(b) - 1)),
        );

        let invalid_right =
            compiler
                .function_builder
                .ins()
                .icmp_imm(ir::condcodes::IntCC::Equal, right, -1);

        let invalid = compiler
            .function_builder
            .ins()
            .band(invalid_left, invalid_right);

        let abort_code = compiler
            .function_builder
            .ins()
            .iconst(ABORT_CODE_TYPE, AbortCode::Overflow);
        compiler.function_builder.ins().brif(
            invalid,
            compiler.abort_block,
            [&BlockArg::Value(abort_code)],
            continue_block,
            [],
        );

        compiler.function_builder.seal_block(continue_block);
        compiler.function_builder.switch_to_block(continue_block);
    }

    {
        // check if division by 0

        let continue_block = compiler.function_builder.create_block();

        let invalid =
            compiler
                .function_builder
                .ins()
                .icmp_imm(ir::condcodes::IntCC::Equal, right, 0);
        let abort_code = compiler
            .function_builder
            .ins()
            .iconst(ABORT_CODE_TYPE, AbortCode::DivisionByZero);
        compiler.function_builder.ins().brif(
            invalid,
            compiler.abort_block,
            [&BlockArg::Value(abort_code)],
            continue_block,
            [],
        );

        compiler.function_builder.seal_block(continue_block);
        compiler.function_builder.switch_to_block(continue_block);
    }

    let value = compiler.function_builder.ins().sdiv(left, right);
    Ok(value)
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
