use std::cmp::Ordering;

use cranelift_codegen::ir::{
    self,
    InstBuilder as _,
};
use cranelift_frontend::FunctionBuilder;
use half::f16;

use crate::compiler::{
    Error,
    function::FunctionCompiler,
    simd::VectorIrType,
    types::{
        ArrayType,
        CastTo,
        FloatWidth,
        IntWidth,
        MatrixType,
        ScalarType,
        Signedness,
        StructType,
        Type,
        VectorType,
    },
    util::ieee16_from_f16,
    value::{
        ArrayValue,
        AsIrValue,
        MatrixValue,
        ScalarValue,
        StructValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

pub trait CompileAs {
    type Output: Sized;

    fn compile_as(
        &self,
        cast_to: CastTo,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self::Output, Error>;
}

impl CompileAs for ScalarValue {
    type Output = Self;

    fn compile_as(
        &self,
        cast_to: CastTo,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        // see https://gpuweb.github.io/gpuweb/wgsl/#value-constructor-builtin-function

        let input_type = self.ty;
        let output_type = input_type.cast(cast_to);

        let value = match (input_type, output_type) {
            (ScalarType::Bool, ScalarType::Bool) => self.value,
            (ScalarType::Bool, ScalarType::Int(_signedness, _int_width)) => {
                function_compiler
                    .function_builder
                    .ins()
                    .uextend(output_type.ir_type(), self.value)
            }
            (ScalarType::Bool, ScalarType::Float(_float_width)) => {
                function_compiler
                    .function_builder
                    .ins()
                    .fcvt_from_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Int(_signedness, _int_width), ScalarType::Bool) => {
                function_compiler.function_builder.ins().icmp_imm(
                    ir::condcodes::IntCC::NotEqual,
                    self.value,
                    0,
                )
            }
            (
                ScalarType::Int(_signedness, input_int_width),
                ScalarType::Int(output_signedness, output_int_width),
            ) => {
                match input_int_width.cmp(&output_int_width) {
                    Ordering::Less => {
                        match output_signedness {
                            Signedness::Signed => {
                                function_compiler
                                    .function_builder
                                    .ins()
                                    .sextend(output_type.ir_type(), self.value)
                            }
                            Signedness::Unsigned => {
                                function_compiler
                                    .function_builder
                                    .ins()
                                    .uextend(output_type.ir_type(), self.value)
                            }
                        }
                    }
                    Ordering::Equal => self.value,
                    Ordering::Greater => {
                        function_compiler
                            .function_builder
                            .ins()
                            .ireduce(output_type.ir_type(), self.value)
                    }
                }
            }

            (ScalarType::Int(Signedness::Signed, _int_width), ScalarType::Float(_float_width)) => {
                function_compiler
                    .function_builder
                    .ins()
                    .fcvt_from_sint(output_type.ir_type(), self.value)
            }
            (
                ScalarType::Int(Signedness::Unsigned, _int_width),
                ScalarType::Float(_float_width),
            ) => {
                function_compiler
                    .function_builder
                    .ins()
                    .fcvt_from_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Float(float_width), ScalarType::Bool) => {
                let zero = ScalarValue::compile_neg_zero(float_width, function_compiler)?;
                function_compiler.function_builder.ins().fcmp(
                    ir::condcodes::FloatCC::NotEqual,
                    self.value,
                    zero.value,
                )
            }
            (ScalarType::Float(_float_width), ScalarType::Int(Signedness::Signed, _int_width)) => {
                function_compiler
                    .function_builder
                    .ins()
                    .fcvt_to_sint(output_type.ir_type(), self.value)
            }
            (
                ScalarType::Float(_float_width),
                ScalarType::Int(Signedness::Unsigned, _int_width),
            ) => {
                function_compiler
                    .function_builder
                    .ins()
                    .fcvt_to_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Float(input_float_width), ScalarType::Float(output_float_width)) => {
                match input_float_width.cmp(&output_float_width) {
                    Ordering::Less => {
                        function_compiler
                            .function_builder
                            .ins()
                            .fpromote(output_type.ir_type(), self.value)
                    }
                    Ordering::Equal => self.value,
                    Ordering::Greater => {
                        function_compiler
                            .function_builder
                            .ins()
                            .fdemote(output_type.ir_type(), self.value)
                    }
                }
            }
        };

        Ok(Self {
            ty: output_type,
            value,
        })
    }
}

impl CompileAs for VectorValue {
    type Output = Self;

    fn compile_as(
        &self,
        cast_to: CastTo,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        self.try_map_as_scalars(|scalar| scalar.compile_as(cast_to, function_compiler))
    }
}

impl CompileAs for MatrixValue {
    type Output = Self;

    fn compile_as(
        &self,
        cast_to: CastTo,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        self.try_map_as_scalars(|scalar| scalar.compile_as(cast_to, function_compiler))
    }
}

impl CompileAs for Value {
    type Output = Self;

    fn compile_as(
        &self,
        cast_to: CastTo,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let value = match self {
            Value::Scalar(scalar_value) => {
                scalar_value.compile_as(cast_to, function_compiler)?.into()
            }
            Value::Vector(vector_value) => {
                vector_value.compile_as(cast_to, function_compiler)?.into()
            }
            Value::Matrix(matrix_value) => {
                matrix_value.compile_as(cast_to, function_compiler)?.into()
            }
            _ => {
                panic!("Invalid cast: {:?} as {cast_to:?}", self.type_of(),)
            }
        };

        Ok(value)
    }
}

pub trait CompileZero<Type>: Sized {
    fn compile_zero(ty: Type, function_compiler: &mut FunctionCompiler) -> Result<Self, Error>;
}

impl CompileZero<ir::Type> for ir::Value {
    fn compile_zero(ty: ir::Type, function_compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let base_ty = ty.lane_of();
        let mut value = if base_ty.is_int() {
            function_compiler.function_builder.ins().iconst(base_ty, 0)
        }
        else if base_ty == ir::types::F16 {
            function_compiler
                .function_builder
                .ins()
                .f16const(ieee16_from_f16(f16::ZERO))
        }
        else if base_ty == ir::types::F32 {
            function_compiler.function_builder.ins().f32const(0.0)
        }
        else {
            panic!("Invalid to zero {ty:?}");
        };

        if ty.lane_count() > 1 {
            value = function_compiler.function_builder.ins().splat(ty, value);
        }

        Ok(value)
    }
}

impl CompileZero<ScalarType> for ScalarValue {
    fn compile_zero(
        ty: ScalarType,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let value = ir::Value::compile_zero(ty.ir_type(), function_compiler)?;
        Ok(Self { ty, value })
    }
}

impl CompileZero<VectorType> for VectorValue {
    fn compile_zero(
        ty: VectorType,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let vectorized = function_compiler
            .context
            .compiler_context
            .simd_context
            .vector(ty);

        let value = ir::Value::compile_zero(vectorized.ty, function_compiler)?;
        let values = std::iter::repeat(value)
            .take(vectorized.count.into())
            .collect();

        Ok(Self { ty, values })
    }
}

impl CompileZero<MatrixType> for MatrixValue {
    fn compile_zero(
        ty: MatrixType,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let vectorized = function_compiler
            .context
            .compiler_context
            .simd_context
            .matrix(ty);

        let value = ir::Value::compile_zero(vectorized.ty, function_compiler)?;
        let values = std::iter::repeat(value)
            .take(vectorized.count.into())
            .collect();

        Ok(Self { ty, values })
    }
}

impl CompileZero<Type> for Value {
    fn compile_zero(ty: Type, function_compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match ty {
            Type::Scalar(ty) => ScalarValue::compile_zero(ty, function_compiler)?.into(),
            Type::Vector(ty) => VectorValue::compile_zero(ty, function_compiler)?.into(),
            Type::Matrix(ty) => MatrixValue::compile_zero(ty, function_compiler)?.into(),
            Type::Struct(_ty) => todo!(),
            Type::Array(_ty) => todo!(),
            _ => panic!("Invalid to zero {ty:?}"),
        };
        Ok(value)
    }
}

macro_rules! define_unary_traits {
    {$(
        $trait:ident :: $method:ident
        $(auto $auto_kind:ident $auto_args:tt)*
    ;)*} => {
        $(
            pub trait $trait {
                type Output: Sized;

                fn $method(&self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error>;
            }

            $(
                define_unary_traits!(@impl_auto($trait, $method, $auto_kind, $auto_args));
            )?
        )*
    };
    (@impl_auto($trait:ident, $method:ident, Value, [$($variant:ident),*])) => {
        impl $trait for Value {
            type Output = Self;

            fn $method(&self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                let output = match self {
                    $(
                        Self::$variant(inner) => $trait::$method(inner, function_compiler)?.into(),
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

                fn $method(&self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                    self.try_map_as_scalars(|scalar| $trait::$method(&scalar, function_compiler))
                }
            }
        )*
    }
}

define_unary_traits! {
    CompileNeg::compile_neg
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];

    CompileLogNot::compile_log_not
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];

    CompileBitNot::compile_bit_not
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];
}

impl CompileNeg for ScalarValue {
    type Output = Self;

    fn compile_neg(&self, function_compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Int(Signedness::Signed, _int_width) => {
                function_compiler.function_builder.ins().ineg(self.value)
            }
            ScalarType::Float(_float_width) => {
                function_compiler.function_builder.ins().fneg(self.value)
            }
            _ => panic!("negation is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}

impl CompileBitNot for ScalarValue {
    type Output = Self;

    fn compile_bit_not(&self, function_compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Int(_signedness, _int_width) => {
                function_compiler.function_builder.ins().bnot(self.value)
            }
            _ => panic!("bitwise not is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}

impl CompileLogNot for ScalarValue {
    type Output = Self;

    fn compile_log_not(&self, function_compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Bool => {
                function_compiler.function_builder.ins().icmp_imm(
                    ir::condcodes::IntCC::Equal,
                    self.value,
                    0,
                )
            }
            _ => panic!("logical not is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}

macro_rules! define_binary_traits {
    {$($trait:ident :: $method:ident $(auto $auto_kind:ident $auto_args:tt)*;)*} => {
        $(
            pub trait $trait<Other = Self> {
                type Output: Sized;

                fn $method(
                    &self,
                    other: &Other,
                    function_compiler: &mut FunctionCompiler,
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

            fn $method(&self, other: &Self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                let output = match (self, other) {
                    $(
                        (Self::$variant(left), Self::$variant(right)) => $trait::$method(left, right, function_compiler)?.into(),
                    )*
                    _ => panic!("{} not implemented for {:?}", stringify!($trait), self.type_of()),
                };
                Ok(output)
            }
        }
    };
    /*(@impl_auto($trait:ident, $method:ident, Value, [$(($left:ident, $right:ident)),*])) => {
        impl $trait for Value {
            type Output = Self;

            fn $method(&self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                let output = match self {
                    $(
                        Self::$variant(inner) => $trait::$method(inner, function_compiler)?.into(),
                    )*
                    _ => panic!("{} not implemented for {:?}", stringify!($trait), self.type_of()),
                };
                Ok(output)
            }
        }
    };*/
    (@impl_auto($trait:ident, $method:ident, map, [$($ty:ident),*])) => {
        $(
            impl $trait for $ty {
                type Output = Self;

                fn $method(&self, other: &Self, function_compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                    self.try_zip_map_as_scalars(other, |left, right| $trait::$method(&left, &right, function_compiler))
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
                    other: &Self,
                    function_compiler: &mut FunctionCompiler,
                ) -> Result<Self::Output, Error> {
                    let invalid  = || -> ! { panic!("{} is not valid for {:?}", stringify!($trait), self.ty) };
                    let value = match self.ty {
                        ScalarType::Bool => {
                            impl_binary_scalar!(@impl($trait, $method, $bool, &mut function_compiler.function_builder, self.value, other.value, invalid))
                        }
                        ScalarType::Int(Signedness::Unsigned, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $unsigned, &mut function_compiler.function_builder, self.value, other.value, invalid))
                        },
                        ScalarType::Int(Signedness::Signed, _int_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $signed, &mut function_compiler.function_builder, self.value, other.value, invalid))
                        },
                        ScalarType::Float(_float_width) => {
                            impl_binary_scalar!(@impl($trait, $method, $float, &mut function_compiler.function_builder, self.value, other.value, invalid))
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
                    other: &Self,
                    function_compiler: &mut FunctionCompiler,
                ) -> Result<Self::Output, Error> {
                    let value = match self.ty {
                        ScalarType::Int(Signedness::Unsigned, _int_width) => {
                            function_compiler.function_builder
                                .ins()
                                .icmp(ir::condcodes::IntCC::$unsigned, self.value, other.value)
                        },
                        ScalarType::Int(Signedness::Signed, _int_width) => {
                            function_compiler.function_builder
                                .ins()
                                .icmp(ir::condcodes::IntCC::$signed, self.value, other.value)
                        },
                        ScalarType::Float(_float_width) => {
                            function_compiler.function_builder
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

pub trait CompileLiteral: Sized {
    fn compile_literal(
        literal: naga::Literal,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error>;
}

impl CompileLiteral for ScalarValue {
    fn compile_literal(
        literal: naga::Literal,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let value = match literal {
            naga::Literal::F32(value) => {
                let value = function_compiler.function_builder.ins().f32const(value);
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F32),
                    value,
                }
            }
            naga::Literal::F16(value) => {
                let value = function_compiler
                    .function_builder
                    .ins()
                    .f16const(ieee16_from_f16(value));
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F16),
                    value,
                }
            }
            naga::Literal::U32(value) => {
                let value = function_compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new(value.into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
                    value,
                }
            }
            naga::Literal::I32(value) => {
                let value = function_compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new(value.into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Signed, IntWidth::I32),
                    value,
                }
            }
            naga::Literal::Bool(value) => {
                let value = function_compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I8, value as i64);
                ScalarValue {
                    ty: ScalarType::Bool,
                    value,
                }
            }
            _ => panic!("Invalid literal: {literal:?}"),
        };

        Ok(value)
    }
}

impl CompileLiteral for Value {
    fn compile_literal(
        literal: naga::Literal,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        Ok(ScalarValue::compile_literal(literal, function_compiler)?.into())
    }
}

pub trait CompileCompose<Inner>: Sized + TypeOf {
    fn compile_compose(
        ty: Self::Type,
        components: Vec<Inner>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error>;
}

impl CompileCompose<ScalarValue> for VectorValue {
    fn compile_compose(
        ty: VectorType,
        components: Vec<ScalarValue>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let vectorization = function_compiler.context.compiler_context.simd_context[ty];

        let values = match vectorization {
            VectorIrType::Plain { ty: _ } => {
                components
                    .into_iter()
                    .map(|component| component.as_ir_value())
                    .collect()
            }
            VectorIrType::Vector { ty: ir_ty } => {
                let mut components = components.into_iter();
                let first = components.next().unwrap();

                let mut vector = function_compiler
                    .function_builder
                    .ins()
                    .splat(ir_ty, first.value);
                let mut lane = 1;

                for component in components {
                    vector = function_compiler.function_builder.ins().insertlane(
                        vector,
                        component.value,
                        lane as u8,
                    );

                    lane += 1;
                }

                std::iter::once(vector).collect()
            }
        };

        Ok(VectorValue { ty, values })
    }
}

#[allow(unused_variables)]
impl CompileCompose<ScalarValue> for MatrixValue {
    fn compile_compose(
        ty: MatrixType,
        components: Vec<ScalarValue>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        todo!("compose matrix from scalars");
    }
}

#[allow(unused_variables)]
impl CompileCompose<VectorValue> for MatrixValue {
    fn compile_compose(
        ty: MatrixType,
        components: Vec<VectorValue>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        todo!("compose matrix from vectors");
    }
}

impl CompileCompose<Value> for StructValue {
    fn compile_compose(
        ty: StructType,
        components: Vec<Value>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let _ = function_compiler;
        Ok(Self {
            ty,
            members: components,
        })
    }
}

impl CompileCompose<Value> for ArrayValue {
    fn compile_compose(
        ty: ArrayType,
        components: Vec<Value>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let _ = function_compiler;
        Ok(Self {
            ty,
            values: components,
        })
    }
}

impl CompileCompose<Value> for Value {
    fn compile_compose(
        ty: Type,
        components: Vec<Value>,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<Self, Error> {
        let value = match ty {
            Type::Vector(vector_type) => {
                let components = components
                    .into_iter()
                    .map(|component| {
                        match component {
                            Value::Scalar(scalar_value) => scalar_value,
                            _ => {
                                panic!(
                                    "Compose is invalid for {ty:?} with components of {:?}",
                                    component.type_of()
                                )
                            }
                        }
                    })
                    .collect();
                VectorValue::compile_compose(vector_type, components, function_compiler)?.into()
            }
            Type::Matrix(matrix_type) => {
                match &components[0] {
                    Value::Scalar(_) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                match component {
                                    Value::Scalar(scalar_value) => scalar_value,
                                    _ => panic!("Mixed compose is invalid for matrices"),
                                }
                            })
                            .collect();
                        MatrixValue::compile_compose(matrix_type, components, function_compiler)?
                            .into()
                    }
                    Value::Vector(_) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                match component {
                                    Value::Vector(vector_value) => vector_value,
                                    _ => panic!("Mixed compose is invalid for matrices"),
                                }
                            })
                            .collect();
                        MatrixValue::compile_compose(matrix_type, components, function_compiler)?
                            .into()
                    }
                    _ => {
                        panic!(
                            "Compose is invalid for {ty:?} with components of {:?}",
                            components[0].type_of()
                        )
                    }
                }
            }
            Type::Struct(struct_type) => {
                StructValue::compile_compose(struct_type, components, function_compiler)?.into()
            }
            Type::Array(array_type) => {
                ArrayValue::compile_compose(array_type, components, function_compiler)?.into()
            }
            _ => panic!("Compose is invalid for {ty:?}"),
        };
        Ok(value)
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
