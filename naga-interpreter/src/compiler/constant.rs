#![allow(unused_variables)]

use arrayvec::ArrayVec;
use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use cranelift_frontend::FunctionBuilder;
use half::f16;

use crate::compiler::{
    Error,
    compiler::Context,
    types::{
        ArrayType,
        FloatWidth,
        IntWidth,
        MatrixType,
        ScalarType,
        Signedness,
        StructType,
        Type,
        VectorType,
    },
    util::{
        ieee16_from_f16,
        math_args_to_array_vec,
    },
    value::{
        ScalarValue,
        TypeOf,
        UnexpectedType,
        Value,
    },
};

#[derive(Clone, Copy, derive_more::Debug)]
pub struct ConstantEvaluator<'source, 'compiler> {
    pub context: &'compiler Context<'source>,
}

impl<'source, 'compiler> ConstantEvaluator<'source, 'compiler> {
    pub fn evaluate_expression(
        &mut self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        use naga::Expression::*;

        let expression = &self.context.source.global_expressions[handle];

        let value = match expression {
            Literal(literal) => self.evaluate_literal(*literal)?,
            Constant(handle) => self.evaluate_constant(*handle)?,
            ZeroValue(handle) => self.evaluate_zero(*handle)?,
            Compose { ty, components } => self.evaluate_compose(*ty, components)?,
            Access { base, index } => self.evaluate_access(*base, *index)?,
            AccessIndex { base, index } => self.evaluate_access_index(*base, *index)?,
            Splat { size, value } => self.evaluate_splat(*size, *value)?,
            Swizzle {
                size,
                vector,
                pattern,
            } => self.evaluate_swizzle(*size, *vector, *pattern)?,
            Unary { op, expr } => self.evaluate_unary(*op, *expr)?,
            Binary { op, left, right } => self.evaluate_binary(*op, *left, *right)?,
            Select {
                condition,
                accept,
                reject,
            } => self.evaluate_select(*condition, *accept, *reject)?,
            Relational { fun, argument } => self.evaluate_relational(*fun, *argument)?,
            Math {
                fun,
                arg,
                arg1,
                arg2,
                arg3,
            } => self.evaluate_math(*fun, math_args_to_array_vec(*arg, *arg1, *arg2, *arg3))?,
            As {
                expr,
                kind,
                convert,
            } => self.evaluate_as(*expr, *kind, *convert)?,
            _ => panic!("Not a constant expression: {expression:?}"),
        };

        Ok(value)
    }

    pub fn evaluate_literal(&mut self, literal: naga::Literal) -> Result<ConstantValue, Error> {
        let value = match literal {
            naga::Literal::F32(value) => ConstantScalar::F32(value),
            naga::Literal::F16(value) => ConstantScalar::F16(value),
            naga::Literal::U32(value) => ConstantScalar::U32(value),
            naga::Literal::I32(value) => ConstantScalar::I32(value),
            naga::Literal::Bool(value) => ConstantScalar::Bool(value),
            _ => panic!("not supported: {literal:?}"),
        };

        Ok(value.into())
    }

    pub fn evaluate_constant(
        &mut self,
        handle: naga::Handle<naga::Constant>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_zero(&mut self, ty: naga::Handle<naga::Type>) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_compose(
        &mut self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_access(
        &mut self,
        base: naga::Handle<naga::Expression>,
        index: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_access_index(
        &mut self,
        base: naga::Handle<naga::Expression>,
        index: u32,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_splat(
        &mut self,
        size: naga::VectorSize,
        value: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_swizzle(
        &mut self,
        size: naga::VectorSize,
        vector: naga::Handle<naga::Expression>,
        pattern: [naga::SwizzleComponent; 4],
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_unary(
        &mut self,
        operator: naga::UnaryOperator,
        input_expression: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        /*use naga::UnaryOperator::*;
        let input_value = self.evaluate_expression(input_expression)?;

        let output = match operator {
            Negate => input_value.evaluate_neg(self)?.into(),
            LogicalNot => input_value.evaluate_log_not(self)?.into(),
            BitwiseNot => input_value.evaluate_bit_not(self)?.into(),
        };

        Ok(output)*/
        todo!();
    }

    pub fn evaluate_binary(
        &mut self,
        operator: naga::BinaryOperator,
        left_expression: naga::Handle<naga::Expression>,
        right_expression: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        /*use naga::BinaryOperator::*;
        let left_value = self.evaluate_expression(left_expression)?;
        let right_value = self.evaluate_expression(right_expression)?;

        let output = match operator {
            Add => left_value.evaluate_add(&right_value, self)?.into(),
            Subtract => left_value.evaluate_sub(&right_value, self)?.into(),
            Multiply => left_value.evaluate_mul(&right_value, self)?.into(),
            Divide => left_value.evaluate_div(&right_value, self)?.into(),
            Modulo => left_value.evaluate_mod(&right_value, self)?.into(),
            Equal => left_value.evaluate_eq(&right_value, self)?.into(),
            NotEqual => left_value.evaluate_neq(&right_value, self)?.into(),
            Less => left_value.evaluate_lt(&right_value, self)?.into(),
            LessEqual => left_value.evaluate_le(&right_value, self)?.into(),
            Greater => left_value.evaluate_gt(&right_value, self)?.into(),
            GreaterEqual => left_value.evaluate_ge(&right_value, self)?.into(),
            And => left_value.evaluate_bit_and(&right_value, self)?.into(),
            ExclusiveOr => left_value.evaluate_bit_xor(&right_value, self)?.into(),
            InclusiveOr => left_value.evaluate_bit_or(&right_value, self)?.into(),
            LogicalAnd => left_value.evaluate_log_and(&right_value, self)?.into(),
            LogicalOr => left_value.evaluate_log_or(&right_value, self)?.into(),
            ShiftLeft => left_value.evaluate_shl(&right_value, self)?.into(),
            ShiftRight => left_value.evaluate_shr(&right_value, self)?.into(),
        };

        Ok(output)*/
        todo!();
    }

    pub fn evaluate_select(
        &mut self,
        condition: naga::Handle<naga::Expression>,
        accept: naga::Handle<naga::Expression>,
        reject: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_relational(
        &mut self,
        function: naga::RelationalFunction,
        argument: naga::Handle<naga::Expression>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_math(
        &mut self,
        function: naga::MathFunction,
        arguments: ArrayVec<naga::Handle<naga::Expression>, 4>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }

    pub fn evaluate_as(
        &mut self,
        expression: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<naga::Bytes>,
    ) -> Result<ConstantValue, Error> {
        todo!();
    }
}

macro_rules! define_constant_value {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Debug)]
        pub enum ConstantValue {
            $($variant($ty),)*
        }

        impl TypeOf for ConstantValue {
            type Type = Type;

            fn type_of(&self) -> Self::Type {
                match self {
                    $(Self::$variant(value) => value.type_of().into(),)*
                }
            }
        }

        impl CompileConstant for ConstantValue {
            type Output = Value;

            fn compile_constant(
                &self,
                context: &Context,
                function_builder: &mut FunctionBuilder,
            ) -> Result<Value, Error> {
                let value = match self {
                    ConstantValue::Scalar(value) => value.compile_constant(context, function_builder)?.into(),
                    ConstantValue::Vector(value) => todo!(), //value.compile_constant(context, function_builder)?.into(),
                    ConstantValue::Matrix(value) => todo!(), //value.compile_constant(context, function_builder)?.into(),
                    ConstantValue::Struct(value) => todo!(), //value.compile_constant(context, function_builder)?.into(),
                    ConstantValue::Array(value) => todo!(), //value.compile_constant(context, function_builder)?.into(),
                };
                Ok(value)
            }
        }

        $(
            impl From<$ty> for ConstantValue {
                fn from(value: $ty) -> Self {
                    Self::$variant(value)
                }
            }

            impl TryFrom<ConstantValue> for $ty {
                type Error = UnexpectedType;

                fn try_from(value: ConstantValue) -> Result<Self, UnexpectedType> {
                    match value {
                        ConstantValue::$variant(value) => Ok(value),
                        _ => Err(UnexpectedType { ty: value.type_of(), expected: stringify!($ty)})
                    }
                }
            }
        )*
    };
}

define_constant_value!(
    Scalar(ConstantScalar),
    Vector(ConstantVector),
    Matrix(ConstantMatrix),
    Struct(ConstantStruct),
    Array(ConstantArray),
);

#[derive(Clone, Copy, Debug)]
pub enum ConstantScalar {
    Bool(bool),
    U32(u32),
    I32(i32),
    F16(f16),
    F32(f32),
}

impl TypeOf for ConstantScalar {
    type Type = ScalarType;

    fn type_of(&self) -> Self::Type {
        match self {
            ConstantScalar::Bool(_) => ScalarType::Bool,
            ConstantScalar::U32(_) => ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
            ConstantScalar::I32(_) => ScalarType::Int(Signedness::Signed, IntWidth::I32),
            ConstantScalar::F16(_) => ScalarType::Float(FloatWidth::F16),
            ConstantScalar::F32(_) => ScalarType::Float(FloatWidth::F32),
        }
    }
}

impl CompileConstant for ConstantScalar {
    type Output = ScalarValue;

    fn compile_constant(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> Result<ScalarValue, Error> {
        let _ = context;

        let value = match self {
            ConstantScalar::Bool(value) => {
                let value = function_builder.ins().iconst(ir::types::I8, *value as i64);
                ScalarValue {
                    ty: ScalarType::Bool,
                    value,
                }
            }
            ConstantScalar::U32(value) => {
                let value = function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new((*value).into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
                    value,
                }
            }
            ConstantScalar::I32(value) => {
                let value = function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new((*value).into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Signed, IntWidth::I32),
                    value,
                }
            }
            ConstantScalar::F16(value) => {
                let value = function_builder.ins().f16const(ieee16_from_f16(*value));
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F16),
                    value,
                }
            }
            ConstantScalar::F32(value) => {
                let value = function_builder.ins().f32const(*value);
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F32),
                    value,
                }
            }
        };

        Ok(value)
    }
}

#[derive(Clone, Debug)]
pub struct ConstantVector {
    pub size: naga::VectorSize,
    pub data: ConstantVectorData<4>,
}

impl TypeOf for ConstantVector {
    type Type = VectorType;

    fn type_of(&self) -> Self::Type {
        VectorType {
            size: self.size,
            scalar: self.data.scalar_type(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConstantMatrix {
    pub columns: naga::VectorSize,
    pub rows: naga::VectorSize,
    pub data: ConstantVectorData<16>,
}

impl TypeOf for ConstantMatrix {
    type Type = MatrixType;

    fn type_of(&self) -> Self::Type {
        MatrixType {
            columns: self.columns,
            rows: self.rows,
            scalar: self.data.scalar_type(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum ConstantVectorData<const N: usize> {
    Bool(ArrayVec<bool, N>),
    U32(ArrayVec<u32, N>),
    I32(ArrayVec<i32, N>),
    F16(ArrayVec<f16, N>),
    F32(ArrayVec<f32, N>),
}

impl<const N: usize> ConstantVectorData<N> {
    pub fn scalar_type(&self) -> ScalarType {
        match self {
            Self::Bool(_) => ScalarType::Bool,
            Self::U32(_) => ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
            Self::I32(_) => ScalarType::Int(Signedness::Signed, IntWidth::I32),
            Self::F16(_) => ScalarType::Float(FloatWidth::F16),
            Self::F32(_) => ScalarType::Float(FloatWidth::F32),
        }
    }

    /*pub fn as_scalars(&self) -> impl Iterator<Item = ScalarValue> {
        todo!();
    }*/
}

#[derive(Clone, Debug)]
pub struct ConstantStruct {
    pub ty: StructType,
    pub members: Vec<ConstantValue>,
}

impl TypeOf for ConstantStruct {
    type Type = StructType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

#[derive(Clone, Debug)]
pub struct ConstantArray {
    pub ty: ArrayType,
    pub elements: Vec<ConstantValue>,
}

impl TypeOf for ConstantArray {
    type Type = ArrayType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

pub trait CompileConstant: Sized {
    type Output;

    fn compile_constant(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> Result<Self::Output, Error>;
}
