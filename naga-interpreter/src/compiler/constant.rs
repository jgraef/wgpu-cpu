#![allow(unused_variables)]

use arrayvec::ArrayVec;
use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use half::f16;

use crate::compiler::{
    Error,
    function::FunctionCompiler,
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
    util::ieee16_from_f16,
    value::{
        ScalarValue,
        TypeOf,
        UnexpectedType,
        Value,
    },
};

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
                compiler: &mut FunctionCompiler,
            ) -> Result<Value, Error> {
                let value = match self {
                    ConstantValue::Scalar(value) => value.compile_constant(compiler)?.into(),
                    ConstantValue::Vector(value) => todo!(), //value.compile_constant(compiler)?.into(),
                    ConstantValue::Matrix(value) => todo!(), //value.compile_constant(compiler)?.into(),
                    ConstantValue::Struct(value) => todo!(), //value.compile_constant(compiler)?.into(),
                    ConstantValue::Array(value) => todo!(), //value.compile_constant(compiler)?.into(),
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

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<ScalarValue, Error> {
        let value = match self {
            ConstantScalar::Bool(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I8, *value as i64);
                ScalarValue {
                    ty: ScalarType::Bool,
                    value,
                }
            }
            ConstantScalar::U32(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new((*value).into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
                    value,
                }
            }
            ConstantScalar::I32(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new((*value).into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Signed, IntWidth::I32),
                    value,
                }
            }
            ConstantScalar::F16(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .f16const(ieee16_from_f16(*value));
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F16),
                    value,
                }
            }
            ConstantScalar::F32(value) => {
                let value = compiler.function_builder.ins().f32const(*value);
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

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error>;
}
