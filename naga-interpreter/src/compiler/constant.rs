#![allow(unused_variables)]

use arrayvec::ArrayVec;
use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use half::f16;

use crate::compiler::{
    Error,
    compiler::Context,
    expression::EvaluateCompose,
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
        ArrayValue,
        MatrixValue,
        ScalarValue,
        StructValue,
        TypeOf,
        UnexpectedType,
        Value,
        VectorValue,
    },
};

pub trait CompileConstant: Sized {
    type Output;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error>;
}

pub trait WriteConstant {
    fn write_into(&self, destination: &mut [u8]);
}

impl WriteConstant for bool {
    fn write_into(&self, destination: &mut [u8]) {
        destination[0] = *self as u8;
    }
}

macro_rules! write_constant_primitive_impl {
    ($($ty:ty),*) => {
        $(
            impl WriteConstant for $ty {
                fn write_into(&self, destination: &mut [u8]) {
                    *bytemuck::from_bytes_mut(destination) = *self;
                }
            }
        )*
    };
}

write_constant_primitive_impl!(u32, i32, f16, f32);

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
                    $(Self::$variant(value) => value.compile_constant(compiler)?.into(),)*
                };
                Ok(value)
            }
        }

        impl WriteConstant for ConstantValue {
            fn write_into(&self, destination: &mut [u8]) {
                match self {
                    $(Self::$variant(value) => value.write_into(destination),)*
                }
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

macro_rules! define_constant_scalar {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Copy, Debug)]
        pub enum ConstantScalar {
            $($variant($ty),)*
        }

        impl WriteConstant for ConstantScalar {
            fn write_into(&self, destination: &mut [u8]) {
                match self {
                    $(Self::$variant(value) => value.write_into(destination),)*
                }
            }
        }

        $(
            impl TryFrom<ConstantScalar> for $ty {
                type Error = ConstantScalar;

                fn try_from(value: ConstantScalar) -> Result<$ty, Self::Error> {
                    match value {
                        ConstantScalar::$variant(value) => Ok(value),
                        _ => Err(value)
                    }
                }
            }
        )*
    };
}

#[rustfmt::skip]
define_constant_scalar!(
    Bool(bool),
    U32(u32),
    I32(i32),
    F16(f16),
    F32(f32),
);

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

impl CompileConstant for ConstantVector {
    type Output = VectorValue;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
        todo!()
    }
}

impl WriteConstant for ConstantVector {
    fn write_into(&self, destination: &mut [u8]) {
        let mut offset = 0;

        let scalar_type = self.data.scalar_type();
        let byte_width: usize = scalar_type.byte_width().into();

        for i in 0..u8::from(self.size) {
            self.data
                .write_into(i, &mut destination[offset..][..byte_width]);
            offset += byte_width;
        }
    }
}

impl EvaluateCompose<ConstantScalar> for ConstantVector {
    fn evaluate_compose(
        context: &Context,
        ty: VectorType,
        components: Vec<ConstantScalar>,
    ) -> Result<Self, Error> {
        assert_eq!(components.len(), usize::from(u8::from(ty.size)));

        Ok(Self {
            size: ty.size,
            data: ConstantVectorData::from_scalars(ty.scalar, components),
        })
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

impl CompileConstant for ConstantMatrix {
    type Output = MatrixValue;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
        todo!("compile constant matrix")
    }
}

impl WriteConstant for ConstantMatrix {
    fn write_into(&self, destination: &mut [u8]) {
        todo!("write constant matrix")
    }
}

impl EvaluateCompose<ConstantScalar> for ConstantMatrix {
    fn evaluate_compose(
        context: &Context,
        ty: MatrixType,
        components: Vec<ConstantScalar>,
    ) -> Result<Self, Error> {
        todo!("compose constant matrix from scalars");
    }
}

impl EvaluateCompose<ConstantVector> for ConstantMatrix {
    fn evaluate_compose(
        context: &Context,
        ty: MatrixType,
        components: Vec<ConstantVector>,
    ) -> Result<Self, Error> {
        todo!("compose constant matrix from vectors");
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

    pub fn write_into(&self, i: impl Into<usize>, destination: &mut [u8]) {
        let i = i.into();
        match self {
            ConstantVectorData::Bool(array_vec) => {
                array_vec[i].write_into(destination);
            }
            ConstantVectorData::U32(array_vec) => {
                array_vec[i].write_into(destination);
            }
            ConstantVectorData::I32(array_vec) => {
                array_vec[i].write_into(destination);
            }
            ConstantVectorData::F16(array_vec) => {
                array_vec[i].write_into(destination);
            }
            ConstantVectorData::F32(array_vec) => {
                array_vec[i].write_into(destination);
            }
        }
    }

    pub fn from_scalars(
        ty: ScalarType,
        components: impl IntoIterator<Item = ConstantScalar>,
    ) -> Self {
        match ty {
            ScalarType::Bool => {
                Self::Bool(
                    components
                        .into_iter()
                        .map(|component| component.try_into().unwrap())
                        .collect(),
                )
            }
            ScalarType::Int(Signedness::Unsigned, IntWidth::I32) => {
                Self::U32(
                    components
                        .into_iter()
                        .map(|component| component.try_into().unwrap())
                        .collect(),
                )
            }
            ScalarType::Int(Signedness::Signed, IntWidth::I32) => {
                Self::I32(
                    components
                        .into_iter()
                        .map(|component| component.try_into().unwrap())
                        .collect(),
                )
            }
            ScalarType::Float(FloatWidth::F16) => {
                Self::F16(
                    components
                        .into_iter()
                        .map(|component| component.try_into().unwrap())
                        .collect(),
                )
            }
            ScalarType::Float(FloatWidth::F32) => {
                Self::F32(
                    components
                        .into_iter()
                        .map(|component| component.try_into().unwrap())
                        .collect(),
                )
            }
        }
    }
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

impl CompileConstant for ConstantStruct {
    type Output = StructValue;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
        todo!()
    }
}

impl WriteConstant for ConstantStruct {
    fn write_into(&self, destination: &mut [u8]) {
        todo!()
    }
}

impl EvaluateCompose<ConstantValue> for ConstantStruct {
    fn evaluate_compose(
        context: &Context,
        ty: StructType,
        components: Vec<ConstantValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            ty,
            members: components,
        })
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

impl CompileConstant for ConstantArray {
    type Output = ArrayValue;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
        todo!()
    }
}

impl WriteConstant for ConstantArray {
    fn write_into(&self, destination: &mut [u8]) {
        todo!()
    }
}

impl EvaluateCompose<ConstantValue> for ConstantArray {
    fn evaluate_compose(
        context: &Context,
        ty: ArrayType,
        components: Vec<ConstantValue>,
    ) -> Result<Self, Error> {
        Ok(Self {
            ty,
            elements: components,
        })
    }
}
