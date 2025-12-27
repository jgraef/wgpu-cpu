#![allow(unused_variables)]

use arrayvec::ArrayVec;
use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use half::f16;

use crate::{
    Error,
    compiler::Context,
    expression::{
        CompileCompose,
        EvaluateCompose,
    },
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
        UnexpectedType,
        VectorType,
    },
    util::ieee16_from_f16,
    value::{
        ArrayValue,
        MatrixValue,
        ScalarValue,
        StructValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

pub trait CompileConstant: Sized {
    type Output;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error>;
}

pub trait WriteConstant {
    fn write_into(&self, context: &Context, destination: &mut [u8]);
}

impl WriteConstant for bool {
    fn write_into(&self, context: &Context, destination: &mut [u8]) {
        destination[0] = *self as u8;
    }
}

macro_rules! write_constant_primitive_impl {
    ($($ty:ty),*) => {
        $(
            impl WriteConstant for $ty {
                fn write_into(&self, context: &Context,destination: &mut [u8]) {
                    let _ = context;
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
            fn write_into(&self, context: &Context, destination: &mut [u8]) {
                match self {
                    $(Self::$variant(value) => value.write_into(context, destination),)*
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
            fn write_into(&self, context: &Context, destination: &mut [u8]) {
                match self {
                    $(Self::$variant(value) => value.write_into(context, destination),)*
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

impl ConstantScalar {
    pub fn as_bool(&self) -> bool {
        match self {
            ConstantScalar::Bool(value) => *value,
            _ => panic!("not a bool {:?}", self),
        }
    }
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

impl CompileConstant for ConstantVector {
    type Output = VectorValue;

    fn compile_constant(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
        // i noticed that this is a naive/silly solution. we can use vconst to compile
        // the constant. but it's written now and i'm too lazy to implement it
        // with vconst right now.
        //
        // an alternative is to just use `WriteConstant` to collect anything that is not
        // a scalar literal into a buffer and just emit loads from that buffer. this
        // might be the right choice for structs and arrays anyway.

        let values = match &self.data {
            ConstantVectorData::Bool(array_vec) => {
                array_vec
                    .iter()
                    .copied()
                    .map(|scalar| ConstantScalar::Bool(scalar).compile_constant(compiler))
                    .collect::<Result<Vec<_>, Error>>()?
            }
            ConstantVectorData::U32(array_vec) => {
                array_vec
                    .iter()
                    .copied()
                    .map(|scalar| ConstantScalar::U32(scalar).compile_constant(compiler))
                    .collect::<Result<Vec<_>, Error>>()?
            }
            ConstantVectorData::I32(array_vec) => {
                array_vec
                    .iter()
                    .copied()
                    .map(|scalar| ConstantScalar::I32(scalar).compile_constant(compiler))
                    .collect::<Result<Vec<_>, Error>>()?
            }
            ConstantVectorData::F16(array_vec) => {
                array_vec
                    .iter()
                    .copied()
                    .map(|scalar| ConstantScalar::F16(scalar).compile_constant(compiler))
                    .collect::<Result<Vec<_>, Error>>()?
            }
            ConstantVectorData::F32(array_vec) => {
                array_vec
                    .iter()
                    .copied()
                    .map(|scalar| ConstantScalar::F32(scalar).compile_constant(compiler))
                    .collect::<Result<Vec<_>, Error>>()?
            }
        };

        let vector_type = VectorType {
            size: self.size,
            scalar: self.data.scalar_type(),
        };
        VectorValue::compile_compose(compiler, vector_type, values)
    }
}

impl WriteConstant for ConstantVector {
    fn write_into(&self, context: &Context, destination: &mut [u8]) {
        let mut offset = 0;

        let scalar_type = self.data.scalar_type();
        let byte_width: usize = scalar_type.byte_width().into();

        for i in 0..u8::from(self.size) {
            self.data
                .write_into(i, context, &mut destination[offset..][..byte_width]);
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
    fn write_into(&self, context: &Context, destination: &mut [u8]) {
        let ty = self.type_of();
        let scalar_size = usize::from(ty.scalar.byte_width());
        let column_stride = usize::from(ty.column_stride()) * scalar_size;

        let mut offset = 0;

        let mut i: usize = 0;
        for _ in 0..u8::from(self.columns) {
            let mut row_offset = offset;
            for _ in 0..u8::from(self.rows) {
                self.data
                    .write_into(i, context, &mut destination[row_offset..][..scalar_size]);
                row_offset += scalar_size;
                i += 1;
            }
            offset += column_stride;
        }
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
        Ok(Self {
            columns: ty.columns,
            rows: ty.rows,
            data: ConstantVectorData::merged(ty.scalar, components.iter().map(|value| &value.data)),
        })
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

    pub fn write_into(&self, i: impl Into<usize>, context: &Context, destination: &mut [u8]) {
        let i = i.into();
        match self {
            ConstantVectorData::Bool(array_vec) => {
                array_vec[i].write_into(context, destination);
            }
            ConstantVectorData::U32(array_vec) => {
                array_vec[i].write_into(context, destination);
            }
            ConstantVectorData::I32(array_vec) => {
                array_vec[i].write_into(context, destination);
            }
            ConstantVectorData::F16(array_vec) => {
                array_vec[i].write_into(context, destination);
            }
            ConstantVectorData::F32(array_vec) => {
                array_vec[i].write_into(context, destination);
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

    pub fn merged<'a, const M: usize>(
        scalar_type: ScalarType,
        components: impl Iterator<Item = &'a ConstantVectorData<M>>,
    ) -> ConstantVectorData<N> {
        let mut output = match scalar_type {
            ScalarType::Bool => ConstantVectorData::Bool(ArrayVec::new()),
            ScalarType::Int(Signedness::Unsigned, IntWidth::I32) => {
                ConstantVectorData::U32(ArrayVec::new())
            }
            ScalarType::Int(Signedness::Signed, IntWidth::I32) => {
                ConstantVectorData::I32(ArrayVec::new())
            }
            ScalarType::Float(FloatWidth::F16) => ConstantVectorData::F16(ArrayVec::new()),
            ScalarType::Float(FloatWidth::F32) => ConstantVectorData::F32(ArrayVec::new()),
        };

        for component in components {
            match (&mut output, component) {
                (ConstantVectorData::Bool(accu), ConstantVectorData::Bool(values)) => {
                    accu.extend(values.iter().copied())
                }
                (ConstantVectorData::U32(accu), ConstantVectorData::U32(values)) => {
                    accu.extend(values.iter().copied())
                }
                (ConstantVectorData::I32(accu), ConstantVectorData::I32(values)) => {
                    accu.extend(values.iter().copied())
                }
                (ConstantVectorData::F16(accu), ConstantVectorData::F16(values)) => {
                    accu.extend(values.iter().copied())
                }
                (ConstantVectorData::F32(accu), ConstantVectorData::F32(values)) => {
                    accu.extend(values.iter().copied())
                }
                _ => unreachable!("invalid"),
            }
        }

        output
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
        let members = self
            .members
            .iter()
            .map(|value| value.compile_constant(compiler))
            .collect::<Result<Vec<Value>, Error>>()?;

        Ok(StructValue {
            ty: self.ty,
            members,
        })
    }
}

impl WriteConstant for ConstantStruct {
    fn write_into(&self, context: &Context, destination: &mut [u8]) {
        let members = self.ty.members(context.source);

        assert_eq!(self.members.len(), members.len());

        for (member, value) in members.iter().zip(&self.members) {
            let offset: usize = member
                .offset
                .try_into()
                .expect("struct member offset overflow");
            let size: usize = context.layouter[member.ty]
                .size
                .try_into()
                .expect("struct member size overflow");
            value.write_into(context, &mut destination[offset..][..size]);
        }
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
        let values = self
            .elements
            .iter()
            .map(|value| value.compile_constant(compiler))
            .collect::<Result<Vec<Value>, Error>>()?;

        Ok(ArrayValue {
            ty: self.ty,
            values,
        })
    }
}

impl WriteConstant for ConstantArray {
    fn write_into(&self, context: &Context, destination: &mut [u8]) {
        let mut offset = 0;
        let stride: usize = self.ty.stride.try_into().expect("array stride overflow");

        for value in &self.elements {
            let size: usize = context.layouter[self.ty.base_type]
                .size
                .try_into()
                .expect("struct member size overflow");
            value.write_into(context, &mut destination[offset..][..size]);

            offset += stride;
        }
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
