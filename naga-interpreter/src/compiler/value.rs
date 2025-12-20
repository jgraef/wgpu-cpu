use std::convert::Infallible;

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
    function::FunctionCompiler,
    types::{
        ArrayType,
        FloatWidth,
        MatrixType,
        PointerType,
        ScalarType,
        StructType,
        Type,
        VectorType,
    },
    util::ieee16_from_f16,
};

pub trait AsIrValue {
    fn try_as_ir_value(&self) -> Option<ir::Value>;

    fn as_ir_value(&self) -> ir::Value {
        self.try_as_ir_value()
            .expect("Tried to get a single IR value from a composite value")
    }
}

pub trait AsIrValues {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_;
}

pub trait FromIrValues: TypeOf + Sized {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E>;

    fn from_ir_values_fn(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> ir::Value,
    ) -> Self {
        Self::try_from_ir_values_fn::<Infallible>(context, ty, |ty| Ok(f(ty)))
            .unwrap_or_else(|e| match e {})
    }

    fn try_from_ir_values_iter<E>(
        context: &Context,
        ty: Self::Type,
        values: impl IntoIterator<Item = Result<ir::Value, E>>,
    ) -> Result<Self, E> {
        let mut values = values.into_iter();
        Self::try_from_ir_values_fn(context, ty, move |_ty| {
            values.next().expect("not enough IR values in iterator")
        })
    }

    fn from_ir_values_iter(
        context: &Context,
        ty: Self::Type,
        values: impl IntoIterator<Item = ir::Value>,
    ) -> Self {
        Self::try_from_ir_values_iter::<Infallible>(context, ty, values.into_iter().map(Ok))
            .unwrap_or_else(|e| match e {})
    }
}

pub trait TypeOf {
    type Type;

    fn type_of(&self) -> Self::Type;
}

#[derive(Clone, Copy, Debug)]
pub struct Pointer {
    pub value: ir::Value,
    pub memory_flags: ir::MemFlags,
    pub offset: i32,
}

#[derive(Clone, Copy, Debug)]
pub struct StackLocation {
    pub stack_slot: ir::StackSlot,
    pub offset: i32,
}

impl From<ir::StackSlot> for StackLocation {
    fn from(value: ir::StackSlot) -> Self {
        Self {
            stack_slot: value,
            offset: 0,
        }
    }
}

pub trait Location: Copy {
    fn with_offset(self, offset: i32) -> Self;

    fn add_offset(&mut self, offset: i32) {
        *self = self.with_offset(offset);
    }
}

impl Location for Pointer {
    fn with_offset(mut self, offset: i32) -> Self {
        self.offset = offset;
        self
    }
}

impl Location for StackLocation {
    fn with_offset(mut self, offset: i32) -> Self {
        self.offset = offset;
        self
    }

    fn add_offset(&mut self, offset: i32) {
        self.offset += offset;
    }
}

pub trait Load<P, T>: Sized {
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: T,
        pointer: P,
    ) -> Result<Self, Error>;
}

impl Load<Pointer, ir::Type> for ir::Value {
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ir::Type,
        pointer: Pointer,
    ) -> Result<Self, Error> {
        let _ = context;
        let value =
            function_builder
                .ins()
                .load(ty, pointer.memory_flags, pointer.value, pointer.offset);
        Ok(value)
    }
}

impl Load<StackLocation, ir::Type> for ir::Value {
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ir::Type,
        pointer: StackLocation,
    ) -> Result<Self, Error> {
        let _ = context;
        let value = function_builder
            .ins()
            .stack_load(ty, pointer.stack_slot, pointer.offset);
        Ok(value)
    }
}

pub trait Store<P> {
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: P,
    ) -> Result<(), Error>;
}

impl Store<Pointer> for ir::Value {
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: Pointer,
    ) -> Result<(), Error> {
        let _ = context;
        function_builder
            .ins()
            .store(pointer.memory_flags, *self, pointer.value, pointer.offset);
        Ok(())
    }
}

impl Store<StackLocation> for ir::Value {
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: StackLocation,
    ) -> Result<(), Error> {
        let _ = context;
        function_builder
            .ins()
            .stack_store(*self, pointer.stack_slot, pointer.offset);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScalarValue {
    pub ty: ScalarType,
    pub value: ir::Value,
}

impl ScalarValue {
    pub fn compile_neg_zero(
        float_width: FloatWidth,
        function_compiler: &mut FunctionCompiler,
    ) -> Result<ScalarValue, Error> {
        let value = match float_width {
            FloatWidth::F16 => {
                function_compiler
                    .function_builder
                    .ins()
                    .f16const(ieee16_from_f16(f16::NEG_ZERO))
            }
            FloatWidth::F32 => function_compiler.function_builder.ins().f32const(-0.0),
        };

        Ok(ScalarValue {
            ty: ScalarType::Float(float_width),
            value,
        })
    }

    pub fn with_ir_value(self, value: ir::Value) -> Self {
        Self { ty: self.ty, value }
    }
}

impl AsIrValue for ScalarValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        Some(self.value)
    }

    fn as_ir_value(&self) -> ir::Value {
        self.value
    }
}

impl AsIrValues for ScalarValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        std::iter::once(self.value)
    }
}

impl FromIrValues for ScalarValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let _ = context;
        Ok(Self {
            ty,
            value: f(ty.ir_type())?,
        })
    }
}

impl TypeOf for ScalarValue {
    type Type = ScalarType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

impl<P> Load<P, ScalarType> for ScalarValue
where
    ir::Value: Load<P, ir::Type>,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ScalarType,
        pointer: P,
    ) -> Result<Self, Error> {
        let value = ir::Value::load(context, function_builder, ty.ir_type(), pointer)?;
        Ok(Self { ty, value })
    }
}

impl<P> Store<P> for ScalarValue
where
    ir::Value: Store<P>,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: P,
    ) -> Result<(), Error> {
        self.value.store(context, function_builder, pointer)?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PointerValue {
    pub ty: PointerType,
    pub inner: PointerValueInner,
}

#[derive(Clone, Copy, Debug)]
pub enum PointerValueInner {
    Pointer(Pointer),
    StackLocation(StackLocation),
}

impl PointerValue {
    pub fn deref(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> Result<Value, Error> {
        let base_type = self.ty.base_type(context);

        let value = match self.inner {
            PointerValueInner::Pointer(pointer) => {
                Value::load(context, function_builder, base_type, pointer)?
            }
            PointerValueInner::StackLocation(stack_location) => {
                Value::load(context, function_builder, base_type, stack_location)?
            }
        };

        Ok(value)
    }

    pub fn from_ir_value(ty: PointerType, value: ir::Value) -> Self {
        Self {
            ty,
            inner: PointerValueInner::Pointer(Pointer {
                value,
                memory_flags: ir::MemFlags::new(),
                offset: 0,
            }),
        }
    }

    pub fn from_stack_slot(ty: PointerType, stack_location: impl Into<StackLocation>) -> Self {
        Self {
            ty,
            inner: PointerValueInner::StackLocation(stack_location.into()),
        }
    }
}

impl AsIrValue for PointerValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        Some(self.as_ir_value())
    }

    fn as_ir_value(&self) -> ir::Value {
        match self.inner {
            PointerValueInner::Pointer(pointer) => {
                if pointer.offset != 0 {
                    todo!();
                }
                pointer.value
            }
            PointerValueInner::StackLocation(_stack_location) => todo!(),
        }
    }
}

impl AsIrValues for PointerValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        std::iter::once(self.as_ir_value())
    }
}

impl FromIrValues for PointerValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let _ = context;
        Ok(Self::from_ir_value(ty, f(context.pointer_type())?))
    }
}

impl TypeOf for PointerValue {
    type Type = PointerType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

impl<P> Load<P, PointerType> for PointerValue
where
    ir::Value: Load<P, ir::Type>,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: PointerType,
        pointer: P,
    ) -> Result<Self, Error> {
        let value = ir::Value::load(context, function_builder, context.pointer_type(), pointer)?;
        Ok(Self::from_ir_value(ty, value))
    }
}

impl<P> Store<P> for PointerValue
where
    ir::Value: Store<P>,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: P,
    ) -> Result<(), Error> {
        self.as_ir_value()
            .store(context, function_builder, pointer)?;
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct VectorValue {
    pub ty: VectorType,
    pub values: ArrayVec<ir::Value, 4>,
}

impl VectorValue {
    pub fn try_map_as_scalars<E>(
        &self,
        f: impl FnMut(ScalarValue) -> Result<ScalarValue, E>,
    ) -> Result<Self, E> {
        let (ty, values) = try_map_array_vec_as_scalars(self.ty.scalar, &self.values, f)?;
        Ok(Self {
            ty: self.ty.with_scalar(ty),
            values,
        })
    }

    pub fn try_zip_map_as_scalars<E>(
        &self,
        other: &Self,
        f: impl FnMut(ScalarValue, ScalarValue) -> Result<ScalarValue, E>,
    ) -> Result<Self, E> {
        let (ty, values) = try_zip_map_array_vec_as_scalars(
            self.ty.scalar,
            &self.values,
            other.ty.scalar,
            &other.values,
            f,
        )?;
        Ok(Self {
            ty: self.ty.with_scalar(ty),
            values,
        })
    }
}

impl AsIrValue for VectorValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        (self.values.len() == 1).then(|| self.values[0])
    }
}

impl AsIrValues for VectorValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        self.values.iter().copied()
    }
}

impl FromIrValues for VectorValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let vectorization = context.simd_context.vector(ty);

        let values = std::iter::from_fn(|| Some(f(vectorization.ty)))
            .take(vectorization.count.into())
            .collect::<Result<_, E>>()?;

        Ok(Self { ty, values })
    }
}

impl TypeOf for VectorValue {
    type Type = VectorType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

impl<P> Load<P, VectorType> for VectorValue
where
    ir::Value: Load<P, ir::Type>,
    P: Location,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: VectorType,
        mut pointer: P,
    ) -> Result<Self, Error> {
        let vectorized = context.simd_context.vector(ty);
        let stride = vectorized.stride();

        let values = std::iter::from_fn(|| {
            let value = ir::Value::load(context, function_builder, vectorized.ty, pointer);
            pointer.add_offset(stride);
            Some(value)
        })
        .take(vectorized.count.into())
        .collect::<Result<_, Error>>()?;

        Ok(Self { ty, values })
    }
}

impl<P> Store<P> for VectorValue
where
    ir::Value: Store<P>,
    P: Location,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        mut pointer: P,
    ) -> Result<(), Error> {
        let vectorized = context.simd_context.vector(self.ty);
        let stride = vectorized.stride();

        for value in self.as_ir_values() {
            value.store(context, function_builder, pointer)?;
            pointer.add_offset(stride);
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct MatrixValue {
    pub ty: MatrixType,
    pub values: ArrayVec<ir::Value, 16>,
}

impl MatrixValue {
    pub fn try_map_as_scalars<E>(
        &self,
        f: impl FnMut(ScalarValue) -> Result<ScalarValue, E>,
    ) -> Result<Self, E> {
        let (ty, values) = try_map_array_vec_as_scalars(self.ty.scalar, &self.values, f)?;
        Ok(Self {
            ty: self.ty.with_scalar(ty),
            values,
        })
    }

    pub fn try_zip_map_as_scalars<E>(
        &self,
        other: &Self,
        f: impl FnMut(ScalarValue, ScalarValue) -> Result<ScalarValue, E>,
    ) -> Result<Self, E> {
        let (ty, values) = try_zip_map_array_vec_as_scalars(
            self.ty.scalar,
            &self.values,
            other.ty.scalar,
            &other.values,
            f,
        )?;
        Ok(Self {
            ty: self.ty.with_scalar(ty),
            values,
        })
    }
}

impl AsIrValue for MatrixValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        (self.values.len() == 1).then(|| self.values[0])
    }
}

impl AsIrValues for MatrixValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        self.values.iter().copied()
    }
}

impl FromIrValues for MatrixValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let vectorization = context.simd_context.matrix(ty);

        let values = std::iter::from_fn(|| Some(f(vectorization.ty)))
            .take(vectorization.count.into())
            .collect::<Result<_, E>>()?;

        Ok(Self { ty, values })
    }
}

impl TypeOf for MatrixValue {
    type Type = MatrixType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

impl<P> Load<P, MatrixType> for MatrixValue
where
    ir::Value: Load<P, ir::Type>,
    P: Location,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: MatrixType,
        mut pointer: P,
    ) -> Result<Self, Error> {
        let vectorized = context.simd_context.matrix(ty);
        let stride = vectorized.stride();

        let values = std::iter::from_fn(|| {
            let value = ir::Value::load(context, function_builder, vectorized.ty, pointer);
            pointer.add_offset(stride);
            Some(value)
        })
        .take(vectorized.count.into())
        .collect::<Result<_, Error>>()?;

        Ok(Self { ty, values })
    }
}

impl<P> Store<P> for MatrixValue
where
    ir::Value: Store<P>,
    P: Location,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        mut pointer: P,
    ) -> Result<(), Error> {
        let vectorized = context.simd_context.matrix(self.ty);
        let stride = vectorized.stride();

        for value in self.as_ir_values() {
            value.store(context, function_builder, pointer)?;
            pointer.add_offset(stride);
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct StructValue {
    pub ty: StructType,
    pub members: Vec<Value>,
}

impl AsIrValue for StructValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        None
    }
}

impl AsIrValues for StructValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        self.members.iter().flat_map(|member| member.as_ir_values())
    }
}

impl FromIrValues for StructValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: Self::Type,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let members = ty
            .members(&context.source)
            .into_iter()
            .map(|member| {
                let member_type = context.types[member.ty];
                Value::try_from_ir_values_fn(context, member_type, &mut f)
            })
            .collect::<Result<_, E>>()?;

        Ok(Self { ty, members })
    }
}

impl TypeOf for StructValue {
    type Type = StructType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

impl<P> Load<P, StructType> for StructValue
where
    Value: Load<P, Type>,
    P: Location,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: StructType,
        pointer: P,
    ) -> Result<Self, Error> {
        let members = ty
            .members(context.source)
            .iter()
            .map(|member| {
                let member_type = context.types[member.ty];
                Value::load(
                    context,
                    function_builder,
                    member_type,
                    pointer.with_offset(
                        i32::try_from(member.offset).expect("struct member offset overflow"),
                    ),
                )
            })
            .collect::<Result<_, Error>>()?;

        Ok(Self { ty, members })
    }
}

impl<P> Store<P> for StructValue
where
    Value: Store<P>,
    P: Location,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: P,
    ) -> Result<(), Error> {
        for (member, member_value) in self.ty.members(context.source).iter().zip(&self.members) {
            member_value.store(
                context,
                function_builder,
                pointer.with_offset(
                    i32::try_from(member.offset).expect("struct member offset overflow"),
                ),
            )?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct ArrayValue {
    pub ty: ArrayType,
    pub values: Vec<Value>,
}

impl AsIrValue for ArrayValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        None
    }
}

impl AsIrValues for ArrayValue {
    fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
        self.values.iter().flat_map(|value| value.as_ir_values())
    }
}

impl FromIrValues for ArrayValue {
    fn try_from_ir_values_fn<E>(
        context: &Context,
        ty: ArrayType,
        mut f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let count = ty.expect_size();
        let base_type = ty.base_type(context);

        let values = (0..count)
            .map(|_i| Value::try_from_ir_values_fn(context, base_type, &mut f))
            .collect::<Result<_, E>>()?;

        Ok(Self { ty, values })
    }
}

impl<P> Load<P, ArrayType> for ArrayValue
where
    Value: Load<P, Type>,
    P: Location,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ArrayType,
        mut pointer: P,
    ) -> Result<Self, Error> {
        let count = ty.expect_size();
        let base_type = ty.base_type(context);
        let stride = i32::try_from(ty.stride).expect("array offset overflow");

        let values = (0..count)
            .map(|_i| {
                let value = Value::load(context, function_builder, base_type, pointer)?;
                pointer.add_offset(stride);
                Ok(value)
            })
            .collect::<Result<Vec<Value>, Error>>()?;

        Ok(Self { ty, values })
    }
}

impl<P> Store<P> for ArrayValue
where
    Value: Store<P>,
    P: Location,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        mut pointer: P,
    ) -> Result<(), Error> {
        let stride = i32::try_from(self.ty.stride).expect("array offset overflow");

        for value in &self.values {
            value.store(context, function_builder, pointer)?;
            pointer.add_offset(stride);
        }

        Ok(())
    }
}

impl TypeOf for ArrayValue {
    type Type = ArrayType;

    fn type_of(&self) -> Self::Type {
        self.ty
    }
}

macro_rules! define_value {
    (@impl_load_store([$(($variant:ident, $ty:ty)),*], $pointer:ty)) => {
        impl Load<$pointer, Type> for Value {
            fn load(
                context: &Context,
                function_builder: &mut FunctionBuilder,
                ty: Type,
                pointer: $pointer,
            ) -> Result<Self, Error> {
                let value = match ty {
                    $(Type::$variant(ty) => {
                        Self::$variant(Load::load(
                            context,
                            function_builder,
                            ty,
                            pointer
                        )?)
                    },)*
                };
                Ok(value)
            }
        }

        impl Store<$pointer> for Value {
            fn store(
                &self,
                context: &Context,
                function_builder: &mut FunctionBuilder,
                pointer: $pointer
            ) -> Result<(), Error> {
                match self {
                    $(Self::$variant(value) => value.store(context, function_builder, pointer),)*
                }
            }
        }
    };
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Debug)]
        pub enum Value {
            $($variant($ty),)*
        }

        impl TypeOf for Value {
            type Type = Type;

            fn type_of(&self) -> Self::Type {
                match self {
                    $(Self::$variant(value) => value.type_of().into(),)*
                }
            }
        }

        impl AsIrValue for Value {
            fn try_as_ir_value(&self) -> Option<ir::Value> {
                match self {
                    $(Self::$variant(value) => value.try_as_ir_value(),)*
                }
            }
        }

        impl AsIrValues for Value {
            fn as_ir_values(&self) -> impl Iterator<Item = ir::Value> + '_ {
                let boxed: Box<dyn Iterator<Item = ir::Value>> = match self {
                    $(Self::$variant(value) => Box::new(value.as_ir_values()),)*
                };
                boxed
            }
        }

        impl FromIrValues for Value {
            fn try_from_ir_values_fn<E>(
                context: &Context,
                ty: Self::Type,
                f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
            ) -> Result<Self, E> {
                // necessary for the compiler to not endlessly recurse
                let f = Box::new(f) as Box<dyn FnMut(ir::Type) -> Result<ir::Value, E>>;

                let output = match ty {
                    $(Type::$variant(ty) => Self::$variant(FromIrValues::try_from_ir_values_fn(context, ty, f)?),)*
                };
                Ok(output)
            }
        }

        define_value!(@impl_load_store([$(($variant, $ty)),*], Pointer));
        define_value!(@impl_load_store([$(($variant, $ty)),*], StackLocation));

        $(
            impl From<$ty> for Value {
                fn from(value: $ty) -> Self {
                    Self::$variant(value)
                }
            }

            impl TryFrom<Value> for $ty {
                type Error = UnexpectedType;

                fn try_from(value: Value) -> Result<Self, UnexpectedType> {
                    match value {
                        Value::$variant(value) => Ok(value),
                        _ => Err(UnexpectedType { ty: value.type_of(), expected: stringify!($ty)})
                    }
                }
            }
        )*
    };
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Expected {expected}, but found {ty:?}")]
pub struct UnexpectedType {
    pub ty: Type,
    pub expected: &'static str,
}

define_value!(
    Scalar(ScalarValue),
    Vector(VectorValue),
    Matrix(MatrixValue),
    Pointer(PointerValue),
    Struct(StructValue),
    Array(ArrayValue),
);

fn try_map_array_vec_as_scalars<const N: usize, E>(
    input_scalar_type: ScalarType,
    input_array_vec: &ArrayVec<ir::Value, N>,
    mut f: impl FnMut(ScalarValue) -> Result<ScalarValue, E>,
) -> Result<(ScalarType, ArrayVec<ir::Value, N>), E> {
    let mut output_array_vec = ArrayVec::new();
    let mut output_scalar_type = None;

    for ir_value in input_array_vec {
        let output_value = f(ScalarValue {
            ty: input_scalar_type,
            value: *ir_value,
        })?;
        if let Some(output_scalar_type) = output_scalar_type {
            assert_eq!(output_scalar_type, output_value.ty);
        }
        else {
            output_scalar_type = Some(output_value.ty)
        }
        output_array_vec.push(output_value.value);
    }

    assert_eq!(output_array_vec.len(), input_array_vec.len());
    Ok((output_scalar_type.unwrap(), output_array_vec))
}

fn try_zip_map_array_vec_as_scalars<const N: usize, E>(
    left_scalar_type: ScalarType,
    left_array_vec: &ArrayVec<ir::Value, N>,
    right_scalar_type: ScalarType,
    right_array_vec: &ArrayVec<ir::Value, N>,
    mut f: impl FnMut(ScalarValue, ScalarValue) -> Result<ScalarValue, E>,
) -> Result<(ScalarType, ArrayVec<ir::Value, N>), E> {
    let mut output_array_vec = ArrayVec::new();
    let mut output_scalar_type = None;

    assert_eq!(left_array_vec.len(), right_array_vec.len());

    for (left_ir_value, right_ir_value) in left_array_vec.into_iter().zip(right_array_vec) {
        let output_value = f(
            ScalarValue {
                ty: left_scalar_type,
                value: *left_ir_value,
            },
            ScalarValue {
                ty: right_scalar_type,
                value: *right_ir_value,
            },
        )?;
        if let Some(output_scalar_type) = output_scalar_type {
            assert_eq!(output_scalar_type, output_value.ty);
        }
        else {
            output_scalar_type = Some(output_value.ty)
        }
        output_array_vec.push(output_value.value);
    }

    assert_eq!(output_array_vec.len(), left_array_vec.len());
    Ok((output_scalar_type.unwrap(), output_array_vec))
}
