use std::convert::Infallible;

use arrayvec::ArrayVec;
use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use cranelift_frontend::FunctionBuilder;
use half::f16;

use crate::{
    Error,
    compiler::Context,
    function::FunctionCompiler,
    types::{
        ArrayType,
        FloatWidth,
        IntWidth,
        MatrixType,
        PointerType,
        PointerTypeBase,
        ScalarType,
        Signedness,
        StructType,
        Type,
        UnexpectedType,
        VectorType,
    },
    util::ieee16_from_f16,
};

pub const POINTER_OUT_OF_BOUNDS_TRAP_CODE: ir::TrapCode = const { ir::TrapCode::user(3).unwrap() };

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
    pub offset: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct PointerRange<L> {
    pub pointer: Pointer,
    pub len: L,
}

impl PointerRange<u32> {
    pub fn check_bounds(&self) -> Result<(), Error> {
        if self.pointer.offset < self.len {
            Ok(())
        }
        else {
            // naga will reject programs doing this.
            panic!(
                "out of bounds pointer access: offset={}, len={}",
                self.pointer.offset, self.len
            );
        }
    }

    pub fn as_dynamic(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> PointerRange<ir::Value> {
        let len = function_builder
            .ins()
            .iconst(context.pointer_type(), i64::from(self.len));
        PointerRange {
            pointer: self.pointer,
            len,
        }
    }
}

impl PointerRange<ir::Value> {
    pub fn check_bounds(&self, function_builder: &mut FunctionBuilder) {
        let in_bounds = function_builder.ins().icmp_imm(
            ir::condcodes::IntCC::UnsignedGreaterThan,
            self.len,
            i64::from(self.pointer.offset),
        );

        // todo: i think bounds checks should be done in the CompileAccess impl, but
        // this is a good failsafe. should we branch to the abort block here instead? we
        // could also call into the runtime to set an abort payload with a helpful error
        // message
        function_builder
            .ins()
            .trapz(in_bounds, POINTER_OUT_OF_BOUNDS_TRAP_CODE);
    }

    pub fn with_dynamic_offset(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        offset: ir::Value,
    ) -> Self {
        let offset = function_builder
            .ins()
            .uextend(context.pointer_type(), offset);
        let (len, overflow) = function_builder.ins().ssub_overflow(self.len, offset);
        // todo: abort shader execution with proper abort code
        function_builder
            .ins()
            .trapnz(overflow, POINTER_OUT_OF_BOUNDS_TRAP_CODE);
        let value = function_builder.ins().iadd(self.pointer.value, offset);
        Self {
            pointer: Pointer {
                value,
                memory_flags: self.pointer.memory_flags,
                offset: self.pointer.offset,
            },
            len,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StackLocation {
    pub stack_slot: ir::StackSlot,
    pub offset: u32,
}

impl StackLocation {
    pub fn as_pointer_range(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> PointerRange<u32> {
        let value = function_builder
            .ins()
            .stack_addr(context.pointer_type(), self.stack_slot, 0);

        let data = function_builder
            .func
            .sized_stack_slots
            .get(self.stack_slot)
            .expect("stack slot data not found");

        PointerRange {
            pointer: Pointer {
                value,
                memory_flags: ir::MemFlags::trusted(),
                offset: self.offset,
            },
            len: data.size,
        }
    }
}

impl From<ir::StackSlot> for StackLocation {
    fn from(value: ir::StackSlot) -> Self {
        Self {
            stack_slot: value,
            offset: 0,
        }
    }
}

pub trait PointerOffset: Copy {
    #[must_use]
    fn add_offset(self, offset: u32) -> Self;
}

impl PointerOffset for Pointer {
    fn add_offset(mut self, offset: u32) -> Self {
        self.offset += offset;
        self
    }
}

impl<L> PointerOffset for PointerRange<L>
where
    L: Copy,
{
    fn add_offset(self, offset: u32) -> Self {
        Self {
            pointer: self.pointer.add_offset(offset),
            len: self.len,
        }
    }
}

impl PointerOffset for StackLocation {
    fn add_offset(mut self, offset: u32) -> Self {
        self.offset += offset;
        self
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
        let value = function_builder.ins().load(
            ty,
            pointer.memory_flags,
            pointer.value,
            i32::try_from(pointer.offset).expect("pointer offset overflow"),
        );
        Ok(value)
    }
}

impl Load<PointerRange<u32>, ir::Type> for ir::Value {
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ir::Type,
        pointer: PointerRange<u32>,
    ) -> Result<Self, Error> {
        pointer.check_bounds()?;
        Self::load(context, function_builder, ty, pointer.pointer)
    }
}

impl Load<PointerRange<ir::Value>, ir::Type> for ir::Value {
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ir::Type,
        pointer: PointerRange<ir::Value>,
    ) -> Result<Self, Error> {
        pointer.check_bounds(function_builder);
        Self::load(context, function_builder, ty, pointer.pointer)
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

        let value = function_builder.ins().stack_load(
            ty,
            pointer.stack_slot,
            i32::try_from(pointer.offset).expect("stack offset overflow"),
        );
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
        function_builder.ins().store(
            pointer.memory_flags,
            *self,
            pointer.value,
            i32::try_from(pointer.offset).expect("pointer offset overflow"),
        );
        Ok(())
    }
}

impl Store<PointerRange<u32>> for ir::Value {
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: PointerRange<u32>,
    ) -> Result<(), Error> {
        pointer.check_bounds()?;
        self.store(context, function_builder, pointer.pointer)
    }
}

impl Store<PointerRange<ir::Value>> for ir::Value {
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: PointerRange<ir::Value>,
    ) -> Result<(), Error> {
        let _ = context;
        pointer.check_bounds(function_builder);
        self.store(context, function_builder, pointer.pointer)
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
        function_builder.ins().stack_store(
            *self,
            pointer.stack_slot,
            i32::try_from(pointer.offset).expect("stack offset overflow"),
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ScalarValue {
    pub ty: ScalarType,
    pub value: ir::Value,
}

impl ScalarValue {
    pub fn compile_neg_zero(compiler: &mut FunctionCompiler, float_width: FloatWidth) -> Self {
        let value = match float_width {
            FloatWidth::F16 => {
                compiler
                    .function_builder
                    .ins()
                    .f16const(ieee16_from_f16(f16::NEG_ZERO))
            }
            FloatWidth::F32 => compiler.function_builder.ins().f32const(-0.0),
        };

        Self {
            ty: ScalarType::Float(float_width),
            value,
        }
    }

    pub fn compile_u32(compiler: &mut FunctionCompiler, literal: u32) -> Self {
        let value = compiler
            .function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(literal));
        Self {
            ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
            value,
        }
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

impl PointerValue {
    pub fn deref_load(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> Result<Value, Error> {
        let base_type = self.ty.base_type(context);

        let value = match self.inner {
            PointerValueInner::StaticPointer(pointer) => {
                Value::load(context, function_builder, base_type, pointer)?
            }
            PointerValueInner::DynamicPointer(pointer) => {
                Value::load(context, function_builder, base_type, pointer)?
            }
            PointerValueInner::StackLocation(stack_location) => {
                Value::load(context, function_builder, base_type, stack_location)?
            }
        };

        Ok(value)
    }

    pub fn deref_store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        value: &Value,
    ) -> Result<(), Error> {
        match self.inner {
            PointerValueInner::StaticPointer(pointer) => {
                value.store(context, function_builder, pointer)?
            }
            PointerValueInner::DynamicPointer(pointer) => {
                value.store(context, function_builder, pointer)?
            }
            PointerValueInner::StackLocation(stack_location) => {
                value.store(context, function_builder, stack_location)?
            }
        }

        Ok(())
    }

    pub fn from_stack_slot(ty: PointerType, stack_location: impl Into<StackLocation>) -> Self {
        Self {
            ty,
            inner: PointerValueInner::StackLocation(stack_location.into()),
        }
    }

    pub fn with_dynamic_offset(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        offset: ir::Value,
        new_base_type: PointerTypeBase,
    ) -> Self {
        let pointer = match self.inner {
            PointerValueInner::StaticPointer(pointer_range) => {
                pointer_range.as_dynamic(context, function_builder)
            }
            PointerValueInner::DynamicPointer(pointer_range) => pointer_range,
            PointerValueInner::StackLocation(stack_location) => {
                stack_location
                    .as_pointer_range(context, function_builder)
                    .as_dynamic(context, function_builder)
            }
        };

        Self {
            ty: PointerType {
                base_type: new_base_type,
                address_space: self.ty.address_space,
            },
            inner: PointerValueInner::DynamicPointer(pointer.with_dynamic_offset(
                context,
                function_builder,
                offset,
            )),
        }
    }
}

impl AsIrValue for PointerValue {
    fn try_as_ir_value(&self) -> Option<ir::Value> {
        match self.inner {
            PointerValueInner::StaticPointer(pointer) => {
                if pointer.pointer.offset == 0 {
                    Some(pointer.pointer.value)
                }
                else {
                    // todo: need to emit instruction to add offset to pointer
                    None
                }
            }
            PointerValueInner::DynamicPointer(_pointer) => {
                // Can't turn a dynamic pointer into a single IR value
                None
            }
            PointerValueInner::StackLocation(_stack_location) => {
                // todo: need to emit instruction to get address
                //todo!();
                None
            }
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
        f: impl FnMut(ir::Type) -> Result<ir::Value, E>,
    ) -> Result<Self, E> {
        let _ = (context, ty, f);
        todo!();
        //let pointer = f(context.pointer_type())?;
        //let len = f(context.pointer_type())?;
        //Ok(Self::from_ir_value(ty, f(context.pointer_type())?))
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
        let _ = (context, function_builder, ty, pointer);
        //let value = ir::Value::load(context, function_builder,
        // context.pointer_type(), pointer)?; Ok(Self::from_ir_value(ty, value))
        todo!("check spec if we need to have this.");
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
        let _ = (context, function_builder, pointer);
        //self.as_ir_value()
        //    .store(context, function_builder, pointer)?;
        //Ok(())
        todo!("check spec if we need to have this.");
    }
}

impl PointerOffset for PointerValue {
    fn add_offset(self, offset: u32) -> Self {
        Self {
            ty: self.ty,
            inner: self.inner.add_offset(offset),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum PointerValueInner {
    StaticPointer(PointerRange<u32>),
    DynamicPointer(PointerRange<ir::Value>),
    StackLocation(StackLocation),
}

impl PointerOffset for PointerValueInner {
    fn add_offset(self, offset: u32) -> Self {
        match self {
            Self::StaticPointer(pointer) => Self::StaticPointer(pointer.add_offset(offset)),
            Self::DynamicPointer(pointer) => Self::DynamicPointer(pointer.add_offset(offset)),
            Self::StackLocation(stack_location) => {
                Self::StackLocation(stack_location.add_offset(offset))
            }
        }
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
    P: PointerOffset,
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
            pointer = pointer.add_offset(stride);
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
    P: PointerOffset,
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
            pointer = pointer.add_offset(stride);
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
    P: PointerOffset,
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
            pointer = pointer.add_offset(stride);
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
    P: PointerOffset,
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
            pointer = pointer.add_offset(stride);
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
    P: PointerOffset,
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
                    pointer.add_offset(member.offset),
                )
            })
            .collect::<Result<_, Error>>()?;

        Ok(Self { ty, members })
    }
}

impl<P> Store<P> for StructValue
where
    Value: Store<P>,
    P: PointerOffset,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        pointer: P,
    ) -> Result<(), Error> {
        for (member, member_value) in self.ty.members(context.source).iter().zip(&self.members) {
            member_value.store(context, function_builder, pointer.add_offset(member.offset))?;
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
    P: PointerOffset,
{
    fn load(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        ty: ArrayType,
        mut pointer: P,
    ) -> Result<Self, Error> {
        let count = ty.expect_size();
        let base_type = ty.base_type(context);

        let values = (0..count)
            .map(|_i| {
                let value = Value::load(context, function_builder, base_type, pointer)?;
                pointer = pointer.add_offset(ty.stride);
                Ok(value)
            })
            .collect::<Result<Vec<Value>, Error>>()?;

        Ok(Self { ty, values })
    }
}

impl<P> Store<P> for ArrayValue
where
    Value: Store<P>,
    P: PointerOffset,
{
    fn store(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        mut pointer: P,
    ) -> Result<(), Error> {
        for value in &self.values {
            value.store(context, function_builder, pointer)?;
            pointer = pointer.add_offset(self.ty.stride);
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
        define_value!(@impl_load_store([$(($variant, $ty)),*], PointerRange<u32>));
        define_value!(@impl_load_store([$(($variant, $ty)),*], PointerRange<ir::Value>));
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
