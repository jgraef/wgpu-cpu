use cranelift_codegen::ir;
use naga::ScalarKind;

use crate::compiler::{
    compiler::Context,
    simd::{
        MatrixIrType,
        VectorIrType,
    },
};

pub trait AsIrTypes {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type>;

    fn as_ir_type(&self, context: &Context) -> ir::Type {
        self.try_as_ir_type(context)
            .expect("Tried to get a single IR type from a composite type")
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a;

    fn try_ir_size(&self, context: &Context) -> Option<usize>;

    fn ir_size(&self, context: &Context) -> usize {
        self.try_ir_size(context)
            .expect("Can't determine IR value count")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CastTo {
    Bool,
    Int(Signedness, Option<IntWidth>),
    Float(Option<FloatWidth>),
}

impl CastTo {
    pub fn from_naga(kind: naga::ScalarKind, convert: Option<naga::Bytes>) -> Self {
        let int_width = || convert.map(|convert| IntWidth::try_from_byte_width(convert).unwrap());
        let float_width =
            || convert.map(|convert| FloatWidth::try_from_byte_width(convert).unwrap());

        match kind {
            ScalarKind::Sint => Self::Int(Signedness::Signed, int_width()),
            ScalarKind::Uint => Self::Int(Signedness::Unsigned, int_width()),
            ScalarKind::Float => Self::Float(float_width()),
            ScalarKind::Bool => Self::Bool,
            _ => panic!("Invalid cast to {kind:?}"),
        }
    }
}

#[derive(Clone, Debug, thiserror::Error)]
pub enum InvalidType {
    #[error("Invalid integer width: {width}")]
    InvalidIntWidth { width: u8 },

    #[error("Invalid float width: {width}")]
    InvalidFloatWidth { width: u8 },

    #[error("Invalid bool width: {width}")]
    InvalidBoolWidth { width: u8 },

    #[error("Invalid abstract scalar: {kind:?}  {width:?}")]
    Abstract { kind: naga::ScalarKind, width: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum IntWidth {
    I32,
}

impl IntWidth {
    pub fn try_from_byte_width(width: u8) -> Result<Self, InvalidType> {
        match width {
            4 => Ok(Self::I32),
            _ => Err(InvalidType::InvalidIntWidth { width }),
        }
    }

    pub fn byte_width(&self) -> u8 {
        match self {
            IntWidth::I32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signedness {
    Signed,
    Unsigned,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FloatWidth {
    F16,
    F32,
}

impl FloatWidth {
    pub fn try_from_byte_width(width: u8) -> Result<Self, InvalidType> {
        match width {
            2 => Ok(Self::F16),
            4 => Ok(Self::F32),
            _ => Err(InvalidType::InvalidFloatWidth { width }),
        }
    }

    pub fn byte_width(&self) -> u8 {
        match self {
            FloatWidth::F16 => 2,
            FloatWidth::F32 => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarType {
    Bool,
    Int(Signedness, IntWidth),
    Float(FloatWidth),
}

impl ScalarType {
    pub fn ir_type(&self) -> ir::Type {
        match self {
            ScalarType::Bool => ir::types::I8,
            ScalarType::Int(_signedness, IntWidth::I32) => ir::types::I32,
            ScalarType::Float(FloatWidth::F16) => ir::types::F16,
            ScalarType::Float(FloatWidth::F32) => ir::types::F32,
        }
    }

    pub fn from_naga(scalar: naga::Scalar) -> Result<Self, InvalidType> {
        let output = match scalar {
            naga::Scalar {
                kind: naga::ScalarKind::Bool,
                width,
            } => {
                if width != 1 {
                    return Err(InvalidType::InvalidBoolWidth { width });
                }
                ScalarType::Bool
            }
            naga::Scalar {
                kind: naga::ScalarKind::Uint,
                width,
            } => ScalarType::Int(Signedness::Unsigned, IntWidth::try_from_byte_width(width)?),
            naga::Scalar {
                kind: naga::ScalarKind::Sint,
                width,
            } => ScalarType::Int(Signedness::Signed, IntWidth::try_from_byte_width(width)?),
            naga::Scalar {
                kind: naga::ScalarKind::Float,
                width,
            } => ScalarType::Float(FloatWidth::try_from_byte_width(width)?),
            naga::Scalar {
                kind: naga::ScalarKind::AbstractFloat,
                width,
            } => {
                return Err(InvalidType::Abstract {
                    kind: naga::ScalarKind::AbstractFloat,
                    width,
                });
            }
            naga::Scalar {
                kind: naga::ScalarKind::AbstractInt,
                width,
            } => {
                return Err(InvalidType::Abstract {
                    kind: naga::ScalarKind::AbstractInt,
                    width,
                });
            }
        };

        Ok(output)
    }

    pub fn byte_width(&self) -> u8 {
        match self {
            ScalarType::Bool => 1,
            ScalarType::Int(_signedness, int_width) => int_width.byte_width(),
            ScalarType::Float(float_width) => float_width.byte_width(),
        }
    }

    pub fn cast(&self, cast_to: CastTo) -> Self {
        match (self, cast_to) {
            (ScalarType::Bool, CastTo::Bool) => Self::Bool,
            (ScalarType::Bool, CastTo::Int(signedness, int_width)) => {
                Self::Int(
                    signedness,
                    int_width.expect("can't bitcast from bool to int"),
                )
            }
            (ScalarType::Bool, CastTo::Float(float_width)) => {
                Self::Float(float_width.expect("can't bitcast from bool to float"))
            }
            (ScalarType::Int(_signedness, _int_width), CastTo::Bool) => Self::Bool,
            (
                ScalarType::Int(_input_signedness, input_int_width),
                CastTo::Int(output_signedness, output_int_width),
            ) => {
                Self::Int(
                    output_signedness,
                    output_int_width.unwrap_or(*input_int_width),
                )
            }
            (ScalarType::Int(_signedness, _int_width), CastTo::Float(float_width)) => {
                Self::Float(float_width.expect("can't bitcast from int to float"))
            }
            (ScalarType::Float(_float_width), CastTo::Bool) => Self::Bool,
            (ScalarType::Float(_float_width), CastTo::Int(signedness, int_width)) => {
                Self::Int(
                    signedness,
                    int_width.expect("can't bitcast from float to int"),
                )
            }
            (ScalarType::Float(input_float_width), CastTo::Float(output_float_width)) => {
                Self::Float(output_float_width.unwrap_or(*input_float_width))
            }
        }
    }
}

impl TryFrom<naga::Scalar> for ScalarType {
    type Error = InvalidType;

    fn try_from(value: naga::Scalar) -> Result<Self, Self::Error> {
        Self::from_naga(value)
    }
}

impl AsIrTypes for ScalarType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        let _ = context;
        Some(self.ir_type())
    }

    fn as_ir_type(&self, context: &Context) -> ir::Type {
        let _ = context;
        self.ir_type()
    }

    fn as_ir_types(&self, context: &Context) -> impl Iterator<Item = ir::Type> {
        let _ = context;
        std::iter::once(self.ir_type())
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        let _ = context;
        Some(1)
    }
}

// note: we can't store the base_type as [`Type`] directly because that would
// make the type recursive, requiring a Box, making the type not Copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerType {
    Pointer {
        base: naga::Handle<naga::Type>,
        space: naga::AddressSpace,
    },
    ScalarPointer {
        base: ScalarType,
        space: naga::AddressSpace,
    },
    VectorPointer {
        base: VectorType,
        space: naga::AddressSpace,
    },
}

impl PointerType {
    pub fn from_naga(
        base: naga::Handle<naga::Type>,
        space: naga::AddressSpace,
    ) -> Result<Self, InvalidType> {
        Ok(Self::Pointer { base, space })
    }

    pub fn from_naga_value(
        scalar: naga::Scalar,
        size: Option<naga::VectorSize>,
        space: naga::AddressSpace,
    ) -> Result<Self, InvalidType> {
        if let Some(size) = size {
            Ok(Self::VectorPointer {
                base: VectorType {
                    size,
                    scalar: ScalarType::from_naga(scalar)?,
                },
                space,
            })
        }
        else {
            Ok(Self::ScalarPointer {
                base: ScalarType::from_naga(scalar)?,
                space,
            })
        }
    }

    pub fn base_type(&self, context: &Context) -> Type {
        match self {
            PointerType::Pointer { base, space: _ } => context.types[*base],
            PointerType::ScalarPointer { base, space: _ } => (*base).into(),
            PointerType::VectorPointer { base, space: _ } => (*base).into(),
        }
    }
}

impl AsIrTypes for PointerType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        Some(context.pointer_type())
    }

    fn as_ir_type(&self, context: &Context) -> ir::Type {
        context.pointer_type()
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        std::iter::once(context.pointer_type())
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        let _ = context;
        Some(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorType {
    pub size: naga::VectorSize,
    pub scalar: ScalarType,
}

impl VectorType {
    pub fn from_naga(scalar: naga::Scalar, size: naga::VectorSize) -> Result<Self, InvalidType> {
        Ok(Self {
            size,
            scalar: ScalarType::from_naga(scalar)?,
        })
    }

    pub fn with_scalar(self, scalar: ScalarType) -> Self {
        Self {
            size: self.size,
            scalar,
        }
    }
}

impl AsIrTypes for VectorType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        match context.simd_context[*self] {
            VectorIrType::Plain { ty: _ } => None,
            VectorIrType::Vector { ty } => Some(ty),
        }
    }

    fn as_ir_types(&self, context: &Context) -> impl Iterator<Item = ir::Type> {
        match context.simd_context[*self] {
            VectorIrType::Plain { ty } => std::iter::repeat_n(ty, u8::from(self.size).into()),
            VectorIrType::Vector { ty } => std::iter::repeat_n(ty, 1),
        }
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        match context.simd_context[*self] {
            VectorIrType::Plain { ty: _ } => Some(u8::from(self.size).into()),
            VectorIrType::Vector { ty: _ } => Some(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixType {
    pub columns: naga::VectorSize,
    pub rows: naga::VectorSize,
    pub scalar: ScalarType,
}

impl MatrixType {
    pub fn from_naga(
        scalar: naga::Scalar,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Self, InvalidType> {
        Ok(Self {
            columns,
            rows,
            scalar: ScalarType::from_naga(scalar)?,
        })
    }

    pub fn num_elements(&self) -> u8 {
        u8::from(self.columns) * u8::from(self.rows)
    }

    pub fn with_scalar(self, scalar: ScalarType) -> Self {
        Self {
            columns: self.columns,
            rows: self.rows,
            scalar,
        }
    }

    pub fn column_vector(self) -> VectorType {
        VectorType {
            size: self.rows,
            scalar: self.scalar,
        }
    }

    pub fn row_vector(self) -> VectorType {
        VectorType {
            size: self.columns,
            scalar: self.scalar,
        }
    }
}

impl AsIrTypes for MatrixType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        match context.simd_context[*self] {
            MatrixIrType::Plain { ty: _ } | MatrixIrType::ColumnVector { ty: _ } => None,
            MatrixIrType::FullVector { ty } => Some(ty),
        }
    }

    fn as_ir_types(&self, context: &Context) -> impl Iterator<Item = ir::Type> {
        match context.simd_context[*self] {
            MatrixIrType::Plain { ty } => {
                std::iter::repeat_n(
                    ty,
                    usize::from(u8::from(self.columns)) * usize::from(u8::from(self.rows)),
                )
            }
            MatrixIrType::ColumnVector { ty } => {
                std::iter::repeat_n(ty, usize::from(u8::from(self.rows)))
            }
            MatrixIrType::FullVector { ty } => std::iter::repeat_n(ty, 1),
        }
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        match context.simd_context[*self] {
            MatrixIrType::Plain { ty: _ } => {
                Some(usize::from(u8::from(self.columns)) * usize::from(u8::from(self.rows)))
            }
            MatrixIrType::ColumnVector { ty: _ } => Some(usize::from(u8::from(self.rows))),
            MatrixIrType::FullVector { ty: _ } => Some(1),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StructType {
    pub handle: naga::Handle<naga::Type>,
}

impl StructType {
    pub fn members<'module>(&self, source: &'module naga::Module) -> &'module [naga::StructMember] {
        match &source.types[self.handle].inner {
            naga::TypeInner::Struct { members, span: _ } => &*members,
            _ => unreachable!("expected struct"),
        }
    }
}

impl AsIrTypes for StructType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        let _ = context;
        None
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        self.members(&context.source)
            .into_iter()
            .flat_map(|member| {
                let member_type = &context.types[member.ty];
                member_type.as_ir_types(context)
            })
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        self.members(&context.source)
            .into_iter()
            .try_fold(0, |sum, member| {
                let member_type = &context.types[member.ty];
                Some(sum + member_type.try_ir_size(context)?)
            })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrayType {
    pub base_type: naga::Handle<naga::Type>,
    pub size: Option<u32>,
    pub stride: u32,
}

impl ArrayType {
    pub fn expect_size(&self) -> u32 {
        self.size.expect("unexpected unsized array")
    }

    pub fn base_type(&self, context: &Context) -> Type {
        context.types[self.base_type]
    }
}

impl AsIrTypes for ArrayType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        let _ = context;
        None
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        let base_type = &context.types[self.base_type];
        let count = self.expect_size();
        (0..count).flat_map(move |_i| base_type.as_ir_types(context))
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        let base_type = &context.types[self.base_type];
        let count = self.size?;
        let base_size = base_type.try_ir_size(context)?;
        Some(usize::try_from(count).expect("array size overflow") * base_size)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Type {
    Scalar(ScalarType),
    Vector(VectorType),
    Matrix(MatrixType),
    Pointer(PointerType),
    Struct(StructType),
    Array(ArrayType),
}

impl Type {
    pub fn from_naga(
        source: &naga::Module,
        handle: naga::Handle<naga::Type>,
    ) -> Result<Self, InvalidType> {
        let ty = &source.types[handle];
        let output = match &ty.inner {
            naga::TypeInner::Scalar(scalar) => ScalarType::from_naga(*scalar)?.into(),
            naga::TypeInner::Vector { size, scalar } => {
                VectorType::from_naga(*scalar, *size)?.into()
            }
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => MatrixType::from_naga(*scalar, *columns, *rows)?.into(),
            naga::TypeInner::Atomic(_scalar) => todo!("atomic scalar"),
            naga::TypeInner::Pointer { base, space } => {
                PointerType::from_naga(*base, *space)?.into()
            }
            naga::TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => PointerType::from_naga_value(*scalar, *size, *space)?.into(),
            naga::TypeInner::Array { base, size, stride } => {
                let size = match size.resolve(source.to_ctx()) {
                    Ok(naga::proc::IndexableLength::Known(n)) => Some(n),
                    Ok(naga::proc::IndexableLength::Dynamic) => None,
                    Err(error) => panic!("could not resolve array length: {error}"),
                };
                Self::Array(ArrayType {
                    base_type: *base,
                    size,
                    stride: *stride,
                })
            }
            naga::TypeInner::Struct {
                members: _,
                span: _,
            } => Self::Struct(StructType { handle }),
            naga::TypeInner::Image {
                dim: _,
                arrayed: _,
                class: _,
            } => todo!("image type"),
            naga::TypeInner::Sampler { comparison: _ } => todo!("sampler type"),
            naga::TypeInner::AccelerationStructure { vertex_return: _ } => {
                todo!("acceleration structure type")
            }
            naga::TypeInner::RayQuery { vertex_return: _ } => todo!("ray query type"),
            naga::TypeInner::BindingArray { base: _, size: _ } => todo!("binding array type"),
        };

        Ok(output)
    }
}

impl AsIrTypes for Type {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        match self {
            Type::Scalar(scalar_type) => scalar_type.try_as_ir_type(context),
            Type::Vector(vector_type) => vector_type.try_as_ir_type(context),
            Type::Matrix(matrix_type) => matrix_type.try_as_ir_type(context),
            Type::Pointer(pointer_type) => pointer_type.try_as_ir_type(context),
            Type::Struct(_handle) => None,
            Type::Array(_array_type) => None,
        }
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        let output: Box<dyn Iterator<Item = ir::Type>> = match self {
            Type::Scalar(scalar_type) => Box::new(scalar_type.as_ir_types(context)),
            Type::Vector(vector_type) => Box::new(vector_type.as_ir_types(context)),
            Type::Matrix(matrix_type) => Box::new(matrix_type.as_ir_types(context)),
            Type::Pointer(pointer_type) => Box::new(pointer_type.as_ir_types(context)),
            Type::Struct(struct_type) => Box::new(struct_type.as_ir_types(context)),
            Type::Array(array_type) => Box::new(array_type.as_ir_types(context)),
        };
        output
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        match self {
            Type::Scalar(scalar_type) => scalar_type.try_ir_size(context),
            Type::Vector(vector_type) => vector_type.try_ir_size(context),
            Type::Matrix(matrix_type) => matrix_type.try_ir_size(context),
            Type::Pointer(pointer_type) => pointer_type.try_ir_size(context),
            Type::Struct(struct_type) => struct_type.try_ir_size(context),
            Type::Array(array_type) => array_type.try_ir_size(context),
        }
    }
}

impl From<ScalarType> for Type {
    fn from(ty: ScalarType) -> Self {
        Self::Scalar(ty)
    }
}

impl From<PointerType> for Type {
    fn from(ty: PointerType) -> Self {
        Self::Pointer(ty)
    }
}

impl From<VectorType> for Type {
    fn from(ty: VectorType) -> Self {
        Self::Vector(ty)
    }
}

impl From<MatrixType> for Type {
    fn from(ty: MatrixType) -> Self {
        Self::Matrix(ty)
    }
}

impl From<StructType> for Type {
    fn from(ty: StructType) -> Self {
        Self::Struct(ty)
    }
}

impl From<ArrayType> for Type {
    fn from(ty: ArrayType) -> Self {
        Self::Array(ty)
    }
}

/*
#[derive(Clone, Debug)]
pub enum IrTypesIter<'a, 'source> {
    Repeat(std::iter::Repeat<ir::Type>),
    Nested(NestedIrTypesIter<'a, 'source>),
}

impl<'a> From<&'a ir::Type> for IrTypesIter<'a> {
    fn from(value: &'a ir::Type) -> Self {
        Self::Single(std::iter::once(value))
    }
}

impl<'a> From<&'a [ir::Type]> for IrTypesIter<'a> {
    fn from(value: &'a [ir::Type]) -> Self {
        Self::Vector(value.into_iter())
    }
}

impl<'a> Iterator for IrTypesIter<'a> {
    type Item = ir::Type;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IrTypesIter::Single(once) => once.next().copied(),
            IrTypesIter::Vector(iter) => iter.next().copied(),
            IrTypesIter::Nested(values) => values.next(),
        }
    }
}

context: &'a Context<'source>,
        inner: std::slice::Iter<'a, naga::StructMember>,
        current: Option<naga::Handle<
*/
