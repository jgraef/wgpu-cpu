use cranelift_codegen::ir;
use naga::ScalarKind;

use crate::{
    compiler::Context,
    simd::{
        MatrixIrType,
        VectorIrType,
    },
};

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Expected {expected}, but found {ty:?}")]
pub struct UnexpectedType {
    pub ty: Type,
    pub expected: &'static str,
}

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

    pub fn is_bool(&self) -> bool {
        matches!(self, ScalarType::Bool)
    }

    pub fn is_integer(&self) -> bool {
        matches!(self, ScalarType::Int(_signedness, _int_width))
    }

    pub fn is_float(&self) -> bool {
        matches!(self, ScalarType::Float(_float_width))
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
pub struct PointerType {
    pub base_type: PointerTypeBase,
    pub address_space: naga::AddressSpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointerTypeBase {
    Pointer(naga::Handle<naga::Type>),
    ScalarPointer(ScalarType),
    VectorPointer(VectorType),
}

impl PointerType {
    pub fn from_naga(
        base_type: naga::Handle<naga::Type>,
        address_space: naga::AddressSpace,
    ) -> Result<Self, InvalidType> {
        Ok(Self {
            base_type: PointerTypeBase::Pointer(base_type),
            address_space,
        })
    }

    pub fn from_naga_value(
        scalar: naga::Scalar,
        size: Option<naga::VectorSize>,
        address_space: naga::AddressSpace,
    ) -> Result<Self, InvalidType> {
        let base_type = if let Some(size) = size {
            PointerTypeBase::VectorPointer(VectorType {
                size,
                scalar: ScalarType::from_naga(scalar)?,
            })
        }
        else {
            PointerTypeBase::ScalarPointer(ScalarType::from_naga(scalar)?)
        };

        Ok(Self {
            base_type,
            address_space,
        })
    }

    pub fn base_type(&self, context: &Context) -> Type {
        match self.base_type {
            PointerTypeBase::Pointer(base) => context.types[base],
            PointerTypeBase::ScalarPointer(base) => base.into(),
            PointerTypeBase::VectorPointer(base) => base.into(),
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

    pub fn with_scalar(&self, scalar: ScalarType) -> Self {
        Self {
            columns: self.columns,
            rows: self.rows,
            scalar,
        }
    }

    pub fn column_vector(&self) -> VectorType {
        VectorType {
            size: self.rows,
            scalar: self.scalar,
        }
    }

    pub fn row_vector(&self) -> VectorType {
        VectorType {
            size: self.columns,
            scalar: self.scalar,
        }
    }

    pub fn column_stride(&self) -> u8 {
        u8::from(self.rows).next_power_of_two()
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
pub struct ImageType {
    pub dimension: naga::ImageDimension,
    pub arrayed: bool,
    pub class: naga::ImageClass,
}

impl AsIrTypes for ImageType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        let _ = context;
        None
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        let _ = context;
        std::iter::empty()
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        let _ = context;
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SamplerType {
    pub comparision: bool,
}

impl AsIrTypes for SamplerType {
    fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
        let _ = context;
        None
    }

    fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
        let _ = context;
        std::iter::empty()
    }

    fn try_ir_size(&self, context: &Context) -> Option<usize> {
        let _ = context;
        None
    }
}

macro_rules! define_type {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum Type {
            $($variant($ty),)*
        }

        impl AsIrTypes for Type {
            fn try_as_ir_type(&self, context: &Context) -> Option<ir::Type> {
                match self {
                    $(Self::$variant(ty) => ty.try_as_ir_type(context),)*
                }
            }

            fn as_ir_types<'a>(&'a self, context: &'a Context) -> impl Iterator<Item = ir::Type> + 'a {
                let output: Box<dyn Iterator<Item = ir::Type>> = match self {
                    $(Self::$variant(ty) => Box::new(ty.as_ir_types(context)),)*
                };
                output
            }

            fn try_ir_size(&self, context: &Context) -> Option<usize> {
                match self {
                    $(Self::$variant(ty) => ty.try_ir_size(context),)*
                }
            }
        }

        $(
            impl From<$ty> for Type {
                fn from(ty: $ty) -> Self {
                    Self::$variant(ty)
                }
            }

            impl TryFrom<Type> for $ty {
                type Error = UnexpectedType;

                fn try_from(ty: Type) -> Result<$ty, UnexpectedType> {
                    match ty {
                        Type::$variant(ty) => Ok(ty),
                        _ => Err(UnexpectedType { ty, expected: stringify!($ty)})
                    }
                }
            }
        )*
    };
}

define_type!(
    Scalar(ScalarType),
    Vector(VectorType),
    Matrix(MatrixType),
    Pointer(PointerType),
    Struct(StructType),
    Array(ArrayType),
    Image(ImageType),
    Sampler(SamplerType),
);

impl Type {
    pub fn from_naga(
        source: &naga::Module,
        handle: naga::Handle<naga::Type>,
    ) -> Result<Self, InvalidType> {
        use naga::TypeInner::*;

        let ty = &source.types[handle];
        let output = match &ty.inner {
            Scalar(scalar) => ScalarType::from_naga(*scalar)?.into(),
            Vector { size, scalar } => VectorType::from_naga(*scalar, *size)?.into(),
            Matrix {
                columns,
                rows,
                scalar,
            } => MatrixType::from_naga(*scalar, *columns, *rows)?.into(),
            Atomic(_scalar) => todo!("atomic scalar"),
            Pointer { base, space } => PointerType::from_naga(*base, *space)?.into(),
            ValuePointer {
                size,
                scalar,
                space,
            } => PointerType::from_naga_value(*scalar, *size, *space)?.into(),
            Array { base, size, stride } => {
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
            Struct {
                members: _,
                span: _,
            } => Self::Struct(StructType { handle }),
            Image {
                dim,
                arrayed,
                class,
            } => {
                Self::Image(ImageType {
                    dimension: *dim,
                    arrayed: *arrayed,
                    class: *class,
                })
            }
            Sampler { comparison } => {
                Self::Sampler(SamplerType {
                    comparision: *comparison,
                })
            }
            AccelerationStructure { vertex_return: _ } => {
                todo!("acceleration structure type")
            }
            RayQuery { vertex_return: _ } => todo!("ray query type"),
            BindingArray { base: _, size: _ } => todo!("binding array type"),
            CooperativeMatrix {
                columns: _,
                rows: _,
                scalar: _,
                role: _,
            } => todo!("cooperative matrix type"),
        };

        Ok(output)
    }
}
