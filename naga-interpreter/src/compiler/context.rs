use cranelift_codegen::{
    ir::{
        AbiParam,
        Type,
        types,
    },
    isa::{
        CallConv,
        TargetFrontendConfig,
    },
};

use crate::compiler::{
    Error,
    util::lanes_for_vector_size,
};

#[derive(derive_more::Debug)]
pub struct Context<'source> {
    pub source: &'source naga::Module,
    #[allow(unused)]
    pub info: &'source naga::valid::ModuleInfo,
    pub layouter: naga::proc::Layouter,
    #[debug(skip)]
    pub target_config: TargetFrontendConfig,
}

impl<'source> Context<'source> {
    pub fn abi_param(&self, ty: &naga::TypeInner) -> Result<AbiParam, Error> {
        Ok(AbiParam::new(self.abi_ty(ty)?))
    }

    pub fn abi_ty(&self, ty: &naga::TypeInner) -> Result<Type, Error> {
        // structs have to be lowered for the ABI. we can just pass a
        // pointer or struct offset? wgsl function arguments are not
        // variables. they can only be loaded via the FunctionCall
        // expression, so we don't have to worry about COW or anything
        // like that.
        //
        // https://users.rust-lang.org/t/help-trying-to-transfer-a-structure-from-cranelift-to-rust/106429/4

        use naga::TypeInner::*;

        let ty = match ty {
            Scalar(scalar) => self.scalar_type(*scalar)?,
            Vector { size, scalar } => self.vector_type(*scalar, *size)?,
            Matrix {
                columns,
                rows,
                scalar,
            } => self.matrix_type(*scalar, *columns, *rows)?,
            Atomic(scalar) => self.scalar_type(*scalar)?,
            Pointer { base: _, space: _ } => self.target_config.pointer_type(),
            ValuePointer {
                size: _,
                scalar: _,
                space: _,
            } => self.target_config.pointer_type(),
            Array {
                base: _,
                size: _,
                stride: _,
            } => self.target_config.pointer_type(),
            Struct {
                members: _,
                span: _,
            } => self.target_config.pointer_type(),
            Image {
                dim: _,
                arrayed: _,
                class: _,
            } => self.target_config.pointer_type(),
            Sampler { comparison: _ } => self.target_config.pointer_type(),
            AccelerationStructure { vertex_return: _ } => self.target_config.pointer_type(),
            RayQuery { vertex_return: _ } => self.target_config.pointer_type(),
            BindingArray { base: _, size: _ } => self.target_config.pointer_type(),
        };

        Ok(ty)
    }

    pub fn calling_convention(&self) -> CallConv {
        self.target_config.default_call_conv
    }

    pub fn pointer_type(&self) -> Type {
        self.target_config.pointer_type()
    }

    pub fn scalar_type(&self, scalar: naga::Scalar) -> Result<Type, Error> {
        use naga::ScalarKind::*;

        match scalar.kind {
            Sint | Uint => {
                match scalar.width {
                    4 => Some(types::I32),
                    _ => None,
                }
            }
            Float => {
                match scalar.width {
                    2 => Some(types::F16),
                    4 => Some(types::F32),
                    _ => None,
                }
            }
            Bool => {
                match scalar.width {
                    1 => Some(types::I8),
                    _ => None,
                }
            }
            AbstractInt | AbstractFloat => {
                panic!("Abstract types must not appear in naga IR")
            }
        }
        .ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Scalar(scalar),
            }
        })
    }

    pub fn vector_type(&self, scalar: naga::Scalar, size: naga::VectorSize) -> Result<Type, Error> {
        let lane = self.scalar_type(scalar)?;
        lane.by(lanes_for_vector_size(size)).ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Vector { size, scalar },
            }
        })
    }

    pub fn matrix_type(
        &self,
        scalar: naga::Scalar,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Type, Error> {
        let lane = self.scalar_type(scalar)?;
        let column_lanes = match columns {
            naga::VectorSize::Bi => 2,
            naga::VectorSize::Tri => 4, // this is intentional, for alignment
            naga::VectorSize::Quad => 4,
        };
        let row_lanes = u32::from(rows);
        lane.by(column_lanes * row_lanes).ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                },
            }
        })
    }
}
