use std::fmt::Debug;

use bytemuck::Pod;
use half::f16;
use naga::{
    Handle,
    Scalar,
    ScalarKind,
    Type,
    TypeInner,
    proc::{
        TypeLayout,
        TypeResolution,
    },
};

use crate::{
    interpreter::{
        memory::{
            Pointer,
            Slice,
            StackFrame,
        },
        module::InterpretedModule,
    },
    memory::{
        ReadMemory,
        ReadWriteMemory,
        WriteMemory,
    },
};

// todo: this could always contain a `&'a TypeInner` because we have the
// life-time anyway and we can make the lookup at construction
#[derive(Clone, Copy, Debug)]
pub enum VariableType<'a> {
    Handle(Handle<Type>),
    Inner(&'a TypeInner),
}

impl<'a> VariableType<'a> {
    pub fn inner_with(&self, module: &'a InterpretedModule) -> &'a TypeInner {
        match self {
            VariableType::Handle(handle) => &module.inner.module.types[*handle].inner,
            VariableType::Inner(type_inner) => *type_inner,
        }
    }
}

impl From<Handle<Type>> for VariableType<'static> {
    fn from(value: Handle<Type>) -> Self {
        Self::Handle(value)
    }
}

impl<'a> From<&'a Handle<Type>> for VariableType<'static> {
    fn from(value: &'a Handle<Type>) -> Self {
        Self::from(*value)
    }
}

impl<'a> From<&'a TypeInner> for VariableType<'a> {
    fn from(value: &'a TypeInner) -> Self {
        Self::Inner(value)
    }
}

impl<'a> From<&'a TypeResolution> for VariableType<'a> {
    fn from(value: &'a TypeResolution) -> Self {
        match value {
            TypeResolution::Handle(handle) => Self::Handle(*handle),
            TypeResolution::Value(type_inner) => Self::Inner(type_inner),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Variable<'module, 'ty> {
    pub(super) module: &'module InterpretedModule,
    pub(super) ty: VariableType<'ty>,
    pub(super) slice: Slice,
}

impl<'module, 'ty> Variable<'module, 'ty> {
    pub fn allocate<'memory, B>(
        ty: impl Into<VariableType<'ty>>,
        module: &'module InterpretedModule,
        stack_frame: &mut StackFrame<'memory, B>,
    ) -> Variable<'module, 'ty> {
        let ty = ty.into();
        let type_layout = module.type_layout(ty);

        let slice = stack_frame.memory.stack.allocate(type_layout);

        Self {
            module,
            ty,
            slice: slice.into(),
        }
    }

    pub fn slice(&self) -> Slice {
        self.slice
    }

    pub fn read<'m, T, M>(&self, memory: &'m M) -> &'m T
    where
        T: Pod,
        M: ReadMemory<Slice>,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes(&memory.read(self.slice.slice(..n)))
    }

    pub fn write<'m, T, M>(&self, memory: &'m mut M) -> &'m mut T
    where
        T: Pod,
        M: WriteMemory<Slice>,
    {
        let n = std::mem::size_of::<T>();
        bytemuck::from_bytes_mut(memory.write(self.slice.slice(..n)))
    }

    pub fn copy_from<M>(&self, source: Variable, memory: &mut M)
    where
        M: ReadWriteMemory<Slice>,
    {
        memory.copy(source.slice, self.slice);
    }

    pub fn cast<'ty2>(&self, ty: impl Into<VariableType<'ty2>>) -> Variable<'module, 'ty2> {
        Variable {
            module: self.module,
            ty: ty.into(),
            slice: self.slice,
        }
    }

    pub fn debug<'a, M>(&'a self, memory: &'a M) -> VariableDebug<'a, M>
    where
        M: ReadMemory<Slice>,
    {
        VariableDebug {
            variable: *self,
            module: self.module,
            memory,
        }
    }

    pub fn component<'ty2>(
        &self,
        index: usize,
        component_ty: impl Into<VariableType<'ty2>>,
    ) -> Variable<'module, 'ty2> {
        let component_ty = component_ty.into();
        let offset = self.module.offset_of(self.ty, component_ty, index);
        let size = self.module.size_of(component_ty);
        Variable {
            module: self.module,
            ty: component_ty,
            slice: self.slice.slice(offset..offset + size),
        }
    }

    pub fn pointer(&self) -> Pointer {
        Pointer::from(self.slice)
    }

    pub fn try_deref<M>(&self, memory: &M) -> Option<Variable<'module, 'static>>
    where
        M: ReadMemory<Slice>,
    {
        let ty_inner = self.ty.inner_with(self.module);
        match ty_inner {
            TypeInner::Pointer { base, space } => {
                let ty = base.into();
                let pointer = self.read::<Pointer, M>(memory);
                let size = self.module.size_of(ty);
                let slice = pointer.deref(*space, size);

                Some(Variable {
                    module: self.module,
                    ty,
                    slice,
                })
            }
            TypeInner::ValuePointer {
                size: _,
                scalar: _,
                space: _,
            } => {
                /*let ty = scalar.into();
                let pointer = self.read::<Pointer, M>(memory);
                let size = module.size_of(ty);
                let slice = pointer.deref(*space, size);

                Some(Variable { ty, slice })*/
                todo!("pain!");
            }
            _ => None,
        }
    }

    pub fn write_zero<M>(&self, memory: &mut M)
    where
        M: WriteMemory<Slice>,
    {
        let target = memory.write(self.slice);
        target.fill(0);
    }

    pub fn type_layout(&self) -> TypeLayout {
        self.module.type_layout(self.ty)
    }

    pub fn size_of(&self) -> u32 {
        self.module.size_of(self.ty)
    }
}

#[derive(Clone, Copy)]
pub struct VariableDebug<'a, M> {
    variable: Variable<'a, 'a>,
    module: &'a InterpretedModule,
    memory: &'a M,
}

impl<'a, M> VariableDebug<'a, M>
where
    M: ReadMemory<Slice>,
{
    fn write_scalar(
        &self,
        variable: Variable,
        scalar: &Scalar,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        macro_rules! write_scalar {
                ($(($scalar:ident @ $width:pat) as $ty:ty;)*) => {
                    match (
                        scalar.kind,
                        scalar.width,
                    ) {
                        $(
                            (ScalarKind::$scalar, $width) => {
                                let input = variable.read::<$ty, M>(self.memory);
                                write!(f, "{input:?}")?;
                            }
                        )*
                        _ => {
                            write!(f, "(?{scalar:?})")?;
                        },
                    }
                };
            }

        write_scalar!(
            (Sint@4) as i32;
            (Uint@4) as u32;
            (Float@4) as f32;
            (Float@2) as f16;
        );

        Ok(())
    }
}

impl<'a, M> Debug for VariableDebug<'a, M>
where
    M: ReadMemory<Slice>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        #![allow(unused_variables)]

        let ty = self.variable.ty.inner_with(&self.module);

        match ty {
            TypeInner::Scalar(scalar) => self.write_scalar(self.variable, scalar, f)?,
            TypeInner::Vector { size, scalar } => {
                let component_ty = TypeInner::Scalar(*scalar);
                write!(f, "[")?;
                for i in 0..(*size as u8) {
                    if i > 0 {
                        write!(f, ", ")?;
                    }

                    let component = self.variable.component(i as usize, &component_ty);

                    self.write_scalar(component, scalar, f)?;
                }
                write!(f, "]")?;
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => todo!(),
            TypeInner::Atomic(scalar) => todo!(),
            TypeInner::Pointer { base, space } => todo!(),
            TypeInner::ValuePointer {
                size,
                scalar,
                space,
            } => todo!(),
            TypeInner::Array { base, size, stride } => todo!(),
            TypeInner::Struct { members, span } => todo!(),
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => todo!(),
            TypeInner::Sampler { comparison } => todo!(),
            TypeInner::AccelerationStructure { vertex_return } => todo!(),
            TypeInner::RayQuery { vertex_return } => todo!(),
            TypeInner::BindingArray { base, size } => todo!(),
        }

        write!(f, " ({ty:?} @ {:?})", self.variable.slice)?;

        Ok(())
    }
}
