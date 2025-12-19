use std::{
    ops::{
        Index,
        Range,
    },
    sync::Arc,
};

use naga::{
    Binding,
    FunctionArgument,
    FunctionResult,
    Handle,
    Type,
    TypeInner,
    UniqueArena,
};

use crate::memory::{
    ReadMemory,
    WriteMemory,
};

#[derive(Clone, Copy, Debug)]
pub struct BindingAddress {
    _todo: (),
}

pub trait ShaderInput {
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]);
}

impl<T> ShaderInput for &T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

impl<T> ShaderInput for &mut T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

pub trait ShaderOutput {
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]);
}

impl<T> ShaderOutput for &mut T
where
    T: ShaderOutput,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        T::read_from(self, binding, ty, source);
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From, derive_more::Into,
)]
pub struct BindingLocation(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct BindingLocationLayout {
    pub offset: u32,
    pub size: u32,
}

impl BindingLocationLayout {
    pub fn range(&self) -> Range<usize> {
        let start = self.offset as usize;
        let end = (self.offset + self.size) as usize;
        start..end
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageLayout {
    pub(super) locations: Arc<[Option<BindingLocationLayout>]>,
    pub(super) size: u32,
}

impl UserDefinedInterStageLayout {
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl FromIterator<Option<BindingLocationLayout>> for UserDefinedInterStageLayout {
    fn from_iter<T: IntoIterator<Item = Option<BindingLocationLayout>>>(iter: T) -> Self {
        let mut size = None;
        let locations = iter
            .into_iter()
            .inspect(|location| {
                if let Some(location) = location {
                    let update_size: &mut u32 = size.get_or_insert_default();
                    *update_size = (*update_size).max(location.offset + location.size);
                }
            })
            .collect();

        Self {
            locations,
            size: size.unwrap_or_default(),
        }
    }
}

impl Index<BindingLocation> for UserDefinedInterStageLayout {
    type Output = Option<BindingLocationLayout>;

    fn index(&self, index: BindingLocation) -> &Self::Output {
        let index: usize = index.0.try_into().unwrap();
        &self.locations[index]
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageBuffer {
    data: Vec<u8>,
    layout: UserDefinedInterStageLayout,
}

impl UserDefinedInterStageBuffer {
    pub fn new(layout: UserDefinedInterStageLayout) -> Self {
        let data = vec![0; layout.size as usize];
        Self { data, layout }
    }
}

impl ReadMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn read(&self, address: BindingLocation) -> &[u8] {
        self.layout[address]
            .as_ref()
            .map_or(&[], |layout| &self.data[layout.range()])
    }
}

impl WriteMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn write(&mut self, address: BindingLocation) -> &mut [u8] {
        self.layout[address]
            .as_ref()
            .map_or(&mut [], |layout| &mut self.data[layout.range()])
    }
}

pub trait VisitIoBindings {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    );
}

impl<T> VisitIoBindings for &mut T
where
    T: VisitIoBindings,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        T::visit(self, binding, ty_handle, ty, offset, name, top_level);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FnVisitIoBindings<F>(pub F);

impl<F> VisitIoBindings for FnVisitIoBindings<F>
where
    F: FnMut(&Binding, Handle<Type>, &Type, u32, Option<&str>, bool),
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        (self.0)(binding, ty_handle, ty, offset, name, top_level)
    }
}

#[derive(Debug)]
pub struct IoBindingVisitor<'a, B> {
    pub types: &'a UniqueArena<Type>,
    pub visit: B,
}

impl<'a, B> IoBindingVisitor<'a, B> {
    pub fn new(types: &'a UniqueArena<Type>, visit: B) -> Self {
        Self { types, visit }
    }
}

impl<'a, F> IoBindingVisitor<'a, FnVisitIoBindings<F>> {
    pub fn from_fn(types: &'a UniqueArena<Type>, f: F) -> Self {
        IoBindingVisitor::new(types, FnVisitIoBindings(f))
    }
}

impl<'a, B> IoBindingVisitor<'a, B>
where
    B: VisitIoBindings,
{
    pub fn visit(
        &mut self,
        binding: Option<&Binding>,
        ty_handle: Handle<Type>,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        let ty = &self.types[ty_handle];
        if let Some(binding) = binding {
            self.visit
                .visit(binding, ty_handle, ty, offset, name, top_level);
        }
        else {
            match &ty.inner {
                TypeInner::Struct { members, span: _ } => {
                    for member in members {
                        self.visit(
                            member.binding.as_ref(),
                            member.ty,
                            offset + member.offset,
                            member.name.as_deref(),
                            false,
                        );
                    }
                }
                _ => panic!("Invalid binding type: {:?}", ty),
            }
        }
    }

    pub fn visit_function_argument(&mut self, argument: &FunctionArgument, offset: u32) {
        self.visit(
            argument.binding.as_ref(),
            argument.ty,
            offset,
            argument.name.as_deref(),
            true,
        );
    }

    pub fn visit_function_result(&mut self, result: &FunctionResult, offset: u32) {
        self.visit(result.binding.as_ref(), result.ty, offset, None, true);
    }
}
