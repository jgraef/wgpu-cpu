use std::{
    fmt::Debug,
    ops::{
        Add,
        RangeBounds,
    },
};

use bytemuck::{
    Pod,
    Zeroable,
};
use naga::{
    AddressSpace,
    proc::Alignment,
};

use crate::shader::{
    ShaderModuleInner,
    bindings::BindingAddress,
    interpreter::{
        Variable,
        VariableType,
    },
};

pub trait ReadMemory<A> {
    fn read(&self, address: A) -> &[u8];
}

impl<T, A> ReadMemory<A> for &T
where
    T: ReadMemory<A>,
{
    fn read(&self, address: A) -> &[u8] {
        T::read(self, address)
    }
}

impl<T, A> ReadMemory<A> for &mut T
where
    T: ReadMemory<A>,
{
    fn read(&self, address: A) -> &[u8] {
        T::read(self, address)
    }
}

pub trait WriteMemory<A> {
    fn write(&mut self, address: A) -> &mut [u8];
}

impl<T, A> WriteMemory<A> for &mut T
where
    T: WriteMemory<A>,
{
    fn write(&mut self, address: A) -> &mut [u8] {
        T::write(self, address)
    }
}

pub trait ReadWriteMemory<A>: ReadMemory<A> + WriteMemory<A> {
    fn copy(&mut self, source: A, target: A);
}

impl<T, A> ReadWriteMemory<A> for &mut T
where
    T: ReadWriteMemory<A>,
{
    fn copy(&mut self, source: A, target: A) {
        T::copy(self, source, target)
    }
}

#[derive(Clone, Debug)]
pub struct Memory<B> {
    pub stack: Stack,
    pub bindings: B,
}

impl<B> Memory<B> {
    pub fn stack_frame(&mut self) -> StackFrame<'_, B> {
        let start = self.stack.data.len();
        StackFrame {
            memory: self,
            start,
        }
    }
}

impl<B> ReadMemory<Slice> for Memory<B>
where
    B: ReadMemory<BindingAddress>,
{
    fn read(&self, address: Slice) -> &[u8] {
        match address {
            Slice::Stack(slice) => self.stack.read(slice),
            Slice::Binding(binding) => self.bindings.read(binding),
        }
    }
}

impl<B> WriteMemory<Slice> for Memory<B>
where
    B: WriteMemory<BindingAddress>,
{
    fn write(&mut self, address: Slice) -> &mut [u8] {
        match address {
            Slice::Stack(slice) => self.stack.write(slice),
            Slice::Binding(binding) => self.bindings.write(binding),
        }
    }
}

impl<B> ReadWriteMemory<Slice> for Memory<B>
where
    B: ReadWriteMemory<BindingAddress>,
{
    fn copy(&mut self, source: Slice, target: Slice) {
        match (source, target) {
            (Slice::Stack(source), Slice::Stack(target)) => {
                self.stack.copy(source, target);
            }
            (Slice::Stack(source), Slice::Binding(target)) => {
                copy(&self.stack, source, &mut self.bindings, target)
            }
            (Slice::Binding(source), Slice::Stack(target)) => {
                copy(&self.bindings, source, &mut self.stack, target)
            }
            (Slice::Binding(source), Slice::Binding(target)) => self.bindings.copy(source, target),
        }
    }
}

pub fn copy<Source, SourceAddress, Target, TargetAddress>(
    source: &Source,
    source_address: SourceAddress,
    target: &mut Target,
    target_address: TargetAddress,
) where
    Source: ReadMemory<SourceAddress>,
    Target: WriteMemory<TargetAddress>,
{
    let source = source.read(source_address);
    let target = target.write(target_address);
    target.copy_from_slice(source);
}

#[derive(Clone, Copy, Debug)]
pub enum Slice {
    Stack(StackSlice),
    Binding(BindingAddress),
}

impl From<StackSlice> for Slice {
    fn from(value: StackSlice) -> Self {
        Self::Stack(value)
    }
}

impl Slice {
    #[track_caller]
    pub fn slice<R, T>(&self, range: R) -> Self
    where
        R: RangeBounds<T>,
        T: Offset,
    {
        let start = match range.start_bound() {
            std::ops::Bound::Included(i) => i.to_usize(),
            std::ops::Bound::Excluded(i) => i.to_usize() + 1,
            std::ops::Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            std::ops::Bound::Included(i) => Some(i.to_usize() + 1),
            std::ops::Bound::Excluded(i) => Some(i.to_usize()),
            std::ops::Bound::Unbounded => None,
        };

        match self {
            Slice::Stack(slice) => slice.slice_impl(start, end).into(),
            Slice::Binding(_binding) => todo!(),
        }
    }
}

pub trait Offset {
    fn to_usize(&self) -> usize;
}

impl Offset for usize {
    fn to_usize(&self) -> usize {
        *self
    }
}

impl Offset for u32 {
    fn to_usize(&self) -> usize {
        *self as usize
    }
}

#[derive(Clone, derive_more::Debug)]
pub struct Stack {
    #[debug("[... {} bytes]", self.data.len())]
    data: Vec<u8>,
    limit: usize,
}

impl Stack {
    pub fn new(limit: u32) -> Self {
        Self {
            data: vec![],
            limit: limit.try_into().unwrap(),
        }
    }

    pub fn allocate(&mut self, size: u32, alignment: Alignment) -> StackSlice {
        let size = size as usize;
        let offset = alignment.round_up(self.data.len() as u32) as usize;
        self.data.resize(offset + size, 0);
        StackSlice { offset, len: size }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn allocated(&self) -> usize {
        self.data.capacity()
    }
}

impl ReadMemory<StackSlice> for Stack {
    fn read(&self, address: StackSlice) -> &[u8] {
        &self.data[address.offset..][..address.len]
    }
}

impl WriteMemory<StackSlice> for Stack {
    fn write(&mut self, address: StackSlice) -> &mut [u8] {
        &mut self.data[address.offset..][..address.len]
    }
}

impl ReadWriteMemory<StackSlice> for Stack {
    fn copy(&mut self, source: StackSlice, target: StackSlice) {
        self.data
            .copy_within(source.offset..(source.offset + source.len), target.offset);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StackSlice {
    offset: usize,
    len: usize,
}

impl StackSlice {
    #[track_caller]
    fn slice_impl(&self, start: usize, end: Option<usize>) -> Self {
        if let Some(end) = end {
            assert!(start <= end);
            assert!(end <= self.len);
            Self {
                offset: self.offset + start,
                len: end - start,
            }
        }
        else {
            assert!(start <= self.len);
            Self {
                offset: self.offset + start,
                len: self.len - start,
            }
        }
    }
}

impl Add<u32> for StackSlice {
    type Output = StackSlice;

    fn add(self, rhs: u32) -> Self::Output {
        let rhs = rhs as usize;
        assert!(rhs <= self.len);
        let mut output = self;
        output.offset += rhs;
        output.len -= rhs;
        output
    }
}

#[derive(Debug)]
pub struct StackFrame<'a, B> {
    pub memory: &'a mut Memory<B>,
    pub start: usize,
}

impl<'a, B> StackFrame<'a, B> {
    pub fn frame(&mut self) -> StackFrame<'_, B> {
        self.memory.stack_frame()
    }

    pub fn allocate_variable<'ty>(
        &mut self,
        ty: impl Into<VariableType<'ty>>,
        module: &ShaderModuleInner,
    ) -> Variable<'ty> {
        let ty = ty.into();
        let type_layout = module.type_layout(ty);

        let slice = self
            .memory
            .stack
            .allocate(type_layout.size, type_layout.alignment);
        Variable {
            ty,
            slice: slice.into(),
        }
    }
}

impl<'a, B> Drop for StackFrame<'a, B> {
    fn drop(&mut self) {
        assert!(self.start <= self.memory.stack.data.len());
        self.memory.stack.data.resize(self.start, 0);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullMemory;

impl<A> ReadMemory<A> for NullMemory
where
    A: Debug,
{
    fn read(&self, address: A) -> &[u8] {
        panic!("Attempt to read from NullMemory: {address:?}");
    }
}

impl<A> WriteMemory<A> for NullMemory
where
    A: Debug,
{
    fn write(&mut self, address: A) -> &mut [u8] {
        panic!("Attempt to write to NullMemory: {address:?}",);
    }
}

impl<A> ReadWriteMemory<A> for NullMemory
where
    A: Debug,
{
    fn copy(&mut self, source: A, target: A) {
        panic!("Attempt to copy in NullMemory: From {source:?} to {target:?}");
    }
}

// Do wgsl pointers have to be 32 bit? naga returns that size for a pointer type
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
#[repr(C)]
pub struct Pointer(u32);

impl Pointer {
    pub fn deref(&self, address_space: AddressSpace, len: u32) -> Slice {
        match address_space {
            AddressSpace::Function => {
                Slice::Stack(StackSlice {
                    offset: self.0 as usize,
                    len: len as usize,
                })
            }
            AddressSpace::Private => todo!(),
            AddressSpace::WorkGroup => todo!(),
            AddressSpace::Uniform => todo!(),
            AddressSpace::Storage { access } => todo!(),
            AddressSpace::Handle => todo!(),
            AddressSpace::Immediate => todo!(),
            AddressSpace::TaskPayload => todo!(),
        }
    }
}

impl From<Slice> for Pointer {
    fn from(value: Slice) -> Self {
        match value {
            Slice::Stack(stack_slice) => Self::from(stack_slice),
            Slice::Binding(binding_address) => todo!(),
        }
    }
}

impl From<StackSlice> for Pointer {
    fn from(value: StackSlice) -> Self {
        Self(value.offset.try_into().unwrap())
    }
}
