use std::marker::PhantomData;

use bytemuck::Pod;

use crate::{
    buffer::{
        BufferReadGuard,
        BufferSlice,
    },
    render_pass::primitive::Separated,
};

/// https://gpuweb.github.io/gpuweb/#index-resolution
pub trait IndexResolution<const SEP: bool> {
    type Item;

    fn resolve(&self, index: u32) -> Self::Item;
}

#[derive(Clone, Debug, Default)]
pub struct DirectIndices;

impl IndexResolution<false> for DirectIndices {
    type Item = u32;

    fn resolve(&self, index: u32) -> Self::Item {
        index
    }
}

#[derive(Debug)]
pub struct IndirectIndices<'state, Index> {
    base_vertex: i32,
    index_buffer_guard: BufferReadGuard<'state>,
    _phantom: PhantomData<Index>,
}

impl<'state, Index> IndirectIndices<'state, Index>
where
    Index: Pod,
{
    pub fn lookup(&self, index: u32) -> Index {
        let index_buffer = bytemuck::cast_slice::<u8, Index>(&*self.index_buffer_guard);
        index_buffer[index as usize]
    }
}

impl<'state, Index> IndirectIndices<'state, Index>
where
    u32: From<Index>,
{
    pub fn to_vertex(&self, index: Index) -> u32 {
        u32::from(index).strict_add_signed(self.base_vertex)
    }
}

impl<'state, Index> IndexResolution<false> for IndirectIndices<'state, Index>
where
    Index: Pod,
    u32: From<Index>,
{
    type Item = u32;

    fn resolve(&self, index: u32) -> Self::Item {
        self.to_vertex(self.lookup(index))
    }
}

impl<'state, Index> IndexResolution<true> for IndirectIndices<'state, Index>
where
    Index: Pod + Eq + IndexSeparator,
    u32: From<Index>,
{
    type Item = Separated<u32>;

    fn resolve(&self, index: u32) -> Self::Item {
        let index = self.lookup(index);
        if index.is_separator() {
            Separated::Separator
        }
        else {
            Separated::Vertex(self.to_vertex(index))
        }
    }
}

pub trait IndexSeparator {
    fn is_separator(&self) -> bool;
}

macro_rules! impl_index_separator {
    ($($ty:ty),*) => {
        $(
            impl IndexSeparator for $ty {
                fn is_separator(&self) -> bool {
                    *self == <$ty>::MAX
                }
            }
        )*
    };
}

impl_index_separator!(u16, u32);

#[derive(Debug)]
pub struct IndexBufferBinding {
    pub buffer_slice: BufferSlice,
    pub index_format: wgpu::IndexFormat,
}

impl IndexBufferBinding {
    pub fn begin<Index>(&self, base_vertex: i32) -> IndirectIndices<'_, Index> {
        IndirectIndices {
            base_vertex,
            index_buffer_guard: self.buffer_slice.read(),
            _phantom: PhantomData,
        }
    }
}
