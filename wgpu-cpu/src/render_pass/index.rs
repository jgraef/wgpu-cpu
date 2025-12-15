use std::marker::PhantomData;

use bytemuck::Pod;

use crate::buffer::{
    BufferReadGuard,
    BufferSlice,
};

/// https://gpuweb.github.io/gpuweb/#index-resolution
pub trait IndexResolution {
    fn resolve(&self, index: u32) -> u32;
}

#[derive(Clone, Debug, Default)]
pub struct DirectIndices;

impl IndexResolution for DirectIndices {
    fn resolve(&self, index: u32) -> u32 {
        index
    }
}

#[derive(Debug)]
pub struct IndirectIndices<'state, IndexFormat> {
    base_vertex: i32,
    index_buffer_guard: BufferReadGuard<'state>,
    _phantom: PhantomData<[IndexFormat]>,
}

impl<'state, IndexFormat> IndexResolution for IndirectIndices<'state, IndexFormat>
where
    IndexFormat: Pod,
    u32: From<IndexFormat>,
{
    fn resolve(&self, index: u32) -> u32 {
        let index_buffer = bytemuck::cast_slice::<u8, IndexFormat>(&*self.index_buffer_guard);
        u32::from(index_buffer[index as usize]).strict_add_signed(self.base_vertex)
    }
}

#[derive(Debug)]
pub struct IndexBufferBinding {
    pub buffer_slice: BufferSlice,
    pub index_format: wgpu::IndexFormat,
}

impl IndexBufferBinding {
    pub fn begin<IndexFormat>(&self, base_vertex: i32) -> IndirectIndices<'_, IndexFormat> {
        IndirectIndices {
            base_vertex,
            index_buffer_guard: self.buffer_slice.read(),
            _phantom: PhantomData,
        }
    }
}
