use std::{
    collections::HashMap,
    ops::Bound,
    sync::Arc,
};

use crate::buffer::{
    Buffer,
    BufferSlice,
};

#[derive(Clone, Debug)]
pub struct BindGroupLayout {
    inner: Arc<BindGroupLayoutInner>,
}

impl BindGroupLayout {
    pub fn new(descriptor: &wgpu::BindGroupLayoutDescriptor) -> Self {
        let mut entries = HashMap::with_capacity(descriptor.entries.len());

        for entry in descriptor.entries {
            if let Some(_duplicate) = entries.insert(entry.binding, *entry) {
                panic!(
                    "Duplicate bind group layout entry label={:?}, binding={}",
                    descriptor.label, entry.binding
                );
            }
        }

        Self {
            inner: Arc::new(BindGroupLayoutInner {
                label: descriptor.label.map(ToOwned::to_owned),
                entries,
            }),
        }
    }
}

impl wgpu::custom::BindGroupLayoutInterface for BindGroupLayout {}

#[derive(Debug)]
pub struct BindGroupLayoutInner {
    label: Option<String>,
    entries: HashMap<u32, wgpu::BindGroupLayoutEntry>,
}

#[derive(Clone, Debug)]
pub struct BindGroup {
    pub layout: BindGroupLayout,
    pub entries: Arc<[BindGroupEntry]>,
}

impl BindGroup {
    pub fn new(desc: &wgpu::BindGroupDescriptor) -> Self {
        let layout = desc.layout.as_custom::<BindGroupLayout>().unwrap().clone();
        let entries = desc
            .entries
            .iter()
            .map(|entry| {
                BindGroupEntry {
                    binding: entry.binding,
                    resource: BindingResource::new(&entry.resource),
                }
            })
            .collect();

        Self { layout, entries }
    }
}

impl wgpu::custom::BindGroupInterface for BindGroup {}

#[derive(Debug)]
pub struct BindGroupEntry {
    pub binding: u32,
    pub resource: BindingResource,
}

#[derive(Debug)]
pub enum BindingResource {
    Buffer(BufferSlice),
}

impl BindingResource {
    pub fn new(resource: &wgpu::BindingResource) -> Self {
        match resource {
            wgpu::BindingResource::Buffer(buffer_binding) => {
                let buffer = buffer_binding.buffer.as_custom::<Buffer>().unwrap();
                let buffer_slice = buffer.slice((
                    Bound::Included(usize::try_from(buffer_binding.offset).unwrap()),
                    buffer_binding.size.map_or(Bound::Unbounded, |size| {
                        Bound::Excluded(usize::try_from(size.get()).unwrap())
                    }),
                ));
                Self::Buffer(buffer_slice)
            }
            wgpu::BindingResource::BufferArray(buffer_bindings) => todo!(),
            wgpu::BindingResource::Sampler(sampler) => todo!(),
            wgpu::BindingResource::SamplerArray(samplers) => todo!(),
            wgpu::BindingResource::TextureView(texture_view) => todo!(),
            wgpu::BindingResource::TextureViewArray(texture_views) => todo!(),
            wgpu::BindingResource::AccelerationStructure(tlas) => todo!(),
            wgpu::BindingResource::ExternalTexture(external_texture) => todo!(),
            _ => todo!(),
        }
    }
}
