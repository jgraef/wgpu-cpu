use std::collections::HashMap;

use naga_cranelift::bindings::BindingResources;
use nalgebra::Point3;

use crate::{
    bind_group::{
        BindGroup,
        BindingResource,
    },
    buffer::BufferReadGuard,
    sampler::Sampler,
    texture::TextureReadGuard,
};

#[derive(Debug)]
pub struct AcquiredBindingResources<'state> {
    bindings: HashMap<naga::ResourceBinding, AcquiredBindingResource<'state>>,
}

impl<'state> AcquiredBindingResources<'state> {
    pub fn new(bind_groups: &'state [Option<BindGroup>]) -> Self {
        let mut bindings = HashMap::new();

        for (group, bind_group) in bind_groups.into_iter().enumerate() {
            if let Some(bind_group) = bind_group {
                for entry in bind_group.entries.iter() {
                    let resource = match &entry.resource {
                        BindingResource::Buffer(buffer_slice) => {
                            AcquiredBindingResource::Buffer(buffer_slice.read())
                        }
                        BindingResource::Sampler(sampler) => {
                            AcquiredBindingResource::Sampler(sampler)
                        }
                        BindingResource::TextureView(texture_view) => {
                            AcquiredBindingResource::TextureView(texture_view.read())
                        }
                    };

                    bindings.insert(
                        naga::ResourceBinding {
                            group: group.try_into().unwrap(),
                            binding: entry.binding,
                        },
                        resource,
                    );
                }
            }
        }

        Self { bindings }
    }
}

// todo: how to make this writable and sharable?
impl<'state> BindingResources for &AcquiredBindingResources<'state> {
    type Image = TextureReadGuard<'state>;
    type Sampler = Sampler;

    fn buffer(&self, binding: naga::ResourceBinding) -> &[u8] {
        let binding = self
            .bindings
            .get(&binding)
            .expect("No such binding resource: {binding:?}");
        match binding {
            AcquiredBindingResource::Buffer(buffer_read_guard) => &*buffer_read_guard,
            _ => panic!("Not a buffer"),
        }
    }

    fn image(&self, binding: naga::ResourceBinding) -> &Self::Image {
        let binding = self
            .bindings
            .get(&binding)
            .expect("No such binding resource: {binding:?}");
        match binding {
            AcquiredBindingResource::TextureView(texture_read_guard) => texture_read_guard,
            _ => panic!("Not a texture view"),
        }
    }

    fn sampler(&self, binding: naga::ResourceBinding) -> &Self::Sampler {
        let binding = self
            .bindings
            .get(&binding)
            .expect("No such binding resource: {binding:?}");
        match binding {
            AcquiredBindingResource::Sampler(sampler) => sampler,
            _ => panic!("Not a sampler"),
        }
    }

    fn image_sample(
        &mut self,
        image: &TextureReadGuard<'state>,
        sampler: &Sampler,
        gather: Option<naga::SwizzleComponent>,
        coordinate: [f32; 2],
        array_index: Option<u32>,
        offset: Option<u32>,
        level: naga::SampleLevel,
        depth_ref: Option<f32>,
        clamp_to_edge: bool,
    ) -> [f32; 4] {
        if gather.is_some() {
            todo!("image_sample: gather");
        }
        if array_index.is_some() {
            todo!("image_sample: array_index")
        }
        if offset.is_some() {
            todo!("image_sample: offset");
        }
        if !matches!(level, naga::SampleLevel::Auto) {
            todo!("image_sample: sample level");
        }
        if depth_ref.is_some() {
            todo!("image_sample: depth_ref");
        }
        if clamp_to_edge {
            todo!("image_sample: clamp_to_edge");
        }

        let uvw = Point3::new(
            texel_coordinate(
                coordinate[0],
                sampler.descriptor.address_mode_u,
                image.info.size.width,
            ),
            texel_coordinate(
                coordinate[1],
                sampler.descriptor.address_mode_v,
                image.info.size.height,
            ),
            0,
        );

        let texel = image.get_pixel(uvw);
        texel.into()
    }
}

#[derive(Debug)]
pub enum AcquiredBindingResource<'state> {
    Buffer(BufferReadGuard<'state>),
    Sampler(&'state Sampler),
    TextureView(TextureReadGuard<'state>),
}

fn texel_coordinate(x: f32, address_mode: wgpu::AddressMode, size: u32) -> u32 {
    use wgpu::AddressMode::*;

    let x = match address_mode {
        ClampToEdge => x.clamp(0.0, 1.0),
        Repeat => x.rem_euclid(1.0),
        MirrorRepeat => {
            let x = x.rem_euclid(2.0);
            if x <= 1.0 { x } else { 2.0 - x }
        }
        ClampToBorder => todo!("ClampToBorder"),
    };

    (x * (size - 1) as f32).round() as u32
}
