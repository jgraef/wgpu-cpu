use wgpu::wgt;

#[derive(Clone, Debug)]
pub struct Sampler {
    pub descriptor: wgt::SamplerDescriptor<Option<String>>,
}

impl Sampler {
    pub fn new(descriptor: &wgpu::SamplerDescriptor) -> Self {
        Self {
            descriptor: wgt::SamplerDescriptor {
                label: descriptor.label.map(ToOwned::to_owned),
                address_mode_u: descriptor.address_mode_u,
                address_mode_v: descriptor.address_mode_v,
                address_mode_w: descriptor.address_mode_w,
                mag_filter: descriptor.mag_filter,
                min_filter: descriptor.min_filter,
                mipmap_filter: descriptor.mipmap_filter,
                lod_min_clamp: descriptor.lod_min_clamp,
                lod_max_clamp: descriptor.lod_max_clamp,
                compare: descriptor.compare,
                anisotropy_clamp: descriptor.anisotropy_clamp,
                border_color: descriptor.border_color,
            },
        }
    }
}

impl wgpu::custom::SamplerInterface for Sampler {}
