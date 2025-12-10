#![allow(dead_code, unused_variables)]

mod adapter;
mod command;
mod device;
mod engine;
mod instance;
mod queue;
mod render_pass;
#[cfg(feature = "softbuffer")]
mod surface;
mod texture;

pub use adapter::Adapter;
pub use device::Device;
pub use instance::Instance;
pub use queue::QueueSender;
#[cfg(feature = "softbuffer")]
pub use surface::Surface;

pub fn make_label_owned(label: &Option<&str>) -> Option<String> {
    label.map(ToOwned::to_owned)
}

pub const TEXTURE_USAGES: wgpu::TextureUsages = const {
    wgpu::TextureUsages::COPY_SRC
        .union(wgpu::TextureUsages::COPY_DST)
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
};
