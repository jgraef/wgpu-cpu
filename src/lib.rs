#![allow(dead_code, unused_variables)]

mod adapter;
mod bind_group;
mod buffer;
mod command;
mod device;
mod engine;
mod instance;
mod pipeline;
mod render_pass;
mod shader;
#[cfg(feature = "softbuffer")]
mod surface;
mod sync;
mod texture;

#[cfg(feature = "image")]
pub use image::dump_texture;
pub use instance::Instance;

pub fn make_label_owned(label: &Option<&str>) -> Option<String> {
    label.map(ToOwned::to_owned)
}

pub const TEXTURE_USAGES: wgpu::TextureUsages = const {
    wgpu::TextureUsages::COPY_SRC
        .union(wgpu::TextureUsages::COPY_DST)
        .union(wgpu::TextureUsages::TEXTURE_BINDING)
        .union(wgpu::TextureUsages::RENDER_ATTACHMENT)
};

// todo: this is for testing. a better interface would be nice
#[cfg(feature = "image")]
mod image {
    use std::path::Path;

    use image::{
        ImageBuffer,
        ImageError,
        Rgba,
    };

    use crate::texture::Texture;

    pub fn dump_texture(texture: &wgpu::Texture, path: impl AsRef<Path>) -> Result<(), ImageError> {
        let size = texture.size();
        if texture.format() != wgpu::TextureFormat::Rgba8UnormSrgb {
            todo!();
        }

        let texture = texture.as_custom::<Texture>().unwrap();

        let guard = texture.buffer.read();
        let image: ImageBuffer<Rgba<u8>, &[u8]> =
            ImageBuffer::from_raw(size.width, size.height, &*guard).unwrap();
        image.save(path)
    }
}
