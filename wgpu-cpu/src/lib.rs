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
mod texture;
mod util;

#[cfg(feature = "image")]
pub use image::dump_texture;

use crate::instance::Instance;

pub fn instance() -> wgpu::Instance {
    wgpu::Instance::from_custom(Instance::default())
}

// todo: this is for testing. a better interface would be nice
#[cfg(feature = "image")]
pub mod image {
    use std::path::Path;

    use bytemuck::Pod;
    use image::{
        ImageBuffer,
        ImageError,
        ImageFormat,
        Luma,
        Pixel,
        PixelWithColorType,
        Rgba,
    };
    use wgpu::Extent3d;

    use crate::{
        buffer::{
            Buffer,
            BufferReadGuard,
        },
        texture::Texture,
    };

    /// Write texture buffer with pixel type `P` directly to file
    fn write_buffer_to_image_directly<P>(
        path: impl AsRef<Path>,
        buffer: &Buffer,
        size: Extent3d,
        format: Option<ImageFormat>,
    ) -> Result<(), ImageError>
    where
        P: Pixel<Subpixel = u8> + PixelWithColorType,
    {
        let buffer = buffer.read();
        let image: ImageBuffer<P, &[u8]> =
            ImageBuffer::from_raw(size.width, size.height, &*buffer).unwrap();

        if let Some(format) = format {
            image.save_with_format(path, format)
        }
        else {
            image.save(path)
        }
    }

    /// Convert texture buffer with pixels `T` to image of pixels `P` and write
    /// to file
    fn write_buffer_to_image_converted<T, P>(
        path: impl AsRef<Path>,
        buffer: &Buffer,
        size: Extent3d,
        mut convert: impl FnMut(T) -> P,
        format: Option<ImageFormat>,
    ) -> Result<(), ImageError>
    where
        T: Pod,
        P: Pixel<Subpixel = u8> + PixelWithColorType,
    {
        let converted = {
            // todo: layout
            let buffer = buffer.read();
            let source: &[T] = bytemuck::cast_slice(&*buffer);

            let mut converted: ImageBuffer<P, Vec<u8>> = ImageBuffer::new(size.width, size.height);

            for (i, (x, y, pixel)) in converted.enumerate_pixels_mut().enumerate() {
                *pixel = convert(source[i]);
            }

            converted
        };

        if let Some(format) = format {
            converted.save_with_format(path, format)
        }
        else {
            converted.save(path)
        }
    }

    pub fn dump_texture(
        texture: &wgpu::Texture,
        path: impl AsRef<Path>,
        file_format: impl Into<Option<ImageFormat>>,
    ) -> Result<(), ImageError> {
        let path = path.as_ref();
        let file_format = file_format.into();

        let size = texture.size();
        let texture_format = texture.format();
        let texture = texture.as_custom::<Texture>().unwrap();

        match texture_format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Uint => {
                write_buffer_to_image_directly::<Rgba<u8>>(
                    path,
                    &texture.buffer,
                    size,
                    file_format,
                )?
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                write_buffer_to_image_converted::<[u8; 4], Rgba<u8>>(
                    path,
                    &texture.buffer,
                    size,
                    |[b, g, r, a]| Rgba([r, g, b, a]),
                    file_format,
                )?
            }
            wgpu::TextureFormat::Depth32Float => {
                write_buffer_to_image_converted::<f32, Luma<u8>>(
                    path,
                    &texture.buffer,
                    size,
                    |depth| Luma([(depth * 255.0) as u8]),
                    file_format,
                )?
            }
            _ => todo!("output texture format not implemented: {texture_format:?}"),
        }

        Ok(())
    }

    pub fn rgba_texture_image(
        texture: &wgpu::Texture,
    ) -> ImageBuffer<Rgba<u8>, BufferReadGuard<'_>> {
        let size = texture.size();
        let texture_format = texture.format();
        let texture = texture.as_custom::<Texture>().unwrap();

        match texture_format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {}
            _ => panic!("Only RGBA supported"),
        }

        let buffer = texture.buffer.read();

        ImageBuffer::from_raw(size.width, size.height, buffer).unwrap()
    }

    #[derive(Clone, Copy, Debug)]
    struct Bgra<T>(pub [T; 4]);
}
