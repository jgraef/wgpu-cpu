use crate::{
    buffer::{
        Buffer,
        BufferReadGuard,
        BufferWriteGuard,
    },
    sync::wait,
};

#[derive(Clone, Debug)]
pub struct Texture {
    pub buffer: Buffer,

    // This is super janky and easy to use wrong. it basically notifies anyone having the Wait part
    // when a TextureWriteGuard for this texture is dropped. This is used by the surface
    // implementation. It attaches a Notify when handing out the texture and when present is
    // called, it waits until the texture actually has been modified.
    notify: Option<wait::Notify>,
}

impl Texture {
    pub fn new(size: wgpu::Extent3d, format: wgpu::TextureFormat) -> Self {
        Self {
            buffer: Buffer::new(calculate_texture_buffer_size(size, format)),
            notify: None,
        }
    }

    pub fn wait(&mut self) -> wait::Wait {
        let (notify, wait) = wait::channel();
        self.notify = Some(notify);
        wait
    }
}

impl wgpu::custom::TextureInterface for Texture {
    fn create_view(
        &self,
        desc: &wgpu::TextureViewDescriptor<'_>,
    ) -> wgpu::custom::DispatchTextureView {
        wgpu::custom::DispatchTextureView::custom(TextureView {
            descriptor: wgpu::wgt::TextureViewDescriptor {
                label: desc.label.map(ToOwned::to_owned),
                format: desc.format,
                dimension: desc.dimension,
                usage: desc.usage,
                aspect: desc.aspect,
                base_mip_level: desc.base_mip_level,
                mip_level_count: desc.mip_level_count,
                base_array_layer: desc.base_array_layer,
                array_layer_count: desc.array_layer_count,
            },
        })
    }

    fn destroy(&self) {
        // nop
    }
}

#[derive(Debug)]
pub struct TextureViewDescriptor {
    pub label: Option<String>,
    pub format: wgpu::TextureFormat,
    pub dimension: wgpu::TextureViewDimension,
    pub usage: wgpu::TextureUsages,
    pub aspect: wgpu::TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
}

#[derive(Clone, Debug)]
pub struct TextureView {
    descriptor: wgpu::wgt::TextureViewDescriptor<Option<String>>,
}

impl wgpu::custom::TextureViewInterface for TextureView {}

#[derive(Clone, Copy, Debug)]
pub struct TextureInfo {
    pub format: wgpu::TextureFormat,
    pub dimension: wgpu::TextureViewDimension,
    pub usage: wgpu::TextureUsages,
    pub aspect: wgpu::TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
}

#[derive(Clone, Debug)]
pub struct TextureViewAttachment {
    pub buffer: Buffer,
    pub info: TextureInfo,
    pub notify: Option<wait::Notify>,
}

impl TextureViewAttachment {
    pub fn from_wgpu(texture_view: &wgpu::TextureView) -> Option<Self> {
        let texture_view_descriptor = &texture_view.as_custom::<TextureView>().as_ref()?.descriptor;
        let wgpu_texture = texture_view.texture();
        let texture = wgpu_texture.as_custom::<Texture>()?;

        Some(Self {
            buffer: texture.buffer.clone(),
            info: TextureInfo {
                format: texture_view_descriptor
                    .format
                    .unwrap_or(wgpu_texture.format()),
                dimension: texture_view_descriptor.dimension.unwrap_or_else(|| {
                    match wgpu_texture.dimension() {
                        wgpu::TextureDimension::D1 => wgpu::TextureViewDimension::D1,
                        wgpu::TextureDimension::D2 => wgpu::TextureViewDimension::D2,
                        wgpu::TextureDimension::D3 => wgpu::TextureViewDimension::D3,
                    }
                }),
                usage: texture_view_descriptor
                    .usage
                    .unwrap_or(wgpu_texture.usage()),
                aspect: texture_view_descriptor.aspect,
                base_mip_level: texture_view_descriptor.base_mip_level,
                mip_level_count: texture_view_descriptor.mip_level_count.unwrap_or_else(|| {
                    wgpu_texture.mip_level_count() - texture_view_descriptor.base_mip_level
                }),
                base_array_layer: texture_view_descriptor.base_array_layer,
                array_layer_count: texture_view_descriptor
                    .array_layer_count
                    .unwrap_or_else(|| {
                        wgpu_texture.depth_or_array_layers()
                            - texture_view_descriptor.base_array_layer
                    }),
            },
            notify: texture.notify.clone(),
        })
    }

    pub fn read(&self) -> TextureReadGuard<'_> {
        TextureReadGuard {
            guard: self.buffer.read(),
            info: &self.info,
        }
    }

    pub fn write(&self) -> TextureWriteGuard<'_> {
        TextureWriteGuard {
            guard: self.buffer.write(),
            info: &self.info,
            notify: self.notify.as_ref(),
        }
    }
}

#[derive(Debug)]
pub struct TextureReadGuard<'a> {
    guard: BufferReadGuard<'a>,
    info: &'a TextureInfo,
}

#[derive(Debug)]
pub struct TextureWriteGuard<'a> {
    guard: BufferWriteGuard<'a>,
    info: &'a TextureInfo,
    notify: Option<&'a wait::Notify>,
}

impl<'a> TextureWriteGuard<'a> {
    pub fn clear(&mut self, color: wgpu::Color) {
        let mut texel = [0; 4];

        fn f64_to_u8(value: f64) -> u8 {
            (value * 255.0).clamp(0.0, 255.0) as u8
        }

        match self.info.format {
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Snorm => {
                texel[0] = f64_to_u8(color.r);
                texel[1] = f64_to_u8(color.g);
                texel[2] = f64_to_u8(color.b);
                texel[3] = f64_to_u8(color.a);
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                texel[0] = f64_to_u8(color.b);
                texel[1] = f64_to_u8(color.g);
                texel[2] = f64_to_u8(color.r);
                texel[3] = f64_to_u8(color.a);
            }
            _ => todo!(),
        }

        let target: &mut [[u8; 4]] = bytemuck::cast_slice_mut(&mut *self.guard);

        target.fill(texel);
    }
}

impl<'a> Drop for TextureWriteGuard<'a> {
    fn drop(&mut self) {
        if let Some(notify) = self.notify.take() {
            notify.notify();
        }
    }
}

fn calculate_texture_buffer_size(size: wgpu::Extent3d, format: wgpu::TextureFormat) -> usize {
    size.width as usize
        * size.height as usize
        * size.depth_or_array_layers as usize
        * bytes_per_texel(format)
}

pub type MaxTexel = [u8; wgpu::TextureFormat::MAX_TARGET_PIXEL_BYTE_COST as usize];

fn bytes_per_texel(format: wgpu::TextureFormat) -> usize {
    match format {
        wgpu::TextureFormat::R8Unorm
        | wgpu::TextureFormat::R8Snorm
        | wgpu::TextureFormat::R8Uint
        | wgpu::TextureFormat::R8Sint
        | wgpu::TextureFormat::Stencil8 => 1,
        wgpu::TextureFormat::R16Uint
        | wgpu::TextureFormat::R16Sint
        | wgpu::TextureFormat::R16Unorm
        | wgpu::TextureFormat::R16Snorm
        | wgpu::TextureFormat::R16Float
        | wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Rg8Snorm
        | wgpu::TextureFormat::Rg8Uint
        | wgpu::TextureFormat::Rg8Sint
        | wgpu::TextureFormat::Depth16Unorm => 2,
        wgpu::TextureFormat::R32Uint
        | wgpu::TextureFormat::R32Sint
        | wgpu::TextureFormat::R32Float
        | wgpu::TextureFormat::Rg16Uint
        | wgpu::TextureFormat::Rg16Sint
        | wgpu::TextureFormat::Rg16Unorm
        | wgpu::TextureFormat::Rg16Snorm
        | wgpu::TextureFormat::Rg16Float
        | wgpu::TextureFormat::Rgba8Unorm
        | wgpu::TextureFormat::Rgba8UnormSrgb
        | wgpu::TextureFormat::Rgba8Snorm
        | wgpu::TextureFormat::Rgba8Uint
        | wgpu::TextureFormat::Rgba8Sint
        | wgpu::TextureFormat::Bgra8Unorm
        | wgpu::TextureFormat::Bgra8UnormSrgb => 4,
        wgpu::TextureFormat::R64Uint
        | wgpu::TextureFormat::Rg32Uint
        | wgpu::TextureFormat::Rg32Sint
        | wgpu::TextureFormat::Rg32Float
        | wgpu::TextureFormat::Rgba16Uint
        | wgpu::TextureFormat::Rgba16Sint
        | wgpu::TextureFormat::Rgba16Unorm
        | wgpu::TextureFormat::Rgba16Snorm
        | wgpu::TextureFormat::Rgba16Float => 8,
        wgpu::TextureFormat::Rgba32Uint
        | wgpu::TextureFormat::Rgba32Sint
        | wgpu::TextureFormat::Rgba32Float
        | wgpu::TextureFormat::Depth24Plus
        | wgpu::TextureFormat::Depth24PlusStencil8
        | wgpu::TextureFormat::Depth32Float
        | wgpu::TextureFormat::Depth32FloatStencil8 => 16,
        _ => todo!(),
    }
}
