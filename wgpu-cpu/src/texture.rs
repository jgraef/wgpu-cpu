use std::ops::Range;

use arrayvec::ArrayVec;
use nalgebra::{
    Point2,
    Point3,
    Vector3,
    Vector4,
};

use crate::{
    buffer::{
        Buffer,
        BufferReadGuard,
        BufferWriteGuard,
    },
    util::sync::wait,
};

#[derive(Clone, Debug)]
pub struct Texture {
    pub buffer: Buffer,

    pub data_layout: TextureDataLayout,

    // This is super janky and easy to use wrong. it basically notifies anyone having the Wait part
    // when a TextureWriteGuard for this texture is dropped. This is used by the surface
    // implementation. It attaches a Notify when handing out the texture and when present is
    // called, it waits until the texture actually has been modified.
    notify: Option<wait::Notify>,
}

impl Texture {
    pub fn new(size: wgpu::Extent3d, format: wgpu::TextureFormat) -> Self {
        let data_layout = TextureDataLayout::from_size(format, size);
        let buffer = Buffer::new_unmapped(data_layout.byte_size());

        Self {
            buffer,
            data_layout,
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

#[derive(Clone, Debug)]
pub struct TextureView {
    descriptor: wgpu::wgt::TextureViewDescriptor<Option<String>>,
}

impl wgpu::custom::TextureViewInterface for TextureView {}

#[derive(Clone, Copy, Debug)]
pub struct TextureInfo {
    pub format: wgpu::TextureFormat,
    pub size: wgpu::Extent3d,
    pub dimension: wgpu::TextureViewDimension,
    pub usage: wgpu::TextureUsages,
    pub aspect: wgpu::TextureAspect,
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
    pub data_layout: TextureDataLayout,
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
                size: wgpu_texture.size(),
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
                // todo: apply depth etc. to data layout
                data_layout: texture.data_layout,
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
    pub info: &'a TextureInfo,
}

impl<'a> TextureReadGuard<'a> {
    pub fn get_pixel(&self, position: Point3<u32>) -> Vector4<f32> {
        let texel_data = &self.guard[self.info.data_layout.texel_byte_range(position)];

        fn u8_to_f32(x: u8) -> f32 {
            x as f32 / 255.0
        }

        match self.info.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                Vector4::new(
                    u8_to_f32(texel_data[0]),
                    u8_to_f32(texel_data[1]),
                    u8_to_f32(texel_data[2]),
                    u8_to_f32(texel_data[3]),
                )
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct TextureWriteGuard<'a> {
    guard: BufferWriteGuard<'a>,
    pub info: &'a TextureInfo,
    notify: Option<&'a wait::Notify>,
}

impl<'a> TextureWriteGuard<'a> {
    pub fn clear_color(&mut self, color: wgpu::Color) {
        let writer = TexelWriter::from_color(wgpu_color_to_vec4(color), self.info.format)
            .truncated(self.info.data_layout.texel_byte_size());

        for range in self.info.data_layout.texel_offset_iter() {
            writer.write(&mut self.guard[range]);
        }
    }

    pub fn clear_depth(&mut self, depth: f32) {
        let target: &mut [f32] = bytemuck::cast_slice_mut(&mut *self.guard);
        target.fill(depth);
    }

    pub fn put_pixel(&mut self, position: Point3<u32>, color: Vector4<f32>) {
        let writer = TexelWriter::from_color(color, self.info.format)
            .truncated(self.info.data_layout.texel_byte_size());

        writer.write(&mut self.guard[self.info.data_layout.texel_byte_range(position)]);
    }

    pub fn get_depth(&self, position: Point2<u32>) -> f32 {
        // todo: read method should also be on TextureReadGuard
        let range = self
            .info
            .data_layout
            .texel_byte_range(Point3::new(position.x, position.y, 0));
        *bytemuck::from_bytes(&self.guard[range])
    }

    pub fn put_depth(&mut self, position: Point2<u32>, depth: f32) {
        let writer = TexelWriter::from_depth(depth, self.info.format)
            .truncated(self.info.data_layout.texel_byte_size());

        writer.write(
            &mut self.guard[self
                .info
                .data_layout
                .texel_byte_range(Point3::new(position.x, position.y, 0))],
        )
    }
}

impl<'a> Drop for TextureWriteGuard<'a> {
    fn drop(&mut self) {
        if let Some(notify) = self.notify.take() {
            notify.notify();
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TextureDataLayout {
    pub offset: usize,
    pub stride: Vector4<usize>,
    pub size: Vector3<usize>,
}

impl TextureDataLayout {
    pub fn from_size(format: wgpu::TextureFormat, size: wgpu::Extent3d) -> Self {
        let size: Vector3<usize> =
            Vector3::new(size.width, size.height, size.depth_or_array_layers).cast();
        let mut stride = Vector4::zeros();

        stride.x = bytes_per_texel(format)
            .unwrap_or_else(|| panic!("Unsupported texture format: {format:?}"));
        stride.y = size.x * stride.x;
        stride.z = size.y * stride.y;
        stride.w = size.z * stride.z;

        Self {
            offset: 0,
            stride,
            size,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.stride.w
    }

    pub fn texel_byte_size(&self) -> usize {
        self.stride.x
    }

    pub fn texel_byte_range(&self, point: Point3<u32>) -> Range<usize> {
        let start = self.texel_byte_offset(point);
        let end = start + self.texel_byte_size();
        start..end
    }

    pub fn texel_byte_offset(&self, point: Point3<u32>) -> usize {
        self.offset + point.coords.cast::<usize>().dot(&self.stride.xyz())
    }

    pub fn with_origin(mut self, origin: wgpu::Origin3d) -> Self {
        self.offset = self.texel_byte_offset(Point3::new(origin.x, origin.y, origin.z));
        self
    }

    pub fn texel_offset_iter(&self) -> TexelIter {
        TexelIter::new(self.stride, self.size, self.offset)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TexelIter {
    exhausted: bool,
    stride: Vector4<usize>,
    size: Vector3<usize>,
    position: Vector3<usize>,
    offset: Vector3<usize>,
}

impl TexelIter {
    fn new(stride: Vector4<usize>, size: Vector3<usize>, offset: usize) -> Self {
        Self {
            exhausted: stride.w == 0,
            stride,
            size,
            position: Vector3::zeros(),
            offset: Vector3::repeat(offset),
        }
    }
}

impl Iterator for TexelIter {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }

        let start = self.offset.x;

        self.position.x += 1;
        if self.position.x < self.size.x {
            self.offset.x += self.stride.x;
        }
        else {
            self.position.y += 1;
            if self.position.y < self.size.y {
                self.offset.y += self.stride.y;
            }
            else {
                self.position.z += 1;

                if self.position.z < self.size.z {
                    self.offset.z += self.stride.z;
                }
                else {
                    self.exhausted = true;
                }

                self.position.y = 0;
                self.offset.y = self.offset.z;
            }

            self.position.x = 0;
            self.offset.x = self.offset.y;
        }

        let end = start + self.stride.x;
        Some(start..end)
    }
}

#[derive(Clone, Debug)]
pub struct TexelWriter {
    texel: ArrayVec<u8, 16>,
}

impl TexelWriter {
    pub fn from_color(color: Vector4<f32>, format: wgpu::TextureFormat) -> Self {
        let mut texel = ArrayVec::new();

        fn f32_to_u8(value: f32) -> u8 {
            const MAX: f32 = u8::MAX as f32;
            (value * MAX).clamp(0.0, MAX) as u8
        }

        match format {
            wgpu::TextureFormat::R8Unorm
            | wgpu::TextureFormat::R8Uint
            | wgpu::TextureFormat::R8Sint => {
                texel.push(f32_to_u8(color.x));
            }
            wgpu::TextureFormat::Rg8Unorm
            | wgpu::TextureFormat::Rg8Uint
            | wgpu::TextureFormat::Rg8Sint => {
                texel.push(f32_to_u8(color.x));
                texel.push(f32_to_u8(color.y));
            }
            wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Rgba8Snorm => {
                texel.push(f32_to_u8(color.x));
                texel.push(f32_to_u8(color.y));
                texel.push(f32_to_u8(color.z));
                texel.push(f32_to_u8(color.w));
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                texel.push(f32_to_u8(color.z));
                texel.push(f32_to_u8(color.y));
                texel.push(f32_to_u8(color.x));
                texel.push(f32_to_u8(color.w));
            }
            _ => panic!("Unsupported texture format: {format:?}"),
        }

        Self { texel }
    }

    pub fn from_depth(depth: f32, format: wgpu::TextureFormat) -> Self {
        let mut texel = ArrayVec::new();

        fn f32_to_u16(value: f32) -> u16 {
            const MAX: f32 = u16::MAX as f32;
            (value * MAX).clamp(0.0, MAX) as u16
        }

        fn f32_to_u24(value: f32) -> u16 {
            const MAX: f32 = ((1u32 << 24) - 1) as f32;
            (value * MAX).clamp(0.0, MAX) as u16
        }

        match format {
            wgpu::TextureFormat::Depth16Unorm => {
                texel.extend(bytemuck::bytes_of(&f32_to_u16(depth)).iter().copied());
            }
            wgpu::TextureFormat::Depth24Plus | wgpu::TextureFormat::Depth24PlusStencil8 => {
                texel.extend(bytemuck::bytes_of(&f32_to_u24(depth)).iter().copied());
            }
            wgpu::TextureFormat::Depth32Float => {
                texel.extend(bytemuck::bytes_of(&depth).iter().copied());
            }
            _ => panic!("Unsupported texture format: {format:?}"),
        }

        Self { texel }
    }

    pub fn truncate(&mut self, texel_size: usize) {
        self.texel.truncate(texel_size);
    }

    pub fn truncated(mut self, texel_size: usize) -> Self {
        self.truncate(texel_size);
        self
    }

    pub fn write(&self, destination: &mut [u8]) {
        destination.copy_from_slice(&self.texel);
    }
}

fn bytes_per_texel(format: wgpu::TextureFormat) -> Option<usize> {
    match format {
        wgpu::TextureFormat::R8Unorm
        | wgpu::TextureFormat::R8Snorm
        | wgpu::TextureFormat::R8Uint
        | wgpu::TextureFormat::R8Sint
        | wgpu::TextureFormat::Stencil8 => Some(1),
        wgpu::TextureFormat::R16Uint
        | wgpu::TextureFormat::R16Sint
        | wgpu::TextureFormat::R16Unorm
        | wgpu::TextureFormat::R16Snorm
        | wgpu::TextureFormat::R16Float
        | wgpu::TextureFormat::Rg8Unorm
        | wgpu::TextureFormat::Rg8Snorm
        | wgpu::TextureFormat::Rg8Uint
        | wgpu::TextureFormat::Rg8Sint
        | wgpu::TextureFormat::Depth16Unorm => Some(2),
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
        | wgpu::TextureFormat::Bgra8UnormSrgb => Some(4),
        wgpu::TextureFormat::R64Uint
        | wgpu::TextureFormat::Rg32Uint
        | wgpu::TextureFormat::Rg32Sint
        | wgpu::TextureFormat::Rg32Float
        | wgpu::TextureFormat::Rgba16Uint
        | wgpu::TextureFormat::Rgba16Sint
        | wgpu::TextureFormat::Rgba16Unorm
        | wgpu::TextureFormat::Rgba16Snorm
        | wgpu::TextureFormat::Rgba16Float => Some(8),
        wgpu::TextureFormat::Rgba32Uint
        | wgpu::TextureFormat::Rgba32Sint
        | wgpu::TextureFormat::Rgba32Float => Some(16),
        wgpu::TextureFormat::Depth24Plus
        | wgpu::TextureFormat::Depth24PlusStencil8
        | wgpu::TextureFormat::Depth32Float => Some(4),
        _ => None,
    }
}

fn wgpu_color_to_vec4(color: wgpu::Color) -> Vector4<f32> {
    Vector4::new(color.r, color.g, color.b, color.a).cast()
}

#[cfg(test)]
mod tests {
    use crate::texture::TextureDataLayout;

    #[test]
    fn data_layout_size() {
        let data_layout = TextureDataLayout::from_size(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::Extent3d {
                width: 3,
                height: 5,
                depth_or_array_layers: 1,
            },
        );

        assert_eq!(data_layout.byte_size(), 60);
    }

    #[test]
    fn data_layout_iter() {
        let data_layout = TextureDataLayout::from_size(
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::Extent3d {
                width: 3,
                height: 5,
                depth_or_array_layers: 1,
            },
        );

        let offsets = data_layout.texel_offset_iter().collect::<Vec<_>>();
        let expected = (0usize..15)
            .map(|i| {
                let start = i * 4;
                let end = start + 4;
                start..end
            })
            .collect::<Vec<_>>();
        assert_eq!(offsets, expected);
    }
}
