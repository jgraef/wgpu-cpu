use wgpu::custom::TextureInterface;

#[derive(Debug)]
pub enum Texture {
    // todo
    Buffer,

    #[cfg(feature = "softbuffer")]
    Surface(crate::surface::SurfaceTexture),
}

impl TextureInterface for Texture {
    fn create_view(
        &self,
        desc: &wgpu::TextureViewDescriptor<'_>,
    ) -> wgpu::custom::DispatchTextureView {
        match self {
            Texture::Buffer => todo!(),
            #[cfg(feature = "softbuffer")]
            Texture::Surface(surface_texture) => surface_texture.create_view(desc),
        }
    }

    fn destroy(&self) {
        match self {
            Texture::Buffer => todo!(),
            #[cfg(feature = "softbuffer")]
            Texture::Surface(surface_texture) => surface_texture.destroy(),
        }
    }
}

#[derive(Debug)]
pub enum TextureView {
    // todo
    Buffer,

    #[cfg(feature = "softbuffer")]
    Surface(crate::surface::SurfaceTextureView),
}
