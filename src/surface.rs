use std::{
    fmt::Debug,
    num::NonZero,
    sync::Arc,
};

use parking_lot::Mutex;
use softbuffer::SoftBufferError;

use crate::{
    Device,
    TEXTURE_USAGES,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    SoftBuffer(#[from] SoftBufferError),
    #[error("Surface not supported")]
    Unsupported,
}

#[derive(Debug)]
pub struct Surface {
    inner: Arc<Mutex<Inner>>,
}

#[derive(derive_more::Debug)]
struct Inner {
    #[debug(skip)]
    context: softbuffer::Context<Display>,
    #[debug(skip)]
    surface: softbuffer::Surface<Display, Window>,
}

impl Surface {
    pub fn new(target: wgpu::SurfaceTargetUnsafe) -> Result<Self, Error> {
        match target {
            wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle,
                raw_window_handle,
            } => {
                let context = softbuffer::Context::new(Display::from(raw_display_handle))?;
                let surface = softbuffer::Surface::new(&context, Window::from(raw_window_handle))?;
                Ok(Self {
                    inner: Arc::new(Mutex::new(Inner { context, surface })),
                })
            }
            _ => Err(Error::Unsupported),
        }
    }
}

impl wgpu::custom::SurfaceInterface for Surface {
    fn get_capabilities(
        &self,
        adapter: &wgpu::custom::DispatchAdapter,
    ) -> wgpu::SurfaceCapabilities {
        wgpu::SurfaceCapabilities {
            // According to [`Buffer`][1] the buffer we get is just 32bit RGB, but it's layout such
            // that it's actually BGR and we don't get a alpha
            //
            // [1]: https://docs.rs/softbuffer/latest/softbuffer/struct.Buffer.html#data-representation
            formats: vec![wgpu::TextureFormat::Bgra8Unorm],
            // Don't know. wgpu doc says wayland doesn't support immediate, so why should softbuffer
            // do this on wayland? but I don't know what softbuffer exactly does.
            present_modes: vec![wgpu::PresentMode::Immediate],
            // No alpha
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            usages: TEXTURE_USAGES,
        }
    }

    fn configure(
        &self,
        device: &wgpu::custom::DispatchDevice,
        config: &wgpu::SurfaceConfiguration,
    ) {
        check_surface_config(config).unwrap();

        let device = device.as_custom::<Device>().unwrap();

        tracing::debug!(?config, "configure surface");

        let mut inner = self.inner.lock();
        inner
            .surface
            .resize(
                NonZero::new(config.width).expect("Surface width must not be zero"),
                NonZero::new(config.height).expect("Surface height must not be zero"),
            )
            .unwrap();

        // todo: pass error to device's error handler
        // or better yet. grab the devices error handler and store it in the
        // surface so we have an error sink for later
    }

    fn get_current_texture(
        &self,
    ) -> (
        Option<wgpu::custom::DispatchTexture>,
        wgpu::SurfaceStatus,
        wgpu::custom::DispatchSurfaceOutputDetail,
    ) {
        (
            Some(wgpu::custom::DispatchTexture::custom(SurfaceTexture {
                inner: self.inner.clone(),
            })),
            wgpu::SurfaceStatus::Good,
            wgpu::custom::DispatchSurfaceOutputDetail::custom(SurfaceOutputDetail {
                inner: self.inner.clone(),
            }),
        )
    }
}

#[derive(Debug)]
pub struct SurfaceOutputDetail {
    inner: Arc<Mutex<Inner>>,
}

impl wgpu::custom::SurfaceOutputDetailInterface for SurfaceOutputDetail {
    fn present(&self) {
        let mut inner = self.inner.lock();
        inner.surface.buffer_mut().unwrap().present().unwrap();
    }

    fn texture_discard(&self) {
        // todo
    }
}

#[derive(Debug)]
pub struct SurfaceTexture {
    inner: Arc<Mutex<Inner>>,
}

impl wgpu::custom::TextureInterface for SurfaceTexture {
    fn create_view(
        &self,
        desc: &wgpu::TextureViewDescriptor<'_>,
    ) -> wgpu::custom::DispatchTextureView {
        wgpu::custom::DispatchTextureView::custom(SurfaceTextureView {
            inner: self.inner.clone(),
        })
    }

    fn destroy(&self) {
        // nop
    }
}

#[derive(Debug)]
pub struct SurfaceTextureView {
    inner: Arc<Mutex<Inner>>,
}

impl wgpu::custom::TextureViewInterface for SurfaceTextureView {}

struct Display(wgpu::rwh::DisplayHandle<'static>);

impl From<wgpu::rwh::RawDisplayHandle> for Display {
    fn from(value: wgpu::rwh::RawDisplayHandle) -> Self {
        Self(unsafe { wgpu::rwh::DisplayHandle::borrow_raw(value) })
    }
}

impl wgpu::rwh::HasDisplayHandle for Display {
    fn display_handle(&self) -> Result<wgpu::rwh::DisplayHandle<'_>, wgpu::rwh::HandleError> {
        Ok(self.0)
    }
}

unsafe impl Send for Display {}
unsafe impl Sync for Display {}

struct Window(wgpu::rwh::WindowHandle<'static>);

impl From<wgpu::rwh::RawWindowHandle> for Window {
    fn from(value: wgpu::rwh::RawWindowHandle) -> Self {
        Self(unsafe { wgpu::rwh::WindowHandle::borrow_raw(value) })
    }
}

impl wgpu::rwh::HasWindowHandle for Window {
    fn window_handle(&self) -> Result<wgpu::rwh::WindowHandle<'_>, wgpu::rwh::HandleError> {
        Ok(self.0)
    }
}

unsafe impl Send for Window {}
//unsafe impl Sync for Window {}

fn check_surface_config(config: &wgpu::SurfaceConfiguration) -> Result<(), SurfaceConfigError> {
    check_surface_format(config.format)?;
    for view_format in &config.view_formats {
        check_surface_format(*view_format)?;
    }
    Ok(())
}

fn check_surface_format(format: wgpu::TextureFormat) -> Result<(), SurfaceConfigError> {
    if format == wgpu::TextureFormat::Bgra8Unorm {
        Ok(())
    }
    else {
        Err(SurfaceConfigError::UnsupportedTextureFormat(format))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SurfaceConfigError {
    #[error("Unsupported surface texture format: {0:?}")]
    UnsupportedTextureFormat(wgpu::TextureFormat),
}
