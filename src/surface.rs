use std::{
    fmt::Debug,
    num::NonZero,
    sync::Arc,
};

use parking_lot::Mutex;
use softbuffer::SoftBufferError;

use crate::{
    Device,
    sync::wait,
    texture::Texture,
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
                    inner: Arc::new(Mutex::new(Inner {
                        context,
                        surface,
                        configured: None,
                    })),
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
            formats: vec![TEXTURE_FORMAT],
            // Don't know. wgpu doc says wayland doesn't support immediate, so why should softbuffer
            // do this on wayland? but I don't know what softbuffer exactly does.
            present_modes: vec![wgpu::PresentMode::Immediate],
            // No alpha
            alpha_modes: vec![wgpu::CompositeAlphaMode::Opaque],
            //usages: TEXTURE_USAGES,
            usages: wgpu::TextureUsages::RENDER_ATTACHMENT,
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

        inner.configured = Some(Configured {
            width: config.width,
            height: config.height,
            format: config.format,
            usage: config.usage,
            buffer: Texture::new(
                wgpu::Extent3d {
                    width: config.width,
                    height: config.height,
                    depth_or_array_layers: 1,
                },
                config.format,
            ),
            wait: None,
        });

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
        let buffer = {
            let mut inner = self.inner.lock();

            let configured = inner
                .configured
                .as_mut()
                .expect("Surface not configured yet");

            let mut buffer = configured.buffer.clone();

            configured.wait = Some(buffer.wait());

            buffer
        };

        (
            Some(wgpu::custom::DispatchTexture::custom(buffer)),
            wgpu::SurfaceStatus::Good,
            wgpu::custom::DispatchSurfaceOutputDetail::custom(SurfaceOutputDetail {
                inner: self.inner.clone(),
            }),
        )
    }
}

#[derive(derive_more::Debug)]
struct Inner {
    #[debug(skip)]
    context: softbuffer::Context<Display>,
    #[debug(skip)]
    surface: softbuffer::Surface<Display, Window>,

    configured: Option<Configured>,
}

#[derive(Debug)]
struct Configured {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    buffer: Texture,
    wait: Option<wait::Wait>,
}

#[derive(Debug)]
pub struct SurfaceOutputDetail {
    inner: Arc<Mutex<Inner>>,
}

impl wgpu::custom::SurfaceOutputDetailInterface for SurfaceOutputDetail {
    fn present(&self) {
        let mut inner = self.inner.lock();
        let inner = &mut *inner;

        {
            let configured = inner
                .configured
                .as_mut()
                .expect("Surface not configured yet");

            if let Some(wait) = configured.wait.take() {
                let _ = wait.wait();
            }

            let source = configured.buffer.buffer.read();

            let mut target = inner.surface.buffer_mut().unwrap();
            let target: &mut [u8] = bytemuck::cast_slice_mut(&mut *target);

            target.copy_from_slice(&*source);
        }

        inner.surface.buffer_mut().unwrap().present().unwrap();
    }

    fn texture_discard(&self) {
        // nop
    }
}

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

const TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

fn check_surface_format(format: wgpu::TextureFormat) -> Result<(), SurfaceConfigError> {
    if format == TEXTURE_FORMAT {
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
