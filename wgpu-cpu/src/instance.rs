use std::pin::Pin;

use wgpu::{
    CreateSurfaceError,
    InstanceDescriptor,
    RequestAdapterError,
    RequestAdapterOptions,
    SurfaceTargetUnsafe,
    custom::{
        DispatchAdapter,
        DispatchSurface,
        EnumerateAdapterFuture,
        InstanceInterface,
        RequestAdapterFuture,
    },
};

use crate::adapter::Adapter;

#[derive(Debug, Default)]
pub struct Instance {
    _placeholder: (),
}

impl InstanceInterface for Instance {
    fn new(instance_descriptor: &InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        let _ = instance_descriptor;
        Self { _placeholder: () }
    }

    unsafe fn create_surface(
        &self,
        target: SurfaceTargetUnsafe,
    ) -> Result<DispatchSurface, CreateSurfaceError> {
        #[cfg(feature = "softbuffer")]
        {
            Ok(DispatchSurface::custom(
                crate::surface::Surface::new(target)
                    .map_err(|error| CreateSurfaceError::custom(error.to_string()))?,
            ))
        }

        #[cfg(not(feature = "softbuffer"))]
        {
            Err(CreateSurfaceError::custom(
                "wgpu-cpu compiled without softbuffer feature, so no surfaces are supported"
                    .to_owned(),
            ))
        }
    }

    fn request_adapter(
        &self,
        options: &RequestAdapterOptions,
    ) -> Pin<Box<dyn RequestAdapterFuture>> {
        let mut compatible = true;

        if let Some(compatible_surface) = &options.compatible_surface {
            compatible = false;

            #[cfg(feature = "softbuffer")]
            if compatible_surface
                .as_custom::<crate::surface::Surface>()
                .is_some()
            {
                compatible = true;
            }
        }

        Box::pin(async move {
            if compatible {
                Ok(DispatchAdapter::custom(Adapter::new()))
            }
            else {
                Err(RequestAdapterError::Custom(
                    "Surface not compatible".to_owned(),
                ))
            }
        })
    }

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        // nop for now. we'll see how we manage queues
        let _ = force_wait;
        true
    }

    fn enumerate_adapters(&self, backends: wgpu::Backends) -> Pin<Box<dyn EnumerateAdapterFuture>> {
        let _ = backends;
        Box::pin(async { vec![DispatchAdapter::custom(Adapter::new())] })
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        wgpu::WgslLanguageFeatures::empty()
    }
}
