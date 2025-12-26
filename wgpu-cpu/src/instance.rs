use std::{
    pin::Pin,
    sync::Arc,
};

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

#[derive(Clone, Debug, Default)]
pub struct InstanceConfig {
    // todo
}

#[derive(Debug)]
pub struct Instance {
    instance_config: Arc<InstanceConfig>,
}

impl Instance {
    pub fn new(config: InstanceConfig) -> Self {
        tracing::debug!(?config, "creating instance");

        Self {
            instance_config: Arc::new(config),
        }
    }
}

impl InstanceInterface for Instance {
    fn new(instance_descriptor: InstanceDescriptor) -> Self
    where
        Self: Sized,
    {
        unreachable!("InstanceInterface::new should not be called for wgpu_cpu");
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

        let result = if compatible {
            Ok(DispatchAdapter::custom(Adapter::new(
                self.instance_config.clone(),
            )))
        }
        else {
            Err(RequestAdapterError::Custom(
                "Surface not compatible".to_owned(),
            ))
        };

        Box::pin(async move { result })
    }

    fn poll_all_devices(&self, force_wait: bool) -> bool {
        // nop for now. we'll see how we manage queues
        let _ = force_wait;
        true
    }

    fn enumerate_adapters(&self, backends: wgpu::Backends) -> Pin<Box<dyn EnumerateAdapterFuture>> {
        let _ = backends;
        let output = vec![DispatchAdapter::custom(Adapter::new(
            self.instance_config.clone(),
        ))];
        Box::pin(async { output })
    }

    fn wgsl_language_features(&self) -> wgpu::WgslLanguageFeatures {
        wgpu::WgslLanguageFeatures::empty()
    }
}
