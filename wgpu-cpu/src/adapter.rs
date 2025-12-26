use std::{
    pin::Pin,
    sync::Arc,
};

use crate::{
    device::create_device_and_queue,
    instance::InstanceConfig,
    util::make_label_owned,
};

#[derive(Debug)]
pub struct Adapter {
    instance_config: Arc<InstanceConfig>,
}

impl Adapter {
    pub fn new(instance_config: Arc<InstanceConfig>) -> Self {
        Self { instance_config }
    }
}

impl wgpu::custom::AdapterInterface for Adapter {
    fn request_device(
        &self,
        desc: &wgpu::DeviceDescriptor<'_>,
    ) -> Pin<Box<dyn wgpu::custom::RequestDeviceFuture>> {
        let descriptor = desc.map_label(make_label_owned);
        let instance_config = self.instance_config.clone();

        Box::pin(async move {
            check_features(&descriptor.required_features)?;
            check_limits(&descriptor.required_limits)?;

            let (device, queue) = create_device_and_queue(descriptor, instance_config)?;

            Ok((
                wgpu::custom::DispatchDevice::custom(device),
                wgpu::custom::DispatchQueue::custom(queue),
            ))
        })
    }

    fn is_surface_supported(&self, surface: &wgpu::custom::DispatchSurface) -> bool {
        #![allow(unused)]
        let mut supported = false;

        #[cfg(feature = "softbuffer")]
        {
            supported = surface.as_custom::<crate::surface::Surface>().is_some();
        }

        supported
    }

    fn features(&self) -> wgpu::Features {
        wgpu::Features::default()
    }

    fn limits(&self) -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn downlevel_capabilities(&self) -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities::default()
    }

    fn get_info(&self) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: "wgpu-cpu".to_owned(),
            vendor: 0,
            device: 0,
            device_type: wgpu::DeviceType::Cpu,
            device_pci_bus_id: "".to_owned(),
            driver: "wgpu-cpu".to_owned(),
            driver_info: "".to_owned(),
            // todo: we can't really specify something custom here
            backend: wgpu::Backend::Noop,
            subgroup_min_size: wgpu::MINIMUM_SUBGROUP_MIN_SIZE,
            subgroup_max_size: wgpu::MAXIMUM_SUBGROUP_MAX_SIZE,
            transient_saves_memory: false,
        }
    }

    fn get_texture_format_features(
        &self,
        format: wgpu::TextureFormat,
    ) -> wgpu::TextureFormatFeatures {
        wgpu::TextureFormatFeatures {
            allowed_usages: wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT,
            flags: wgpu::TextureFormatFeatureFlags::empty(),
        }
    }

    fn get_presentation_timestamp(&self) -> wgpu::PresentationTimestamp {
        todo!()
    }

    fn cooperative_matrix_properties(&self) -> Vec<wgpu::wgt::CooperativeMatrixProperties> {
        todo!()
    }
}

fn check_features(required_features: &wgpu::Features) -> Result<(), wgpu::RequestDeviceError> {
    // todo
    Ok(())
}

fn check_limits(required_limits: &wgpu::Limits) -> Result<(), wgpu::RequestDeviceError> {
    // todo
    Ok(())
}
