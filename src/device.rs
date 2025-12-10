use std::{
    pin::Pin,
    sync::Arc,
};

use parking_lot::Mutex;

use crate::{
    command::CommandEncoder,
    make_label_owned,
};

#[derive(Debug)]
pub struct Device {
    descriptor: wgpu::wgt::DeviceDescriptor<Option<String>>,
    state: Mutex<State>,
}

impl Device {
    pub fn new(descriptor: wgpu::wgt::DeviceDescriptor<Option<String>>) -> Self {
        Self {
            descriptor,
            state: Default::default(),
        }
    }
}

impl wgpu::custom::DeviceInterface for Device {
    fn features(&self) -> wgpu::Features {
        self.descriptor.required_features
    }

    fn limits(&self) -> wgpu::Limits {
        self.descriptor.required_limits.clone()
    }

    fn create_shader_module(
        &self,
        desc: wgpu::ShaderModuleDescriptor<'_>,
        shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> wgpu::custom::DispatchShaderModule {
        todo!()
    }

    unsafe fn create_shader_module_passthrough(
        &self,
        desc: &wgpu::ShaderModuleDescriptorPassthrough<'_>,
    ) -> wgpu::custom::DispatchShaderModule {
        todo!()
    }

    fn create_bind_group_layout(
        &self,
        desc: &wgpu::BindGroupLayoutDescriptor<'_>,
    ) -> wgpu::custom::DispatchBindGroupLayout {
        todo!()
    }

    fn create_bind_group(
        &self,
        desc: &wgpu::BindGroupDescriptor<'_>,
    ) -> wgpu::custom::DispatchBindGroup {
        todo!()
    }

    fn create_pipeline_layout(
        &self,
        desc: &wgpu::PipelineLayoutDescriptor<'_>,
    ) -> wgpu::custom::DispatchPipelineLayout {
        todo!()
    }

    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPipeline {
        todo!()
    }

    fn create_mesh_pipeline(
        &self,
        desc: &wgpu::MeshPipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPipeline {
        todo!()
    }

    fn create_compute_pipeline(
        &self,
        desc: &wgpu::ComputePipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchComputePipeline {
        todo!()
    }

    unsafe fn create_pipeline_cache(
        &self,
        desc: &wgpu::PipelineCacheDescriptor<'_>,
    ) -> wgpu::custom::DispatchPipelineCache {
        todo!()
    }

    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> wgpu::custom::DispatchBuffer {
        todo!()
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::custom::DispatchTexture {
        todo!()
    }

    fn create_external_texture(
        &self,
        desc: &wgpu::ExternalTextureDescriptor<'_>,
        planes: &[&wgpu::TextureView],
    ) -> wgpu::custom::DispatchExternalTexture {
        todo!()
    }

    fn create_blas(
        &self,
        desc: &wgpu::CreateBlasDescriptor<'_>,
        sizes: wgpu::BlasGeometrySizeDescriptors,
    ) -> (Option<u64>, wgpu::custom::DispatchBlas) {
        todo!()
    }

    fn create_tlas(&self, desc: &wgpu::CreateTlasDescriptor<'_>) -> wgpu::custom::DispatchTlas {
        todo!()
    }

    fn create_sampler(&self, desc: &wgpu::SamplerDescriptor<'_>) -> wgpu::custom::DispatchSampler {
        todo!()
    }

    fn create_query_set(
        &self,
        desc: &wgpu::QuerySetDescriptor<'_>,
    ) -> wgpu::custom::DispatchQuerySet {
        todo!()
    }

    fn create_command_encoder(
        &self,
        desc: &wgpu::CommandEncoderDescriptor<'_>,
    ) -> wgpu::custom::DispatchCommandEncoder {
        wgpu::custom::DispatchCommandEncoder::custom(CommandEncoder::new(
            desc.map_label(make_label_owned),
        ))
    }

    fn create_render_bundle_encoder(
        &self,
        desc: &wgpu::RenderBundleEncoderDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderBundleEncoder {
        todo!()
    }

    fn set_device_lost_callback(&self, device_lost_callback: wgpu::custom::BoxDeviceLostCallback) {
        todo!()
    }

    fn on_uncaptured_error(&self, handler: Arc<dyn wgpu::UncapturedErrorHandler>) {
        todo!()
    }

    fn push_error_scope(&self, filter: wgpu::ErrorFilter) -> u32 {
        todo!()
    }

    fn pop_error_scope(&self, index: u32) -> Pin<Box<dyn wgpu::custom::PopErrorScopeFuture>> {
        todo!()
    }

    unsafe fn start_graphics_debugger_capture(&self) {
        todo!()
    }

    unsafe fn stop_graphics_debugger_capture(&self) {
        todo!()
    }

    fn poll(
        &self,
        poll_type: wgpu::wgt::PollType<u64>,
    ) -> Result<wgpu::PollStatus, wgpu::PollError> {
        todo!()
    }

    fn get_internal_counters(&self) -> wgpu::InternalCounters {
        todo!()
    }

    fn generate_allocator_report(&self) -> Option<wgpu::AllocatorReport> {
        todo!()
    }

    fn destroy(&self) {
        // nop
    }
}

#[derive(derive_more::Debug)]
struct State {
    #[debug(skip)]
    uncaptured_error_handler: Arc<dyn wgpu::UncapturedErrorHandler>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            uncaptured_error_handler: Arc::new(|error| {
                panic!("Uncaptured wgpu error: {error}");
            }),
        }
    }
}
