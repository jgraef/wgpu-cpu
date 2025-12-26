use std::{
    collections::{
        HashSet,
        VecDeque,
    },
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use parking_lot::{
    Condvar,
    Mutex,
    MutexGuard,
};

use crate::{
    buffer::Buffer,
    command::{
        Command,
        CommandBuffer,
        CommandEncoder,
    },
    engine::Engine,
    instance::InstanceConfig,
    pipeline::{
        PipelineLayout,
        RenderPipeline,
    },
    shader::ShaderModule,
    texture::Texture,
    util::make_label_owned,
};

pub fn create_device_and_queue(
    descriptor: wgpu::wgt::DeviceDescriptor<Option<String>>,
    instance_config: Arc<InstanceConfig>,
) -> Result<(Device, Queue), wgpu::RequestDeviceError> {
    let inner = Arc::new(DeviceInner {
        state: Mutex::new(DeviceState {
            uncaptured_error_handler: Arc::new(|error| {
                panic!("Uncaptured wgpu error: {error}");
            }),
            queue_state: QueueState {
                next_submission_index: 0,
                queue: VecDeque::new(),
                has_receiver: true,
                sender_count: 1,
                inflight_submissions: HashSet::new(),
            },
        }),
        queue_submitted: Condvar::new(),
        queue_processed: Condvar::new(),
        instance_config,
    });

    Engine::spawn(QueueReceiver {
        inner: inner.clone(),
    })
    .map_err(|error| wgpu::RequestDeviceError::custom(error.to_string()))?;

    Ok((
        Device {
            descriptor,
            inner: inner.clone(),
        },
        Queue { inner },
    ))
}

#[derive(Debug)]
pub struct Device {
    descriptor: wgpu::wgt::DeviceDescriptor<Option<String>>,
    inner: Arc<DeviceInner>,
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
        wgpu::custom::DispatchShaderModule::custom(
            ShaderModule::new(desc.source, shader_bound_checks).unwrap(),
        )
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
        wgpu::custom::DispatchPipelineLayout::custom(PipelineLayout::new(desc))
    }

    fn create_render_pipeline(
        &self,
        desc: &wgpu::RenderPipelineDescriptor<'_>,
    ) -> wgpu::custom::DispatchRenderPipeline {
        wgpu::custom::DispatchRenderPipeline::custom(RenderPipeline::new(desc).unwrap())
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
        wgpu::custom::DispatchBuffer::custom(Buffer::new_mapped_at_creation(desc.size as usize))
    }

    fn create_texture(&self, desc: &wgpu::TextureDescriptor<'_>) -> wgpu::custom::DispatchTexture {
        wgpu::custom::DispatchTexture::custom(Texture::new(desc.size, desc.format))
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
        // nop: we don't need that since we never lose the device... well if we
        // did, we won't have to worry about notifying the user :D
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
        fn wait<'a>(
            guard: &mut MutexGuard<'a, DeviceState>,
            condition: &Condvar,
            timeout: Option<Duration>,
        ) -> Result<(), wgpu::PollError> {
            if let Some(timeout) = timeout {
                let result = condition.wait_for(guard, timeout);
                if result.timed_out() {
                    return Err(wgpu::PollError::Timeout);
                }
            }
            else {
                condition.wait(guard);
            }
            Ok(())
        }

        match poll_type {
            wgpu::wgt::PollType::Wait {
                submission_index,
                timeout,
            } => {
                let mut device_state = self.inner.state.lock();

                loop {
                    device_state.queue_state.assert_engine_alive();

                    if device_state.queue_state.inflight_submissions.is_empty() {
                        return Ok(wgpu::PollStatus::QueueEmpty);
                    }

                    if let Some(submission_index) = submission_index {
                        if !device_state
                            .queue_state
                            .inflight_submissions
                            .contains(&submission_index)
                        {
                            return Ok(wgpu::PollStatus::WaitSucceeded);
                        }
                        else {
                            wait(&mut device_state, &self.inner.queue_processed, timeout)?;
                        }
                    }
                    else {
                        wait(&mut device_state, &self.inner.queue_processed, timeout)?;
                        return Ok(wgpu::PollStatus::WaitSucceeded);
                    }
                }
            }
            wgpu::wgt::PollType::Poll => {
                // nop
                Ok(wgpu::PollStatus::Poll)
            }
        }
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

#[derive(Debug)]
pub struct DeviceInner {
    state: Mutex<DeviceState>,
    queue_submitted: Condvar,
    queue_processed: Condvar,
    instance_config: Arc<InstanceConfig>,
}

#[derive(derive_more::Debug)]
pub struct DeviceState {
    #[debug(skip)]
    uncaptured_error_handler: Arc<dyn wgpu::UncapturedErrorHandler>,

    queue_state: QueueState,
}

#[derive(Clone, Debug)]
pub struct Queue {
    inner: Arc<DeviceInner>,
}

impl wgpu::custom::QueueInterface for Queue {
    fn write_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
    ) {
        todo!()
    }

    fn create_staging_buffer(
        &self,
        size: wgpu::BufferSize,
    ) -> Option<wgpu::custom::DispatchQueueWriteBuffer> {
        todo!()
    }

    fn validate_write_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        size: wgpu::BufferSize,
    ) -> Option<()> {
        todo!()
    }

    fn write_staging_buffer(
        &self,
        buffer: &wgpu::custom::DispatchBuffer,
        offset: wgpu::BufferAddress,
        staging_buffer: &wgpu::custom::DispatchQueueWriteBuffer,
    ) {
        todo!()
    }

    fn write_texture(
        &self,
        texture: wgpu::TexelCopyTextureInfo<'_>,
        data: &[u8],
        data_layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) {
        todo!()
    }

    fn submit(
        &self,
        command_buffers: &mut dyn Iterator<Item = wgpu::custom::DispatchCommandBuffer>,
    ) -> u64 {
        let mut device_state = self.inner.state.lock();
        device_state.queue_state.assert_engine_alive();

        let submission_index = device_state.queue_state.next_submission_index;
        device_state.queue_state.next_submission_index += 1;

        let command_buffers = command_buffers
            .map(|command_buffer| command_buffer.as_custom::<CommandBuffer>().unwrap().take())
            .collect::<Vec<_>>();

        device_state.queue_state.queue.push_back(Submission {
            command_buffers,
            submission_index,
        });
        device_state
            .queue_state
            .inflight_submissions
            .insert(submission_index);

        self.inner.queue_submitted.notify_all();

        submission_index
    }

    fn get_timestamp_period(&self) -> f32 {
        todo!()
    }

    fn on_submitted_work_done(&self, callback: wgpu::custom::BoxSubmittedWorkDoneCallback) {
        todo!()
    }

    fn compact_blas(
        &self,
        blas: &wgpu::custom::DispatchBlas,
    ) -> (Option<u64>, wgpu::custom::DispatchBlas) {
        todo!()
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        tracing::debug!("QueueSender dropped");
        let mut device_state = self.inner.state.lock();
        device_state.queue_state.sender_count -= 1;
        self.inner.queue_submitted.notify_all();
    }
}

#[derive(Debug)]
pub struct QueueState {
    next_submission_index: u64,
    queue: VecDeque<Submission>,
    has_receiver: bool,
    sender_count: usize,
    inflight_submissions: HashSet<u64>,
}

impl QueueState {
    pub fn assert_engine_alive(&self) {
        if !self.has_receiver {
            panic!("Execution engine dead");
        }
    }
}

#[derive(Debug)]
pub struct Submission {
    pub command_buffers: Vec<Vec<Command>>,
    pub submission_index: u64,
}

#[derive(Debug)]
pub struct QueueReceiver {
    inner: Arc<DeviceInner>,
}

impl QueueReceiver {
    pub fn receive(&self, processed: Option<u64>) -> Option<Submission> {
        let mut device_state = self.inner.state.lock();

        if let Some(processed) = processed {
            device_state
                .queue_state
                .inflight_submissions
                .remove(&processed);
            self.inner.queue_processed.notify_all();
        }

        loop {
            if let Some(item) = device_state.queue_state.queue.pop_front() {
                return Some(item);
            }
            else if device_state.queue_state.sender_count == 0 {
                return None;
            }
            else {
                self.inner.queue_submitted.wait(&mut device_state);
            }
        }
    }
}

impl Drop for QueueReceiver {
    fn drop(&mut self) {
        tracing::debug!("QueueReceiver dropped");
        let mut device_state = self.inner.state.lock();
        device_state.queue_state.has_receiver = false;
        self.inner.queue_processed.notify_all();
    }
}
