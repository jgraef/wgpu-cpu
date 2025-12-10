use std::{
    collections::VecDeque,
    sync::Arc,
};

use parking_lot::{
    Condvar,
    Mutex,
};
use wgpu::custom::QueueInterface;

use crate::command::{
    Command,
    CommandBuffer,
};

#[derive(Debug)]
pub struct QueueSender {
    shared: Arc<Shared>,
}

impl QueueInterface for QueueSender {
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
        let mut state = self.shared.state.lock();
        let submission_index = state.next_submission_index;
        state.next_submission_index += 1;

        let mut any_submissions = false;
        for command_buffer in command_buffers {
            let command_buffer = command_buffer.as_custom::<CommandBuffer>().unwrap();
            let commands = command_buffer.take();
            if !commands.is_empty() {
                tracing::debug!(?commands, "submitting");
                state.queue.push_back(Submission {
                    commands,
                    submission_index,
                });
                any_submissions = true;
            }
        }

        if !any_submissions {
            // put in one empty submissions so we at least get feedback on when it's done
            // todo: of course we could just mark it as done immediately
            state.queue.push_back(Submission {
                commands: vec![],
                submission_index,
            });
        }

        self.shared.condition.notify_all();

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

impl Drop for QueueSender {
    fn drop(&mut self) {
        tracing::debug!("Queue dropped");
        let mut state = self.shared.state.lock();
        state.closed = true;
        self.shared.condition.notify_all();
    }
}

#[derive(Debug, Default)]
struct SharedState {
    next_submission_index: u64,
    queue: VecDeque<Submission>,
    closed: bool,
}

#[derive(Debug, Default)]
struct Shared {
    state: Mutex<SharedState>,
    condition: Condvar,
}

#[derive(Debug)]
pub struct Submission {
    pub commands: Vec<Command>,
    pub submission_index: u64,
}

#[derive(Debug)]
pub struct QueueReceiver {
    shared: Arc<Shared>,
}

impl QueueReceiver {
    pub fn receive(&self) -> Option<Submission> {
        let mut state = self.shared.state.lock();

        while !state.closed {
            if let Some(item) = state.queue.pop_front() {
                return Some(item);
            }
            else {
                self.shared.condition.wait(&mut state);
            }
        }

        None
    }
}

pub fn create_queue() -> (QueueSender, QueueReceiver) {
    let shared = Arc::new(Shared::default());
    (
        QueueSender {
            shared: shared.clone(),
        },
        QueueReceiver { shared },
    )
}
