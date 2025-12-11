use crate::{
    command::Command,
    device::QueueReceiver,
};

#[derive(Debug)]
pub struct Engine {
    receiver: QueueReceiver,
}

impl Engine {
    pub fn spawn(receiver: QueueReceiver) -> Result<(), std::io::Error> {
        let engine = Engine { receiver };

        let _join_handle = std::thread::Builder::new()
            .name("wgpu-cpu engine".to_owned())
            .spawn(move || {
                tracing::debug!("engine thread started");
                engine.run();
                tracing::debug!("engine thread exited");
            })?;

        Ok(())
    }

    pub fn run(mut self) {
        let mut last_processed = None;

        while let Some(submission) = self.receiver.receive(last_processed) {
            tracing::debug!(?submission.submission_index, "processing submission");
            last_processed = Some(submission.submission_index);
            for commands in submission.command_buffers {
                self.execute_submission(commands);
            }
        }
    }

    pub fn execute_submission(&mut self, commands: Vec<Command>) {
        for command in commands {
            tracing::debug!(?command, "executing command");

            match command {
                Command::RenderPass(command) => command.execute(),
            }
        }
    }
}
