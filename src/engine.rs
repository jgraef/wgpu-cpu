use crate::{
    QueueSender,
    command::Command,
    queue::{
        QueueReceiver,
        Submission,
        create_queue,
    },
};

#[derive(Debug)]
pub struct Engine {
    receiver: QueueReceiver,
}

impl Engine {
    pub fn spawn() -> Result<QueueSender, std::io::Error> {
        let (sender, receiver) = create_queue();

        let engine = Engine { receiver };

        let _join_handle = std::thread::Builder::new()
            .name("wgpu-cpu engine".to_owned())
            .spawn(move || {
                tracing::debug!("engine thread started");
                engine.run();
                tracing::debug!("engine thread exited");
            })?;

        Ok(sender)
    }

    pub fn run(mut self) {
        while let Some(submission) = self.receiver.receive() {
            self.execute_submission(submission);
            //
        }
    }

    pub fn execute_submission(&mut self, submission: Submission) {
        for command in submission.commands {
            tracing::debug!(?command, "executing command");

            match command {
                Command::RenderPass(command) => command.execute(),
            }
        }
    }
}
