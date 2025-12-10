use std::sync::mpsc;

#[derive(Clone, Debug)]
pub struct Notify {
    sender: mpsc::SyncSender<()>,
}

impl Notify {
    pub fn notify(&self) {
        let _ = self.sender.send(());
    }
}

#[derive(Debug)]
pub struct Wait {
    receiver: mpsc::Receiver<()>,
}

impl Wait {
    pub fn wait(self) -> Result<(), Closed> {
        self.receiver.recv().map_err(|_| Closed)
    }
}

pub fn channel() -> (Notify, Wait) {
    let (sender, receiver) = mpsc::sync_channel(1);
    (Notify { sender }, Wait { receiver })
}

#[derive(Debug)]
pub struct Closed;
