use std::{
    ops::{
        Deref,
        DerefMut,
    },
    sync::Arc,
};

use parking_lot::{
    RwLock,
    RwLockReadGuard,
    RwLockWriteGuard,
};

#[derive(derive_more::Debug, Clone)]
pub struct Buffer {
    #[debug(skip)]
    data: Arc<RwLock<Vec<u8>>>,
    size: usize,
}

impl Buffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: Arc::new(RwLock::new(vec![0; size])),
            size,
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn read(&self) -> BufferReadGuard<'_> {
        BufferReadGuard {
            guard: self.data.read(),
        }
    }

    pub fn write(&self) -> BufferWriteGuard<'_> {
        BufferWriteGuard {
            guard: self.data.write(),
        }
    }
}

#[derive(Debug)]
pub struct BufferReadGuard<'a> {
    guard: RwLockReadGuard<'a, Vec<u8>>,
}

impl<'a> Deref for BufferReadGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

#[derive(Debug)]
pub struct BufferWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Vec<u8>>,
}

impl<'a> Deref for BufferWriteGuard<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.guard
    }
}

impl<'a> DerefMut for BufferWriteGuard<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.guard
    }
}
