use std::fmt::Debug;

pub trait ReadMemory<A> {
    fn read(&self, address: A) -> &[u8];
}

impl<T, A> ReadMemory<A> for &T
where
    T: ReadMemory<A>,
{
    fn read(&self, address: A) -> &[u8] {
        T::read(self, address)
    }
}

impl<T, A> ReadMemory<A> for &mut T
where
    T: ReadMemory<A>,
{
    fn read(&self, address: A) -> &[u8] {
        T::read(self, address)
    }
}

pub trait WriteMemory<A> {
    fn write(&mut self, address: A) -> &mut [u8];
}

impl<T, A> WriteMemory<A> for &mut T
where
    T: WriteMemory<A>,
{
    fn write(&mut self, address: A) -> &mut [u8] {
        T::write(self, address)
    }
}

pub trait ReadWriteMemory<A>: ReadMemory<A> + WriteMemory<A> {
    fn copy(&mut self, source: A, target: A);
}

impl<T, A> ReadWriteMemory<A> for &mut T
where
    T: ReadWriteMemory<A>,
{
    fn copy(&mut self, source: A, target: A) {
        T::copy(self, source, target)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullMemory;

impl<A> ReadMemory<A> for NullMemory
where
    A: Debug,
{
    fn read(&self, address: A) -> &[u8] {
        panic!("Attempt to read from NullMemory: {address:?}");
    }
}

impl<A> WriteMemory<A> for NullMemory
where
    A: Debug,
{
    fn write(&mut self, address: A) -> &mut [u8] {
        panic!("Attempt to write to NullMemory: {address:?}",);
    }
}

impl<A> ReadWriteMemory<A> for NullMemory
where
    A: Debug,
{
    fn copy(&mut self, source: A, target: A) {
        panic!("Attempt to copy in NullMemory: From {source:?} to {target:?}");
    }
}
