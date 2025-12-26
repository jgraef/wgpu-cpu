use std::sync::Arc;

use naga_cranelift::{
    CompiledModule,
    bindings::{
        BindingLocation,
        UserDefinedInterStageLayout,
    },
    compile_jit,
};
use parking_lot::Mutex;

use crate::shader::memory::{
    ReadMemory,
    WriteMemory,
};

#[derive(Clone, Debug)]
pub struct ShaderModule {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    module: naga::Module,
    info: naga::valid::ModuleInfo,
    compilation_info: wgpu::CompilationInfo,
}

impl ShaderModule {
    pub fn new(
        shader_source: wgpu::ShaderSource,
        shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> Result<Self, Error> {
        let module = match shader_source {
            wgpu::ShaderSource::Wgsl(wgsl) => naga::front::wgsl::parse_str(&wgsl)?,
            _ => return Err(Error::Unsupported),
        };

        let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
        let info = validator.validate(&module).unwrap();

        Ok(Self {
            inner: Arc::new(Inner {
                module,
                info,
                compilation_info: wgpu::CompilationInfo { messages: vec![] },
            }),
        })
    }

    pub fn for_pipeline(
        &self,
        pipeline_compilation_options: &wgpu::PipelineCompilationOptions,
    ) -> Result<CompiledModule, Error> {
        let constants = pipeline_compilation_options
            .constants
            .iter()
            .map(|(key, value)| ((*key).to_owned(), *value))
            .collect();

        let (module, info) = naga::back::pipeline_constants::process_overrides(
            &self.inner.module,
            &self.inner.info,
            None,
            &constants,
        )?;

        // todo: get necessary info for early depth test here and store it

        let module = compile_jit(&module, &info)?;

        Ok(module)
    }
}

impl wgpu::custom::ShaderModuleInterface for ShaderModule {
    fn get_compilation_info(
        &self,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::ShaderCompilationInfoFuture>> {
        let compilation_info = self.inner.compilation_info.clone();
        Box::pin(async move { compilation_info })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Provided shader source variant is not supported")]
    Unsupported,

    #[error(transparent)]
    ParseError(#[from] naga::front::wgsl::ParseError),

    #[error(transparent)]
    Validation(#[from] naga::WithSpan<naga::valid::ValidationError>),

    #[error(transparent)]
    PipelineConstants(#[from] naga::back::pipeline_constants::PipelineConstantError),

    #[error(transparent)]
    Compiler(#[from] naga_cranelift::Error),
}

#[derive(Clone, Debug)]
pub struct UserDefinedIoBufferPool {
    layout: UserDefinedInterStageLayout,
    free: Arc<Mutex<Vec<UserDefinedInterStageBuffer>>>,
}

impl UserDefinedIoBufferPool {
    pub fn new(layout: UserDefinedInterStageLayout) -> Self {
        Self {
            layout,
            free: Default::default(),
        }
    }

    pub fn allocate(&self) -> UserDefinedInterStagePoolBufferMut {
        let mut free = self.free.lock();
        let buffer = free
            .pop()
            .unwrap_or_else(|| UserDefinedInterStageBuffer::new(self.layout.clone()));
        UserDefinedInterStagePoolBufferMut {
            inner: UserDefinedInterStagePoolBufferInner {
                buffer,
                free: self.free.clone(),
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageBuffer {
    data: Vec<u8>,
    layout: UserDefinedInterStageLayout,
}

impl UserDefinedInterStageBuffer {
    pub fn new(layout: UserDefinedInterStageLayout) -> Self {
        let data = vec![0; layout.size() as usize];
        Self { data, layout }
    }
}

impl ReadMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn read(&self, address: BindingLocation) -> &[u8] {
        let layout = self.layout[address];
        &self.data[layout.range()]
    }
}

impl WriteMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn write(&mut self, address: BindingLocation) -> &mut [u8] {
        let layout = self.layout[address];
        &mut self.data[layout.range()]
    }
}

#[derive(Debug)]
struct UserDefinedInterStagePoolBufferInner {
    buffer: UserDefinedInterStageBuffer,
    free: Arc<Mutex<Vec<UserDefinedInterStageBuffer>>>,
}

impl Drop for UserDefinedInterStagePoolBufferInner {
    fn drop(&mut self) {
        let mut free = self.free.lock();
        free.push(std::mem::take(&mut self.buffer));
    }
}

#[derive(Debug)]
pub struct UserDefinedInterStagePoolBufferMut {
    inner: UserDefinedInterStagePoolBufferInner,
}

impl UserDefinedInterStagePoolBufferMut {
    pub fn read_only(self) -> UserDefinedInterStagePoolBuffer {
        UserDefinedInterStagePoolBuffer {
            inner: Arc::new(self.inner),
        }
    }
}

impl ReadMemory<BindingLocation> for UserDefinedInterStagePoolBufferMut {
    fn read(&self, address: BindingLocation) -> &[u8] {
        self.inner.buffer.read(address)
    }
}

impl WriteMemory<BindingLocation> for UserDefinedInterStagePoolBufferMut {
    fn write(&mut self, address: BindingLocation) -> &mut [u8] {
        self.inner.buffer.write(address)
    }
}

#[derive(Clone, Debug)]
pub struct UserDefinedInterStagePoolBuffer {
    inner: Arc<UserDefinedInterStagePoolBufferInner>,
}

impl ReadMemory<BindingLocation> for UserDefinedInterStagePoolBuffer {
    fn read(&self, address: BindingLocation) -> &[u8] {
        self.inner.buffer.read(address)
    }
}

pub mod memory {
    // do we need this? should we move this? this is leftover from when we still had
    // an interpreter

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
}
