use std::sync::Arc;

use naga_interpreter::{
    bindings::{
        BindingLocation,
        UserDefinedInterStageBuffer,
        UserDefinedInterStageLayout,
    },
    memory::{
        ReadMemory,
        WriteMemory,
    },
};
use parking_lot::Mutex;

#[derive(Clone, Debug)]
pub struct ShaderModule {
    inner: Arc<Inner>,
}

#[derive(Debug)]
struct Inner {
    module: naga_interpreter::interpreter::ShaderModule,
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

        Ok(Self {
            inner: Arc::new(Inner {
                module: naga_interpreter::interpreter::ShaderModule::new(module)?,
                compilation_info: wgpu::CompilationInfo { messages: vec![] },
            }),
        })
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

impl AsRef<naga_interpreter::interpreter::ShaderModule> for ShaderModule {
    fn as_ref(&self) -> &naga_interpreter::interpreter::ShaderModule {
        &self.inner.module
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Provided shader source variant is not supported")]
    Unsupported,

    #[error(transparent)]
    ParseError(#[from] naga::front::wgsl::ParseError),

    #[error(transparent)]
    Module(#[from] naga_interpreter::interpreter::ModuleError),
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
