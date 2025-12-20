use std::sync::Arc;

use naga_interpreter::{
    bindings::{
        BindingLocation,
        ShaderInput,
        ShaderOutput,
        UserDefinedInterStageBuffer,
        UserDefinedInterStageLayout,
    },
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        InterStageLayout,
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
    module: naga::Module,
    info: naga::valid::ModuleInfo,
    compilation_info: wgpu::CompilationInfo,
    backend: ShaderBackend,
}

impl ShaderModule {
    pub fn new(
        backend: ShaderBackend,
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
                backend,
            }),
        })
    }

    pub fn for_pipeline(
        &self,
        pipeline_compilation_options: &wgpu::PipelineCompilationOptions,
    ) -> Result<PipelineShaderModule, Error> {
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

        let module = match self.inner.backend {
            ShaderBackend::Interpreter => {
                PipelineShaderModule::Interpreted {
                    module: naga_interpreter::interpreter::InterpretedModule::new(
                        module.into_owned(),
                        info.into_owned(),
                    )?,
                }
            }
            ShaderBackend::Compiler => {
                PipelineShaderModule::Compiled {
                    module: naga_interpreter::compiler::compile_jit(&module, &info)?,
                }
            }
        };

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
    Interpreter(#[from] naga_interpreter::interpreter::Error),

    #[error(transparent)]
    Compiler(#[from] naga_interpreter::compiler::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShaderBackend {
    #[default]
    Interpreter,
    Compiler,
}

#[derive(Clone, Debug)]
pub enum PipelineShaderModule {
    Interpreted {
        module: naga_interpreter::interpreter::InterpretedModule,
    },
    Compiled {
        module: naga_interpreter::compiler::CompiledModule,
    },
}

impl naga_interpreter::backend::Module for PipelineShaderModule {
    fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        match self {
            PipelineShaderModule::Interpreted { module } => module.find_entry_point(name, stage),
            PipelineShaderModule::Compiled { module } => module.find_entry_point(name, stage),
        }
    }

    fn run_entry_point<I, O>(&self, index: EntryPointIndex, input: I, output: O)
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        match self {
            PipelineShaderModule::Interpreted { module } => {
                module.run_entry_point(index, input, output)
            }
            PipelineShaderModule::Compiled { module } => {
                module.run_entry_point(index, input, output)
            }
        }
    }

    fn inter_stage_layout(&self, entry_point: EntryPointIndex) -> Option<&InterStageLayout> {
        match self {
            PipelineShaderModule::Interpreted { module } => module.inter_stage_layout(entry_point),
            PipelineShaderModule::Compiled { module } => module.inter_stage_layout(entry_point),
        }
    }

    fn early_depth_test(&self, entry_point: EntryPointIndex) -> Option<naga::EarlyDepthTest> {
        match self {
            PipelineShaderModule::Interpreted { module } => module.early_depth_test(entry_point),
            PipelineShaderModule::Compiled { module } => module.early_depth_test(entry_point),
        }
    }
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
