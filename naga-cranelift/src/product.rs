use std::{
    collections::HashMap,
    sync::Arc,
};

use cranelift_jit::JITModule;
use cranelift_module::FuncId;

use crate::{
    bindings::{
        BindingResources,
        InterStageLayout,
        ShaderInput,
        ShaderOutput,
    },
    function::{
        AbortCode,
        AbortPayload,
    },
    runtime::{
        BindingStackLayout,
        DefaultRuntime,
        DefaultRuntimeError,
        Runtime,
        RuntimeContext,
        RuntimeData,
    },
    variable::PrivateMemoryLayout,
};

#[derive(Clone, Copy, Debug)]
pub struct EntryPointIndex {
    pub(crate) index: usize,
}

impl From<EntryPointIndex> for usize {
    fn from(value: EntryPointIndex) -> Self {
        value.index
    }
}

#[cfg(test)]
impl From<usize> for EntryPointIndex {
    fn from(value: usize) -> Self {
        Self { index: value }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EntryPointNotFound {
    #[error("Entry point '{name}' not found")]
    NameNotFound { name: String },
    #[error("No entry point for shader stage {stage:?} found")]
    NotFound { stage: naga::ShaderStage },
    #[error("There are multiple entry points for this shader stage: {stage:?}")]
    NotUnique { stage: naga::ShaderStage },
    #[error(
        "Found entry point '{name}', but it is for shader stage {module_stage:?}, and not {expected_stage:?}"
    )]
    WrongStage {
        name: String,
        module_stage: naga::ShaderStage,
        expected_stage: naga::ShaderStage,
    },
}

#[derive(Clone, Debug)]
pub struct CompiledModule {
    inner: Arc<CompiledModuleInner>,
}

impl CompiledModule {
    /// # Safety
    ///
    /// This is unsafe because we want to guarantuee that these the contained
    /// entry points are safe to run at creation.
    pub unsafe fn new(
        jit_module: JITModule,
        entry_points: Vec<CompiledEntryPoint>,
        private_memory_layout: PrivateMemoryLayout,
    ) -> Self {
        let mut entry_points_by_name = HashMap::with_capacity(entry_points.len());
        let mut entry_points_by_stage = HashMap::with_capacity(entry_points.len());
        for (i, entry_point) in entry_points.iter().enumerate() {
            entry_points_by_name.insert(entry_point.name.clone(), i);
            entry_points_by_stage.insert(entry_point.stage, i);
        }

        Self {
            inner: Arc::new(CompiledModuleInner {
                jit_module: Some(jit_module),
                entry_points,
                entry_points_by_name,
                entry_points_by_stage,
                private_memory_layout,
            }),
        }
    }

    pub fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        let index = if let Some(name) = name {
            let index = *self.inner.entry_points_by_name.get(name).ok_or_else(|| {
                EntryPointNotFound::NameNotFound {
                    name: name.to_owned(),
                }
            })?;

            let item = &self.inner.entry_points[index];
            if item.stage != stage {
                return Err(EntryPointNotFound::WrongStage {
                    name: name.to_owned(),
                    module_stage: item.stage,
                    expected_stage: stage,
                });
            }

            index
        }
        else {
            *self
                .inner
                .entry_points_by_stage
                .get(&stage)
                .ok_or_else(|| EntryPointNotFound::NotFound { stage })?
        };

        Ok(EntryPointIndex { index })
    }

    pub fn entry_point(&self, index: EntryPointIndex) -> EntryPoint<'_> {
        let inner = &self.inner.entry_points[index.index];

        let function_pointer = self
            .inner
            .jit_module
            .as_ref()
            .expect("JIT module gone")
            .get_finalized_function(inner.function_id);

        EntryPoint {
            inner,
            function_pointer,
            private_memory_layout: &self.inner.private_memory_layout,
        }
    }
}

#[derive(derive_more::Debug)]
struct CompiledModuleInner {
    // this is in an Option, so we can take it out on Drop. MaybeUninit would work too
    #[debug(skip)]
    jit_module: Option<JITModule>,

    entry_points: Vec<CompiledEntryPoint>,
    entry_points_by_name: HashMap<String, usize>,
    entry_points_by_stage: HashMap<naga::ShaderStage, usize>,

    private_memory_layout: PrivateMemoryLayout,
}

impl Drop for CompiledModuleInner {
    fn drop(&mut self) {
        // JITModule leaks the memory when dropped:
        //
        // https://docs.rs/cranelift-jit/latest/src/cranelift_jit/memory/system.rs.html#222

        let jit_module = self
            .jit_module
            .take()
            .expect("JIT module not here on Drop!");
        unsafe {
            // SAFETY: make sure to keep this around when executing
            jit_module.free_memory();
        }
    }
}

// SAFETY: Make sure to only compile modules that are safe to share. Also
// therefore we must not make the constructor for `CompiledModule` pub.
unsafe impl Sync for CompiledModuleInner {}

// note: the fields can be pub because it doesn't actually do anything with the
// function yet.
#[derive(Debug)]
pub struct CompiledEntryPoint {
    pub name: String,
    pub stage: naga::ShaderStage,
    pub inter_stage_layout: Option<InterStageLayout>,
    pub early_depth_test: Option<naga::EarlyDepthTest>,
    pub function_id: FuncId,
    pub input_layout: Vec<BindingStackLayout>,
    pub output_layout: Vec<BindingStackLayout>,
}

#[derive(Clone, Copy)]
pub struct EntryPoint<'a> {
    inner: &'a CompiledEntryPoint,
    function_pointer: *const u8,
    private_memory_layout: &'a PrivateMemoryLayout,
}

impl<'a> EntryPoint<'a> {
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn stage(&self) -> naga::ShaderStage {
        self.inner.stage
    }

    pub fn function_pointer(&self) -> *const u8 {
        self.function_pointer
    }

    pub fn inter_stage_layout(&self) -> Option<&'a InterStageLayout> {
        self.inner.inter_stage_layout.as_ref()
    }

    pub fn early_depth_test(&self) -> Option<naga::EarlyDepthTest> {
        self.inner.early_depth_test
    }

    pub fn function<R>(&self) -> impl Fn(R) -> Result<(), EntryPointError<R::Error>>
    where
        R: Runtime,
    {
        |runtime: R| {
            let mut runtime_context_data = RuntimeData::new(runtime);
            let mut runtime_context = RuntimeContext::new(&mut runtime_context_data);

            let abort_code = unsafe {
                // SAFETY: We just created the context struct and it's valid until the end of
                // the scope. Thus it's safe to call the function with this as
                // argument. The function itself is safe because `EntryPoint`s
                // can only be created from `CompiledModule`s, which can only be
                // created with an safety guarantuee that the compiled code is safe,
                // takes 1 pointer arguments and returns nothing.

                let function = std::mem::transmute::<
                    _,
                    unsafe extern "C" fn(*const RuntimeContext) -> AbortCode,
                >(self.function_pointer);

                // note: the order of these arguments must be synchronized with the compiled
                // code, which is generated in [`Compiler::compile_entry_point`].
                function(&mut runtime_context as *mut _)
            };

            match (abort_code, runtime_context_data.abort_payload) {
                (AbortCode::Ok, None) => Ok(()),
                (AbortCode::RuntimePanic, Some(AbortPayload::RuntimePanic(payload))) => {
                    std::panic::resume_unwind(payload);
                }
                (AbortCode::RuntimeError, Some(AbortPayload::RuntimeError(error))) => {
                    Err(EntryPointError::RuntimeError(error))
                }
                (AbortCode::Kill, None) => Err(EntryPointError::Killed),
                (AbortCode::PointerOutOfBounds, None) => Err(EntryPointError::PointerOutOfBounds),
                (AbortCode::DivisionByZero, None) => Err(EntryPointError::DivisionByZero),
                (AbortCode::Overflow, None) => Err(EntryPointError::Overflow),
                (abort_code, abort_payload) => {
                    panic!(
                        "Mismatch between abort code and payload: code={abort_code:?}, payload={abort_payload:?}"
                    )
                }
            }
        }
    }

    pub fn run_with_runtime<R>(&self, runtime: R) -> Result<(), EntryPointError<R::Error>>
    where
        R: Runtime,
    {
        self.function()(runtime)
    }

    pub fn run<I, O, B>(
        &self,
        input: I,
        output: O,
        binding_resources: B,
    ) -> Result<(), EntryPointError<DefaultRuntimeError>>
    where
        I: ShaderInput,
        O: ShaderOutput,
        B: BindingResources,
    {
        let runtime = DefaultRuntime::new(
            input,
            &self.inner.input_layout,
            output,
            &self.inner.output_layout,
            binding_resources,
            &self.private_memory_layout,
        );

        self.run_with_runtime(runtime)
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub enum EntryPointError<R> {
    #[error(transparent)]
    RuntimeError(#[from] R),

    #[error("Shader killed")]
    Killed,

    #[error("Pointer out of bounds")]
    PointerOutOfBounds,

    #[error("Division by zero")]
    DivisionByZero,

    #[error("Integer overflow")]
    Overflow,
}

#[cfg(test)]
mod tests {
    use crate::{
        CompiledModule,
        util::tests::assert_send_sync,
    };

    #[test]
    fn compiled_module_is_send_sync() {
        assert_send_sync::<CompiledModule>();
    }
}
