use std::sync::Arc;

use cranelift_jit::JITModule;
use cranelift_module::FuncId;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    compiler::runtime::{
        BindingStackLayout,
        DefaultRuntime,
        Runtime,
        RuntimeContext,
        RuntimeData,
    },
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        EntryPoints,
        InterStageLayout,
    },
};

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
        entry_points: EntryPoints<CompiledEntryPoint>,
    ) -> Self {
        Self {
            inner: Arc::new(CompiledModuleInner {
                jit_module: Some(jit_module),
                entry_points,
            }),
        }
    }

    pub fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        self.inner.entry_points.find(name, stage)
    }

    pub fn entry_point(&self, index: EntryPointIndex) -> EntryPoint<'_> {
        let inner = &self.inner.entry_points[index];

        let function_pointer = self
            .inner
            .jit_module
            .as_ref()
            .expect("JIT module gone")
            .get_finalized_function(inner.data.function_id);

        EntryPoint {
            inner,
            function_pointer,
        }
    }
}

#[derive(derive_more::Debug)]
struct CompiledModuleInner {
    // this is in an Option, so we can take it out on Drop. MaybeUninit would work too
    #[debug(skip)]
    jit_module: Option<JITModule>,

    entry_points: EntryPoints<CompiledEntryPoint>,
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
    pub function_id: FuncId,
    pub input_layout: Vec<BindingStackLayout>,
    pub output_layout: Vec<BindingStackLayout>,
}

#[derive(Clone, Copy)]
pub struct EntryPoint<'a> {
    inner: &'a crate::entry_point::EntryPoint<CompiledEntryPoint>,
    function_pointer: *const u8,
}

impl<'a> EntryPoint<'a> {
    pub fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
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

    pub fn function<R>(&self) -> impl Fn(R) -> Result<(), R::Error>
    where
        R: Runtime,
    {
        |runtime: R| {
            let mut runtime_context_data = RuntimeData::new(runtime);
            let mut runtime_context = RuntimeContext::new(&mut runtime_context_data);

            unsafe {
                // SAFETY: We just created the context struct and it's valid until the end of
                // the scope. Thus it's safe to call the function with this as
                // argument. The function itself is safe because `EntryPoint`s
                // can only be created from `CompiledModule`s, which can only be
                // created with an safety guarantuee that the compiled code is safe,
                // takes 1 pointer arguments and returns nothing.

                let function = std::mem::transmute::<_, unsafe extern "C" fn(*const RuntimeContext)>(
                    self.function_pointer,
                );

                // note: the order of these arguments must be synchronized with the compiled
                // code, which is generated in [`Compiler::compile_entry_point`].
                function(&mut runtime_context as *mut _);
            }

            let _runtime = runtime_context_data.into_inner()?;
            Ok(())
        }
    }

    pub fn run_with_runtime<R>(&self, runtime: R) -> Result<(), R::Error>
    where
        R: Runtime,
    {
        self.function()(runtime)
    }

    pub fn run<I, O>(&self, input: I, output: O)
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        let runtime = DefaultRuntime {
            input,
            input_layout: &self.inner.data.input_layout,
            output,
            output_layout: &self.inner.data.output_layout,
        };

        self.run_with_runtime(runtime).expect("runtime error");
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        compiler::CompiledModule,
        util::test::assert_send_sync,
    };

    #[test]
    fn compiled_module_is_send_sync() {
        assert_send_sync::<CompiledModule>();
    }
}
