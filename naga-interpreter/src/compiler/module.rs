use std::sync::Arc;

use cranelift_jit::JITModule;
use cranelift_module::FuncId;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    compiler::bindings::{
        BindingStackLayout,
        ShimData,
        ShimVtable,
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
    pub(super) fn new(
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

    pub fn function<I, O>(&self) -> impl Fn(I, O)
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        let function = unsafe {
            // SAFETY: This function signature matches what our compiled code expects. This
            // still returns an unsafe function, because it's the responsibility of the
            // caller to ensure the vtable and data matches
            std::mem::transmute::<_, unsafe fn(&ShimVtable, &mut ShimData<I, O>)>(
                self.function_pointer,
            )
        };

        let shim_vtable = ShimVtable::new::<I, O>();

        move |input: I, output: O| {
            let mut shim_data = ShimData {
                input,
                input_layout: &self.inner.data.input_layout,
                output,
                output_layout: &self.inner.data.output_layout,
                panic: Ok(()),
            };

            unsafe {
                // SAFETY: We just created the vtable and data with matching types. Thus it's
                // safe to call the function with these arguments
                function(&shim_vtable, &mut shim_data)
            }

            if let Err(panic) = shim_data.panic {
                std::panic::resume_unwind(panic);
            }
        }
    }

    pub fn run<I, O>(&self, input: I, output: O)
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        self.function()(input, output);
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
