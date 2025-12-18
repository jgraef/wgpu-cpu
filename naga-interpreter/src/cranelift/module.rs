use std::sync::Arc;

use cranelift_jit::JITModule;
use cranelift_module::FuncId;

use crate::entry_point::{
    EntryPointIndex,
    EntryPointNotFound,
    EntryPoints,
};

#[derive(Clone, Debug)]
pub struct CompiledModule {
    pub(super) inner: Arc<CompiledModuleInner>,
}

impl CompiledModule {
    pub fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        self.inner.entry_points.find(name, stage)
    }
}

#[derive(derive_more::Debug)]
pub(super) struct CompiledModuleInner {
    #[debug(skip)]
    pub(super) jit_module: JITModule,
    pub(super) entry_points: EntryPoints<EntryPoint>,
}

#[derive(Clone, Copy, Debug)]
pub struct EntryPoint {
    pub(super) function_id: FuncId,
    // todo: io bindings
}
