pub mod bindings;
pub mod compiler;
pub mod module;
#[cfg(test)]
mod tests;

use cranelift_module::ModuleError;
pub use module::CompiledModule;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    compiler::compiler::Compiler,
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        InterStageLayout,
    },
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unsupported type: {ty:?}")]
    UnsupportedType { ty: naga::TypeInner },

    #[error(transparent)]
    Cranelift(#[from] ModuleError),

    #[error(transparent)]
    Layout(#[from] naga::proc::LayoutError),
}

pub fn compile(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<CompiledModule, Error> {
    let compiler = Compiler::new(module, info)?;
    let module = compiler.compile()?;
    Ok(module)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CompilerBackend;

impl crate::backend::Backend for CompilerBackend {
    type Module = CompiledModule;
    type Error = Error;

    fn create_module(
        &self,
        ir: naga::Module,
        info: naga::valid::ModuleInfo,
    ) -> Result<Self::Module, Self::Error> {
        compile(&ir, &info)
    }
}

impl crate::backend::Module for CompiledModule {
    fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        self.find_entry_point(name, stage)
    }

    fn run_entry_point<I, O>(&self, index: EntryPointIndex, input: I, output: O)
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        self.entry_point(index).run(input, output);
    }

    fn inter_stage_layout(&self, entry_point: EntryPointIndex) -> Option<&InterStageLayout> {
        self.entry_point(entry_point).inter_stage_layout()
    }

    fn early_depth_test(&self, entry_point: EntryPointIndex) -> Option<naga::EarlyDepthTest> {
        self.entry_point(entry_point).early_depth_test()
    }
}
