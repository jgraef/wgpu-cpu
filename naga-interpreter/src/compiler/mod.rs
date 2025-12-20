pub mod bindings;
pub mod compiler;
pub mod context;
pub mod function;
pub mod module;
#[cfg(test)]
mod tests;
pub(self) mod util;
pub mod value;

use cranelift_codegen::settings::Configurable;
use cranelift_jit::{
    JITBuilder,
    JITModule,
};
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

    #[error("Host machine is not supported: {0}")]
    HostNotSupported(&'static str),
}

pub fn compile_jit(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<CompiledModule, Error> {
    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift_native::builder().map_err(Error::HostNotSupported)?;
    let isa = isa_builder
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .unwrap();

    let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut jit_module = JITModule::new(jit_builder);

    let mut compiler = Compiler::new(module, info, &mut jit_module)?;
    let entry_points = compiler.compile_all_entry_points()?;

    jit_module.finalize_definitions()?;

    Ok(CompiledModule::new(jit_module, entry_points))
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
        compile_jit(&ir, &info)
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
