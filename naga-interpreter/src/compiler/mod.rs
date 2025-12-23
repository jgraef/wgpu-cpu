pub mod compiler;
pub mod constant;
pub mod expression;
pub mod function;
pub mod product;
pub mod runtime;
pub mod simd;
pub mod statement;
#[cfg(test)]
mod tests;
pub mod types;
pub mod util;
pub mod value;
pub mod variable;

use std::sync::Arc;

use cranelift_codegen::{
    isa,
    settings::Configurable as _,
};
use cranelift_jit as jit;
pub use product::CompiledModule;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    compiler::{
        compiler::{
            Compiler,
            Config,
        },
        expression::ExpressionNotConstant,
        types::InvalidType,
        util::ClifOutput,
        value::UnexpectedType,
    },
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        InterStageLayout,
    },
};

// todo: check which of these should panic instead (because we assume the source
// module to be correct).
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unsupported type: {ty:?}")]
    UnsupportedType { ty: naga::TypeInner },

    #[error(transparent)]
    Output(#[from] cranelift_module::ModuleError),

    #[error(transparent)]
    Codegen(#[from] cranelift_codegen::CodegenError),

    #[error(transparent)]
    Layout(#[from] naga::proc::LayoutError),

    #[error("Host machine is not supported: {0}")]
    HostNotSupported(&'static str),

    #[error(transparent)]
    InvalidType(#[from] InvalidType),

    #[error(transparent)]
    UnexpectedType(#[from] UnexpectedType),

    #[error("Call to undeclared function: {name:?}")]
    UndeclaredFunctionCall {
        name: Option<String>,
        handle: naga::Handle<naga::Function>,
    },

    #[error(transparent)]
    Io(#[from] std::io::Error),

    // this should probaly panic
    #[error("transparent")]
    ExpressionNotConstant(#[from] ExpressionNotConstant),
}

/// JIT-compile a [`naga::Module`] for execution on the CPU.
pub fn compile_jit(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<CompiledModule, Error> {
    let isa = system_isa()?;
    let jit_builder = jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    let mut jit_module = jit::JITModule::new(jit_builder);

    // since we're compiling for JIT we can use any calling convention internally.
    let config = Config {
        // fixme: doesn't support exceptions
        //calling_convention: Some(isa::CallConv::Fast),
        ..Default::default()
    };

    let mut compiler = Compiler::new(module, info, &mut jit_module, config)?;
    compiler.declare_all_functions()?;
    compiler.compile_all_functions()?;
    let entry_points = compiler.compile_all_entry_points()?;

    jit_module.finalize_definitions()?;

    let compiled_module = unsafe {
        // SAFETY: The whole compilation process was done without any outside
        // intervention. Therefore the compiled entry point functions follow our rules:
        // They take 2 pointer arguments, return nothing, and are safe to run.
        CompiledModule::new(jit_module, entry_points)
    };

    Ok(compiled_module)
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

/// JIT-compile a [`naga::Module`] for execution on the CPU.
pub fn compile_clif<Writer>(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    config: Config,
    isa: Option<Arc<dyn isa::TargetIsa>>,
    output: &mut Writer,
) -> Result<(), Error>
where
    Writer: std::io::Write,
{
    let isa = if let Some(isa) = isa {
        isa
    }
    else {
        system_isa()?
    };

    let mut clif_output = ClifOutput::new(isa);
    let mut compiler = Compiler::new(module, info, &mut clif_output, config)?;
    compiler.declare_all_functions()?;
    compiler.compile_all_functions()?;
    let _entry_points = compiler.compile_all_entry_points()?;
    clif_output.finalize();

    for (_func_id, function) in &clif_output.functions {
        writeln!(output, "{function:#?}")?;
    }

    Ok(())
}

pub fn compile_clif_to_string(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
    config: Config,
    isa: Option<Arc<dyn isa::TargetIsa>>,
) -> Result<String, Error> {
    let mut buf = vec![];
    compile_clif(module, info, config, isa, &mut buf)?;
    Ok(String::from_utf8(buf).unwrap())
}

pub fn system_isa() -> Result<Arc<dyn isa::TargetIsa>, Error> {
    let mut flag_builder = cranelift_codegen::settings::builder();
    flag_builder.set("use_colocated_libcalls", "false").unwrap();
    flag_builder.set("is_pic", "false").unwrap();

    let isa_builder = cranelift_native::builder().map_err(Error::HostNotSupported)?;
    let isa = isa_builder.finish(cranelift_codegen::settings::Flags::new(flag_builder))?;

    Ok(isa)
}
