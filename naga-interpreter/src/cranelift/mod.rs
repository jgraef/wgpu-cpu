use crate::cranelift::{
    compiler::{
        Compiler,
        Error,
    },
    module::CompiledModule,
};

pub mod bindings;
pub mod compiler;
pub mod module;
#[cfg(test)]
mod tests;

pub fn compile(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<CompiledModule, Error> {
    let compiler = Compiler::new(module, info)?;
    let module = compiler.compile()?;
    Ok(module)
}
