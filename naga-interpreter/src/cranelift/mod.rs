mod bindings;
mod compiler;
mod module;
#[cfg(test)]
mod tests;

pub use compiler::{
    Compiler,
    Error,
};
pub use module::CompiledModule;

pub fn compile(
    module: &naga::Module,
    info: &naga::valid::ModuleInfo,
) -> Result<CompiledModule, Error> {
    let compiler = Compiler::new(module, info)?;
    let module = compiler.compile()?;
    Ok(module)
}
