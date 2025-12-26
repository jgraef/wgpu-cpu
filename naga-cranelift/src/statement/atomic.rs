use crate::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct AtomicStatement {
    pub pointer: naga::Handle<naga::Expression>,
    pub function: naga::AtomicFunction,
    pub value: naga::Handle<naga::Expression>,
    pub result: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for AtomicStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
