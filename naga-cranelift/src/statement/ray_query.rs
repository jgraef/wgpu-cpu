use crate::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Debug)]
pub struct RayQueryStatement {
    pub query: naga::Handle<naga::Expression>,
    pub function: naga::RayQueryFunction,
}

impl CompileStatement for RayQueryStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
