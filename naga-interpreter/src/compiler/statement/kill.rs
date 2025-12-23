use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct KillStatement;

impl CompileStatement for KillStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
