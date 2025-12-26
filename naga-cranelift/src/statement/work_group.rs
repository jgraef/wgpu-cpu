use crate::{
    Error,
    function::FunctionCompiler,
    statement::{
        CompileStatement,
        ControlFlow,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct WorkGroupUniformLoadStatement {}

impl CompileStatement for WorkGroupUniformLoadStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        todo!()
    }
}
