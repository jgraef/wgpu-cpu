use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Debug)]
pub struct LoopStatement {
    pub body: naga::Block,
    pub continuing: naga::Block,
    pub break_if: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for LoopStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
