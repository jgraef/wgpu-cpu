use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Debug)]
pub struct SwitchStatement {
    pub selector: naga::Handle<naga::Expression>,
    pub cases: Vec<naga::SwitchCase>,
}

impl CompileStatement for SwitchStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
