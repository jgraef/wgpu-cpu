use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct SubgroupBallotResultExpression;

impl CompileExpression for SubgroupBallotResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubgroupOperationResultExpression {
    pub ty: naga::Handle<naga::Type>,
}

impl CompileExpression for SubgroupOperationResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
