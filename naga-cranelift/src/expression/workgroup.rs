use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct WorkGroupUniformLoadResultExpression {
    pub ty: naga::Handle<naga::Type>,
}

impl CompileExpression for WorkGroupUniformLoadResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
