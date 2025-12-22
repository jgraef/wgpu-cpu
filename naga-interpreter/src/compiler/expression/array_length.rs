use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct ArrayLengthExpression {
    pub array: naga::Handle<naga::Expression>,
}

impl CompileExpression for ArrayLengthExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
