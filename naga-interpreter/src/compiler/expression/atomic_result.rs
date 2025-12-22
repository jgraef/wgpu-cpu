use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct AtomicResultExpression {
    pub ty: naga::Handle<naga::Type>,
    pub comparison: bool,
}

impl CompileExpression for AtomicResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
