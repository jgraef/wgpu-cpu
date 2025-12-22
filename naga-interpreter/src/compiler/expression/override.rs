use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct OverrideExpression {
    pub handle: naga::Handle<naga::Override>,
}

impl CompileExpression for OverrideExpression {


    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
