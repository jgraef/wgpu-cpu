use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct DerivativeExpression {
    pub axis: naga::DerivativeAxis,
    pub control: naga::DerivativeControl,
    pub expression: naga::Handle<naga::Expression>,
}

impl CompileExpression for DerivativeExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
