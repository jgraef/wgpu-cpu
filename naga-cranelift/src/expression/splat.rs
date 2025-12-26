use crate::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct SplatExpression {
    pub size: naga::VectorSize,
    pub value: naga::Handle<naga::Expression>,
}

impl CompileExpression for SplatExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for SplatExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}
