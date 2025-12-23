use crate::compiler::{
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
pub struct AccessExpression {
    pub base: naga::Handle<naga::Expression>,
    pub index: naga::Handle<naga::Expression>,
}

impl CompileExpression for AccessExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for AccessExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AccessIndexExpression {
    pub base: naga::Handle<naga::Expression>,
    pub index: u32,
}

impl CompileExpression for AccessIndexExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for AccessIndexExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}
