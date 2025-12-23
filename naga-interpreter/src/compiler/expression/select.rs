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
pub struct SelectExpression {
    pub condition: naga::Handle<naga::Expression>,
    pub accept: naga::Handle<naga::Expression>,
    pub reject: naga::Handle<naga::Expression>,
}

impl CompileExpression for SelectExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for SelectExpression {
    type Output = ConstantValue;

    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}
