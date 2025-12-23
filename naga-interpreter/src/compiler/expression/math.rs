use arrayvec::ArrayVec;

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

#[derive(Clone, Debug)]
pub struct MathExpression {
    pub function: naga::MathFunction,
    pub arguments: ArrayVec<naga::Handle<naga::Expression>, 4>,
}

impl CompileExpression for MathExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for MathExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}
