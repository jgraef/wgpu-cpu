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
pub struct SwizzleExpression {
    pub size: naga::VectorSize,
    pub vector: naga::Handle<naga::Expression>,
    pub pattern: [naga::SwizzleComponent; 4],
}

impl CompileExpression for SwizzleExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

impl EvaluateExpression for SwizzleExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}
