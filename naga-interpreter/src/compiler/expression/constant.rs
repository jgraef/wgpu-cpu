use crate::compiler::{
    Error,
    compiler::Context,
    constant::{
        CompileConstant,
        ConstantValue,
    },
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct ConstantUseExpression {
    pub handle: naga::Handle<naga::Constant>,
}

impl CompileExpression for ConstantUseExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let value = self.evaluate_expression(&compiler.context)?;
        value.compile_constant(compiler)
    }
}

impl EvaluateExpression for ConstantUseExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        let constant = &context.source.constants[self.handle];
        constant.init.evaluate_expression(context)
    }
}
