use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::{
        PointerValue,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct LoadExpression {
    pub pointer: naga::Handle<naga::Expression>,
}

impl CompileExpression for LoadExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let pointer: PointerValue = compiler.compile_expression(self.pointer)?.try_into()?;
        pointer.deref_load(compiler.context, &mut compiler.function_builder)
    }
}
