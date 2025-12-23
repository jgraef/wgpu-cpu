use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
    value::{
        PointerValue,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct StoreStatement {
    pub pointer: naga::Handle<naga::Expression>,
    pub value: naga::Handle<naga::Expression>,
}

impl CompileStatement for StoreStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let pointer: PointerValue = self.pointer.compile_expression(compiler)?.try_into()?;
        let value: Value = self.value.compile_expression(compiler)?;
        pointer.deref_store(compiler.context, &mut compiler.function_builder, &value)
    }
}
