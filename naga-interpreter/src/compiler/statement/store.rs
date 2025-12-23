use crate::compiler::{
    Error,
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
        let pointer: PointerValue = compiler.compile_expression(self.pointer)?.try_into()?;
        let value: Value = compiler.compile_expression(self.value)?;
        pointer.deref_store(compiler.context, &mut compiler.function_builder, &value)
    }
}
