use crate::{
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

#[derive(Clone, Copy, Debug)]
pub struct CooperativeStoreStatement {
    pub target: naga::Handle<naga::Expression>,
    pub data: naga::CooperativeData,
}

impl CompileStatement for CooperativeStoreStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!("compile cooperative store statement");
    }
}
