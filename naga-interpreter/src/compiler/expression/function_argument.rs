use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::{
        FromIrValues,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct FunctionArgumentExpression {
    pub index: u32,
}

impl CompileExpression for FunctionArgumentExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let index: usize = self
            .index
            .try_into()
            .expect("function argument index overflow");
        let argument = &compiler.declaration.arguments[index];
        let block_params = compiler.function_builder.block_params(compiler.entry_block);
        let block_params = block_params[argument.block_inputs.clone()].iter().copied();

        Ok(Value::from_ir_values_iter(
            &compiler.context,
            argument.ty,
            block_params,
        ))
    }
}
