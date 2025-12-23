use cranelift_codegen::ir::InstBuilder;

use crate::compiler::{
    Error,
    compiler::FuncBuilderExt,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
    value::AsIrValues,
};

#[derive(Clone, Copy, Debug)]
pub struct ReturnStatement {
    pub value: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for ReturnStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let mut return_values = vec![];

        if let Some(expression) = self.value {
            let value = expression.compile_expression(compiler)?;
            return_values.extend(value.as_ir_values());
        }

        compiler.function_builder.ins().return_(&return_values);
        compiler.function_builder.switch_to_void_block();

        Ok(())
    }
}
