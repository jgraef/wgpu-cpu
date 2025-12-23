use cranelift_codegen::ir::InstBuilder;

use crate::compiler::{
    Error,
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

        // fixme: return ControlFlow to stop compiling this block. this is a bit tricky
        // because we also return Results for now we'll just switch to a new
        // block for the rest. this block will not be jumped to, but we still do the
        // work compiling it.
        let void_block = compiler.function_builder.create_block();
        compiler.function_builder.seal_block(void_block);
        compiler.function_builder.switch_to_block(void_block);
        compiler.function_builder.set_cold_block(void_block); // very cold lol

        Ok(())
    }
}
