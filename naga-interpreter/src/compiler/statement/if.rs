use cranelift_codegen::ir::InstBuilder;

use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
    value::{
        AsIrValue,
        ScalarValue,
    },
};

#[derive(Clone, Debug)]
pub struct IfStatement {
    pub condition: naga::Handle<naga::Expression>,
    pub accept: naga::Block,
    pub reject: naga::Block,
}

impl CompileStatement for IfStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let condition_value: ScalarValue =
            self.condition.compile_expression(compiler)?.try_into()?;
        let condition_value = condition_value.as_ir_value();

        let accept_block = compiler.function_builder.create_block();
        let reject_block = compiler.function_builder.create_block();
        let exit_block = compiler.function_builder.create_block();

        compiler
            .function_builder
            .ins()
            .brif(condition_value, accept_block, [], reject_block, []);

        compiler.function_builder.seal_block(accept_block);
        compiler.function_builder.seal_block(reject_block);

        compiler.function_builder.switch_to_block(accept_block);
        self.accept.compile_statement(compiler)?;
        compiler.function_builder.ins().jump(exit_block, []);

        compiler.function_builder.switch_to_block(reject_block);
        self.reject.compile_statement(compiler)?;
        compiler.function_builder.ins().jump(exit_block, []);

        compiler.function_builder.seal_block(exit_block);
        compiler.function_builder.switch_to_block(exit_block);

        Ok(())
    }
}
