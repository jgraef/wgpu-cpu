use cranelift_codegen::ir::InstBuilder;

use crate::compiler::{
    Error,
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
            compiler.compile_expression(self.condition)?.try_into()?;
        let condition_value = condition_value.as_ir_value();

        let accept_block = compiler.function_builder.create_block();
        let reject_block = compiler.function_builder.create_block();
        let continue_block = compiler.function_builder.create_block();

        compiler
            .function_builder
            .ins()
            .brif(condition_value, accept_block, [], reject_block, []);

        compiler.function_builder.seal_block(accept_block);
        compiler.function_builder.seal_block(reject_block);

        compiler.function_builder.switch_to_block(accept_block);
        self.accept.compile_statement(compiler)?;
        compiler.function_builder.ins().jump(continue_block, []);

        compiler.function_builder.switch_to_block(reject_block);
        self.reject.compile_statement(compiler)?;
        compiler.function_builder.ins().jump(continue_block, []);

        compiler.function_builder.seal_block(continue_block);
        compiler.function_builder.switch_to_block(continue_block);

        Ok(())
    }
}
