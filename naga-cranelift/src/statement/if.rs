use cranelift_codegen::ir::InstBuilder;

use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::{
        BlockStatement,
        CompileStatement,
        ControlFlow,
    },
    value::{
        AsIrValue,
        ScalarValue,
    },
};

#[derive(Clone, Debug)]
pub struct IfStatement {
    pub condition: naga::Handle<naga::Expression>,
    pub accept: BlockStatement,
    pub reject: BlockStatement,
}

impl CompileStatement for IfStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
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

        let mut control_flow = ControlFlow::Diverged;

        {
            compiler.function_builder.switch_to_block(accept_block);
            if self.accept.compile_statement(compiler)?.is_continuing() {
                compiler.function_builder.ins().jump(exit_block, []);
                control_flow = ControlFlow::Continue;
            }
        }

        {
            compiler.function_builder.switch_to_block(reject_block);
            if self.reject.compile_statement(compiler)?.is_continuing() {
                compiler.function_builder.ins().jump(exit_block, []);
                control_flow = ControlFlow::Continue;
            }
        }

        compiler.function_builder.seal_block(exit_block);
        compiler.function_builder.switch_to_block(exit_block);

        Ok(control_flow)
    }
}
