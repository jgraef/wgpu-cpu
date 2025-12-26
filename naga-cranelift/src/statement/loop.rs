use cranelift_codegen::ir::InstBuilder;

use crate::{
    Error,
    expression::{
        CompileExpression,
        Expression,
    },
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
pub struct LoopStatement {
    pub body: BlockStatement,
    pub continuing: BlockStatement,
    pub break_if: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for LoopStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        let body_block = compiler.function_builder.create_block();
        let continuing_block = compiler.function_builder.create_block();
        let exit_block = compiler.function_builder.create_block();

        compiler.function_builder.ins().jump(body_block, []);

        compiler.function_builder.switch_to_block(body_block);
        compiler
            .loop_switch_stack
            .push_loop(continuing_block, exit_block);
        let body_control_flow = self.body.compile_statement(compiler)?;
        compiler
            .loop_switch_stack
            .pop_loop(continuing_block, exit_block);

        if body_control_flow.is_continuing() {
            compiler.function_builder.ins().jump(continuing_block, []);

            compiler.function_builder.switch_to_block(continuing_block);
            compiler.loop_switch_stack.push_continuing();
            let continuing_control_flow = self.continuing.compile_statement(compiler)?;
            assert!(
                continuing_control_flow.is_continuing(),
                "diverging control flow is not allowed in a loop continuing block"
            );
            compiler.loop_switch_stack.pop_continuing();

            if let Some(break_if) = self.break_if {
                // don't use evaluate_expression on the handle, or it will reuse values from
                // previous iterations
                let expression: Expression = compiler.function.expressions[break_if].clone().into();
                let condition_value: ScalarValue = expression
                    .compile_expression(compiler)?
                    .try_into()
                    .expect("expected bool");
                let condition_value = condition_value.as_ir_value();

                compiler.function_builder.ins().brif(
                    condition_value,
                    body_block,
                    [],
                    exit_block,
                    [],
                );
            }
            else {
                compiler.function_builder.ins().jump(body_block, []);
            }
        }

        compiler.function_builder.seal_block(body_block);
        compiler.function_builder.seal_block(continuing_block);
        compiler.function_builder.seal_block(exit_block);
        compiler.function_builder.switch_to_block(exit_block);

        Ok(body_control_flow)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContinueStatement;

impl CompileStatement for ContinueStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        let continuing_block = compiler.loop_switch_stack.get_continuing_block();
        compiler.function_builder.ins().jump(continuing_block, []);
        Ok(ControlFlow::Diverged)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BreakStatement;

impl CompileStatement for BreakStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        let break_block = compiler.loop_switch_stack.get_break_block();
        compiler.function_builder.ins().jump(break_block, []);
        Ok(ControlFlow::Diverged)
    }
}
