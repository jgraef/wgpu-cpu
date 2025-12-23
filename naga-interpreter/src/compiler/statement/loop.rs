use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::compiler::{
    Error,
    compiler::FuncBuilderExt,
    expression::{
        CompileExpression,
        Expression,
    },
    function::FunctionCompiler,
    statement::CompileStatement,
    value::{
        AsIrValue,
        ScalarValue,
    },
};

#[derive(Clone, Debug)]
pub struct LoopStatement {
    pub body: naga::Block,
    pub continuing: naga::Block,
    pub break_if: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for LoopStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let body_block = compiler.function_builder.create_block();
        let continuing_block = compiler.function_builder.create_block();
        let exit_block = compiler.function_builder.create_block();

        compiler.function_builder.ins().jump(body_block, []);

        compiler.function_builder.switch_to_block(body_block);
        compiler.loop_stack.push_loop(continuing_block, exit_block);
        self.body.compile_statement(compiler)?;
        compiler.loop_stack.pop_loop(continuing_block, exit_block);
        compiler.function_builder.ins().jump(continuing_block, []);

        compiler.function_builder.switch_to_block(continuing_block);
        compiler.loop_stack.push_continuing();
        self.continuing.compile_statement(compiler)?;
        compiler.loop_stack.pop_continuing();

        if let Some(break_if) = self.break_if {
            // don't use evaluate_expression on the handle, or it will reuse values from
            // previous iterations
            let expression: Expression = compiler.function.expressions[break_if].clone().into();
            let condition_value: ScalarValue = expression
                .compile_expression(compiler)?
                .try_into()
                .expect("expected bool");
            let condition_value = condition_value.as_ir_value();

            compiler
                .function_builder
                .ins()
                .brif(condition_value, body_block, [], exit_block, []);
        }
        else {
            compiler.function_builder.ins().jump(body_block, []);
        }

        compiler.function_builder.seal_block(body_block);
        compiler.function_builder.seal_block(continuing_block);
        compiler.function_builder.seal_block(exit_block);
        compiler.function_builder.switch_to_block(exit_block);

        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ContinueStatement;

impl CompileStatement for ContinueStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let continuing_block = compiler.loop_stack.get_continuing_block();
        compiler.function_builder.ins().jump(continuing_block, []);
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BreakStatement;

impl CompileStatement for BreakStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let break_block = compiler.loop_stack.get_break_block();
        compiler.function_builder.ins().jump(break_block, []);
        compiler.function_builder.switch_to_void_block();
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct LoopStack {
    pub stack: Vec<LoopState>,
}

impl LoopStack {
    pub fn push_loop(&mut self, continuing_block: ir::Block, exit_block: ir::Block) {
        self.stack.push(LoopState::Body {
            continuing_block,
            exit_block,
        });
    }

    pub fn pop_loop(
        &mut self,
        expected_continuing_block: ir::Block,
        expected_exit_block: ir::Block,
    ) {
        match self.pop() {
            LoopState::Body {
                continuing_block,
                exit_block,
            } => {
                assert_eq!(continuing_block, expected_continuing_block);
                assert_eq!(exit_block, expected_exit_block);
            }
            LoopState::Continuing => panic!("in continuing block"),
        }
    }

    pub fn push_continuing(&mut self) {
        self.stack.push(LoopState::Continuing);
    }

    pub fn pop_continuing(&mut self) {
        match self.pop() {
            LoopState::Continuing => {}
            _ => todo!("not in continuing block"),
        }
    }

    pub fn pop(&mut self) -> LoopState {
        self.stack.pop().expect("not in loop")
    }

    pub fn top(&self) -> LoopState {
        *self.stack.last().expect("not in loop")
    }

    pub fn get_break_block(&self) -> ir::Block {
        match self.top() {
            LoopState::Body {
                continuing_block: _,
                exit_block,
            } => exit_block,
            LoopState::Continuing => panic!("in continuing block"),
        }
    }

    pub fn get_continuing_block(&self) -> ir::Block {
        match self.top() {
            LoopState::Body {
                continuing_block,
                exit_block: _,
            } => continuing_block,
            LoopState::Continuing => panic!("in continuing block"),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LoopState {
    Body {
        continuing_block: ir::Block,
        exit_block: ir::Block,
    },
    Continuing,
}
