use cranelift_codegen::ir::{
    self,
    InstBuilder as _,
};

use crate::{
    Error,
    function::{
        ABORT_CODE_TYPE,
        AbortCode,
        FunctionCompiler,
    },
    statement::{
        CompileStatement,
        ControlFlow,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct KillStatement;

impl CompileStatement for KillStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        assert!(!compiler.loop_switch_stack.is_continuing());

        let abort_code = compiler
            .function_builder
            .ins()
            .iconst(ABORT_CODE_TYPE, AbortCode::Kill);
        compiler
            .function_builder
            .ins()
            .jump(compiler.abort_block, [&ir::BlockArg::Value(abort_code)]);

        Ok(ControlFlow::Diverged)
    }
}
