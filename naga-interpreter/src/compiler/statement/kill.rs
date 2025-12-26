use cranelift_codegen::ir::{
    self,
    InstBuilder as _,
};

use crate::compiler::{
    Error,
    compiler::FuncBuilderExt,
    function::{
        ABORT_CODE_TYPE,
        AbortCode,
        FunctionCompiler,
    },
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct KillStatement;

impl CompileStatement for KillStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        assert!(!compiler.loop_switch_stack.is_continuing());

        let abort_code = compiler
            .function_builder
            .ins()
            .iconst(ABORT_CODE_TYPE, AbortCode::Kill);
        compiler
            .function_builder
            .ins()
            .jump(compiler.abort_block, [&ir::BlockArg::Value(abort_code)]);

        compiler.function_builder.switch_to_void_block();

        Ok(())
    }
}
