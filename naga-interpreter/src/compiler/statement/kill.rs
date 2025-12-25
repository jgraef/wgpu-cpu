use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct KillStatement;

impl CompileStatement for KillStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        assert!(!compiler.loop_switch_stack.is_continuing());

        // emits a call into the runtime kill method which sets the abort payload and
        // then returns through the normal abort mechanism.
        compiler.runtime_context.kill(
            &compiler.context,
            &mut compiler.function_builder,
            compiler.abort_block,
        );

        // the abort mechanism is conditional though and cranelift codegen doesn't know
        // that the kill function we called always returns a value that will cause a
        // jump to the abort block afterwards. but we need to terminate this block, so
        // we'll trap. this trap should never be hit because the normal abort
        // mechanism always takes over.
        const KILL_TRAP_CODE: ir::TrapCode = const { ir::TrapCode::user(2).unwrap() };
        compiler.function_builder.ins().trap(KILL_TRAP_CODE);

        Ok(())
    }
}
