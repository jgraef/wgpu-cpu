use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::{
        CompileStatement,
        Statement,
    },
};

#[derive(Clone, Debug)]
pub struct BlockStatement {
    pub block: naga::ir::Block,
}

impl CompileStatement for naga::Block {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        // I don't think we actually have to emit a block in cranelift IR.
        // We only have to emit blocks, if we want to jump to them from multiple other
        // blocks, or as an entry point for functions.

        for (statement, span) in self.span_iter() {
            compiler.set_source_span(*span);
            let statement: Statement = statement.clone().into();
            statement.compile_statement(compiler)?;
        }
        Ok(())
    }
}

impl CompileStatement for BlockStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        self.block.compile_statement(compiler)
    }
}
