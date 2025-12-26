use crate::{
    Error,
    function::FunctionCompiler,
    statement::{
        CompileStatement,
        ControlFlow,
        Statement,
    },
};

#[derive(Clone, Debug)]
pub struct BlockStatement {
    pub statements: Vec<(Statement, naga::Span)>,
}

impl From<&naga::Block> for BlockStatement {
    fn from(block: &naga::Block) -> Self {
        let statements = block
            .span_iter()
            .map(|(statement, span)| {
                let statement: Statement = statement.into();
                (statement, *span)
            })
            .collect();
        Self { statements }
    }
}

impl CompileStatement for BlockStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        for (statement, span) in &self.statements {
            compiler.set_source_span(*span);

            if statement.compile_statement(compiler)?.is_diverged() {
                return Ok(ControlFlow::Diverged);
            }
        }

        Ok(ControlFlow::Continue)
    }
}
