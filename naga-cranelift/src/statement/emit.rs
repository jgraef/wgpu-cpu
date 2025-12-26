use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Debug)]
pub struct EmitStatement {
    pub expressions: naga::Range<naga::Expression>,
}

impl CompileStatement for EmitStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        for expression in self.expressions.clone() {
            // all compiled expressions are automatically stored as emitted.
            expression.compile_expression(compiler)?;
        }
        Ok(())
    }
}
