use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct CallResultExpression {
    pub function: naga::Handle<naga::Function>,
}

impl CompileExpression for CallResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        panic!("function result was not emitted");
    }
}
