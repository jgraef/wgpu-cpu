use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct RayQueryVertexPositionsExpression {
    pub query: naga::Handle<naga::Expression>,
    pub committed: bool,
}

impl CompileExpression for RayQueryVertexPositionsExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RayQueryProceedResultExpression;

impl CompileExpression for RayQueryProceedResultExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RayQueryGetIntersectionExpression {
    pub query: naga::Handle<naga::Expression>,
    pub committed: bool,
}

impl CompileExpression for RayQueryGetIntersectionExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
