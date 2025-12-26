use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct CooperativeLoadExpression {
    pub columns: naga::CooperativeSize,
    pub rows: naga::CooperativeSize,
    pub role: naga::CooperativeRole,
    pub data: naga::CooperativeData,
}

impl CompileExpression for CooperativeLoadExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!("compile cooperative load expression");
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CooperativeMultiplyAddExpression {
    pub a: naga::Handle<naga::Expression>,
    pub b: naga::Handle<naga::Expression>,
    pub c: naga::Handle<naga::Expression>,
}

impl CompileExpression for CooperativeMultiplyAddExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!("compile cooperative multiply add expression");
    }
}
