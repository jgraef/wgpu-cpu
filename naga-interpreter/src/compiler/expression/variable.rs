use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::{
        PointerValue,
        Value,
    },
    variable::GlobalVariable,
};
#[derive(Clone, Copy, Debug)]
pub struct GlobalVariableExpression {
    pub handle: naga::Handle<naga::GlobalVariable>,
}

impl CompileExpression for GlobalVariableExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let global_variable = compiler.context.global_variables[self.handle];

        match global_variable {
            GlobalVariable::Memory {
                address_space,
                offset,
            } => {
                todo!("get pointer to global memory");
            }
            GlobalVariable::Resource {
                address_space,
                binding,
            } => {
                todo!("get pointer to resource binding");
            }
        }

        //todo!();
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LocalVariableExpression {
    pub handle: naga::Handle<naga::LocalVariable>,
}

impl CompileExpression for LocalVariableExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let local_variable = compiler.local_variables[self.handle];

        let value =
            PointerValue::from_stack_slot(local_variable.pointer_type, local_variable.stack_slot);

        Ok(value.into())
    }
}
