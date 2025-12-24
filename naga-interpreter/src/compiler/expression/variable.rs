use cranelift_codegen::ir::MemFlags;

use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::{
        Pointer,
        PointerValue,
        PointerValueInner,
        Value,
    },
    variable::GlobalVariableInner,
};
#[derive(Clone, Copy, Debug)]
pub struct GlobalVariableExpression {
    pub handle: naga::Handle<naga::GlobalVariable>,
}

impl CompileExpression for GlobalVariableExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let global_variable = compiler.global_variables[self.handle];

        let value = match global_variable.inner {
            GlobalVariableInner::Memory { offset, len } => {
                let base_pointer = compiler
                    .runtime_context
                    .private_memory(compiler.context, &mut compiler.function_builder);

                PointerValue {
                    ty: global_variable.pointer_type,
                    inner: PointerValueInner::Pointer(Pointer {
                        value: base_pointer,
                        memory_flags: MemFlags::new(),
                        offset: offset.try_into().expect("pointer offset overflow"),
                    }),
                }
            }
            GlobalVariableInner::Resource { binding } => {
                todo!("get pointer to resource binding");
            }
        };

        Ok(value.into())
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
