use crate::compiler::{
    Error,
    compiler::FuncBuilderExt,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Debug)]
pub struct CallStatement {
    pub function: naga::Handle<naga::Function>,
    pub arguments: Vec<naga::Handle<naga::Expression>>,
    pub result: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for CallStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let argument_values = self
            .arguments
            .iter()
            .map(|argument| argument.compile_expression(compiler))
            .collect::<Result<Vec<_>, Error>>()?;

        /*
        compiler.compile_call()
        let type_layout = context.layouter[variable.ty];
                        let stack_slot_key = ir::StackSlotKey::new(handle.index().try_into().unwrap());

                        let stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
                            kind: ir::StackSlotKind::ExplicitSlot,
                            size: type_layout.size,
                            align_shift: alignment_log2(type_layout.alignment),
                            key: Some(stack_slot_key),
                        });*/

        // todo: error would be nicer
        let imported_function = compiler
            .imported_functions
            .get(self.function)
            .unwrap_or_else(|| panic!("Function not imported: {:?}", self.function));

        let result_value = compiler.function_builder.call_shader_function(
            compiler.context,
            imported_function.function_ref,
            &compiler.runtime_context,
            argument_values,
            imported_function.declaration.return_type,
        );

        if let Some(result_handle) = &self.result {
            if let Some(result_value) = result_value {
                compiler
                    .emitted_expression
                    .insert(*result_handle, result_value);
            }
            else {
                panic!("Expected function to return a value");
            }
        }

        Ok(())
    }
}
