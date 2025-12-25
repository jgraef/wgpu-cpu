use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::CompileStatement,
    value::{
        AsIrValues,
        Load,
        StackLocation,
        Value,
    },
};

#[derive(Clone, Debug)]
pub struct CallStatement {
    pub function: naga::Handle<naga::Function>,
    pub arguments: Vec<naga::Handle<naga::Expression>>,
    pub result: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for CallStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        // todo: error would be nicer
        let imported_function = compiler
            .imported_functions
            .get(self.function)
            .unwrap_or_else(|| panic!("Function not imported: {:?}", self.function));
        let function_ref = imported_function.function_ref;

        let mut argument_values = Vec::with_capacity(self.arguments.len() + 2);

        // first argument: runtime context
        argument_values.push(compiler.runtime_context.pointer);

        // optional second argument: result pointer
        let result = imported_function
            .declaration
            .return_type
            .as_ref()
            .map(|result| {
                let result_stack_slot = compiler
                    .function_builder
                    .create_sized_stack_slot(result.stack_slot_data.clone());
                let result_pointer = compiler.function_builder.ins().stack_addr(
                    compiler.context.pointer_type(),
                    result_stack_slot,
                    0,
                );
                argument_values.push(result_pointer);

                (result_stack_slot, result.ty, self.result.expect("function returns a type, but expression doesn't provide a return expression"))
            });

        // remaining arguments: actual function arguments
        for argument in &self.arguments {
            argument_values.extend(argument.compile_expression(compiler)?.as_ir_values());
        }

        // call function
        let inst = compiler
            .function_builder
            .ins()
            .call(function_ref, &argument_values);

        // check returned abort code
        let abort_code = compiler.function_builder.inst_results(inst)[0];
        let continue_block = compiler.function_builder.create_block();
        compiler.function_builder.ins().brif(
            abort_code,
            compiler.abort_block,
            [&ir::BlockArg::Value(abort_code)],
            continue_block,
            [],
        );
        compiler.function_builder.seal_block(continue_block);
        compiler.function_builder.switch_to_block(continue_block);

        // load result value
        if let Some((stack_slot, result_type, result_expression)) = result {
            let result_value = Value::load(
                &compiler.context,
                &mut compiler.function_builder,
                result_type,
                StackLocation::from(stack_slot),
            )?;

            compiler
                .emitted_expression
                .insert(result_expression, result_value);
        }

        Ok(())
    }
}
