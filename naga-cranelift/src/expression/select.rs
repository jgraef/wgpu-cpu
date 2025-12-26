use cranelift_codegen::ir::{
    BlockArg,
    InstBuilder,
};

use crate::{
    Error,
    compiler::Context,
    constant::{
        ConstantScalar,
        ConstantValue,
    },
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    types::AsIrTypes,
    value::{
        AsIrValue,
        AsIrValues,
        FromIrValues,
        ScalarValue,
        TypeOf,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct SelectExpression {
    pub condition: naga::Handle<naga::Expression>,
    pub accept: naga::Handle<naga::Expression>,
    pub reject: naga::Handle<naga::Expression>,
}

impl CompileExpression for SelectExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let condition_value: ScalarValue =
            self.condition.compile_expression(compiler)?.try_into()?;
        let condition_value = condition_value.as_ir_value();

        // select would work, but for larger values a branch might be better. i think
        // cranelift will optimize anyway
        let accept_block = compiler.function_builder.create_block();
        let reject_block = compiler.function_builder.create_block();
        let exit_block = compiler.function_builder.create_block();

        compiler
            .function_builder
            .ins()
            .brif(condition_value, accept_block, [], reject_block, []);

        compiler.function_builder.seal_block(accept_block);
        compiler.function_builder.seal_block(reject_block);

        let accept_type;
        {
            compiler.function_builder.switch_to_block(accept_block);
            let accept_value = self.accept.compile_expression(compiler)?;
            accept_type = accept_value.type_of();
            for ty in accept_type.as_ir_types(compiler.context) {
                compiler.function_builder.append_block_param(exit_block, ty);
            }
            let block_args = accept_value
                .as_ir_values()
                .map(BlockArg::Value)
                .collect::<Vec<_>>();
            compiler
                .function_builder
                .ins()
                .jump(exit_block, &block_args);
        }

        {
            compiler.function_builder.switch_to_block(reject_block);
            let reject_value = self.reject.compile_expression(compiler)?;
            assert_eq!(accept_type, reject_value.type_of());
            let block_args = reject_value
                .as_ir_values()
                .map(BlockArg::Value)
                .collect::<Vec<_>>();
            compiler
                .function_builder
                .ins()
                .jump(exit_block, &block_args);
        }

        compiler.function_builder.seal_block(exit_block);
        compiler.function_builder.switch_to_block(exit_block);

        let value = Value::from_ir_values_iter(
            compiler.context,
            accept_type,
            compiler
                .function_builder
                .block_params(exit_block)
                .iter()
                .copied(),
        );

        Ok(value)
    }
}

impl EvaluateExpression for SelectExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        let condition_value: ConstantScalar =
            self.condition.evaluate_expression(context)?.try_into()?;
        if condition_value.as_bool() {
            self.accept.evaluate_expression(context)
        }
        else {
            self.reject.evaluate_expression(context)
        }
    }
}
