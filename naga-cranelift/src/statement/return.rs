use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::{
    Error,
    expression::CompileExpression,
    function::{
        ABORT_CODE_TYPE,
        FunctionCompiler,
    },
    statement::{
        CompileStatement,
        ControlFlow,
    },
    value::{
        Pointer,
        Store,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct ReturnStatement {
    pub value: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for ReturnStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        // note: should we instead generate an exit block that returns the result in
        // FunctionCompiler::new (like we do for the abort block)? then we'd just need
        // to jump to it from here.

        if let Some(expression) = self.value {
            let result_pointer = compiler.function_builder.block_params(compiler.entry_block)[1];
            let value = expression.compile_expression(compiler)?;
            value.store(
                compiler.context,
                &mut compiler.function_builder,
                Pointer {
                    value: result_pointer,
                    memory_flags: ir::MemFlags::trusted(),
                    offset: 0,
                },
            )?;
        }

        let abort_code = compiler.function_builder.ins().iconst(ABORT_CODE_TYPE, 0);
        compiler.function_builder.ins().return_(&[abort_code]);

        Ok(ControlFlow::Diverged)
    }
}
