use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::compiler::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    statement::{
        BlockStatement,
        CompileStatement,
    },
    value::{
        AsIrValue,
        ScalarValue,
    },
};

#[derive(Clone, Debug)]
pub struct SwitchStatement {
    pub selector: naga::Handle<naga::Expression>,
    pub cases: Vec<SwitchCase>,
}

impl CompileStatement for SwitchStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        let selector_value: ScalarValue = self.selector.compile_expression(compiler)?.try_into()?;
        let selector_value = selector_value.as_ir_value();

        let mut switch_block = compiler.function_builder.current_block().unwrap();
        let exit_block = compiler.function_builder.create_block();
        let mut fall_through_block = None;
        let mut default_block = None;

        compiler.loop_switch_stack.push_switch(exit_block);

        for case in &self.cases {
            let case_block = fall_through_block
                .take()
                .unwrap_or_else(|| compiler.function_builder.create_block());

            let mut compile_non_default = |switch_value: i64| {
                let control = compiler.function_builder.ins().icmp_imm(
                    ir::condcodes::IntCC::Equal,
                    selector_value,
                    i64::from(switch_value),
                );

                switch_block = compiler.function_builder.create_block();

                compiler
                    .function_builder
                    .ins()
                    .brif(control, case_block, [], switch_block, []);

                compiler.function_builder.seal_block(case_block);
                compiler.function_builder.seal_block(switch_block);
            };

            match case.value {
                naga::SwitchValue::I32(value) => {
                    compile_non_default(value.into());
                }
                naga::SwitchValue::U32(value) => {
                    compile_non_default(value.into());
                }
                naga::SwitchValue::Default => {
                    assert!(default_block.is_none());
                    default_block = Some(case_block);
                }
            }

            compiler.function_builder.switch_to_block(case_block);
            case.body.compile_statement(compiler)?;

            let successor_block = if case.fall_through {
                let next = compiler.function_builder.create_block();
                fall_through_block = Some(next);
                next
            }
            else {
                exit_block
            };
            compiler.function_builder.ins().jump(successor_block, []);

            compiler.function_builder.switch_to_block(switch_block);
        }

        compiler.loop_switch_stack.pop_switch(exit_block);

        let default_block = default_block.expect("no default switch case");
        compiler.function_builder.ins().jump(default_block, []);
        compiler.function_builder.switch_to_block(exit_block);

        compiler.function_builder.seal_block(default_block);
        compiler.function_builder.seal_block(exit_block);

        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SwitchCase {
    /// Value, upon which the case is considered true.
    pub value: naga::SwitchValue,

    /// Body of the case.
    pub body: BlockStatement,

    /// If true, the control flow continues to the next case in the list,
    /// or default.
    pub fall_through: bool,
}

impl From<&naga::SwitchCase> for SwitchCase {
    fn from(case: &naga::SwitchCase) -> Self {
        Self {
            value: case.value,
            body: (&case.body).into(),
            fall_through: case.fall_through,
        }
    }
}
