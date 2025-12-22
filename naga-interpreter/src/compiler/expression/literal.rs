use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::compiler::{
    Error,
    compiler::Context,
    constant::ConstantScalar,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    types::{
        FloatWidth,
        IntWidth,
        ScalarType,
        Signedness,
    },
    util::ieee16_from_f16,
    value::{
        ScalarValue,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct LiteralExpression {
    pub literal: naga::Literal,
}

impl CompileExpression for LiteralExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        Value::compile_literal(compiler, self.literal)
    }
}

impl EvaluateExpression for LiteralExpression {
    type Output = ConstantScalar;

    fn evaluate_expression(&self, context: &Context) -> Result<ConstantScalar, Error> {
        todo!()
    }
}

pub trait CompileLiteral: Sized {
    fn compile_literal(
        compiler: &mut FunctionCompiler,
        literal: naga::Literal,
    ) -> Result<Self, Error>;
}

impl CompileLiteral for ScalarValue {
    fn compile_literal(
        compiler: &mut FunctionCompiler,
        literal: naga::Literal,
    ) -> Result<Self, Error> {
        let value = match literal {
            naga::Literal::F32(value) => {
                let value = compiler.function_builder.ins().f32const(value);
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F32),
                    value,
                }
            }
            naga::Literal::F16(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .f16const(ieee16_from_f16(value));
                ScalarValue {
                    ty: ScalarType::Float(FloatWidth::F16),
                    value,
                }
            }
            naga::Literal::U32(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new(value.into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
                    value,
                }
            }
            naga::Literal::I32(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I32, ir::immediates::Imm64::new(value.into()));
                ScalarValue {
                    ty: ScalarType::Int(Signedness::Signed, IntWidth::I32),
                    value,
                }
            }
            naga::Literal::Bool(value) => {
                let value = compiler
                    .function_builder
                    .ins()
                    .iconst(ir::types::I8, value as i64);
                ScalarValue {
                    ty: ScalarType::Bool,
                    value,
                }
            }
            _ => panic!("Invalid literal: {literal:?}"),
        };

        Ok(value)
    }
}

impl CompileLiteral for Value {
    fn compile_literal(
        compiler: &mut FunctionCompiler,
        literal: naga::Literal,
    ) -> Result<Self, Error> {
        Ok(ScalarValue::compile_literal(compiler, literal)?.into())
    }
}
