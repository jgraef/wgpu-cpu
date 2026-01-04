use std::cmp::Ordering;

use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    types::{
        CastTo,
        ScalarType,
        Signedness,
    },
    value::{
        MatrixValue,
        ScalarValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct AsExpression {
    pub expression: naga::Handle<naga::Expression>,
    pub target: CastTo,
}

impl CompileExpression for AsExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let input_value = self.expression.compile_expression(compiler)?;
        input_value.compile_as(compiler, self.target)
    }
}

impl EvaluateExpression for AsExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

pub trait CompileAs {
    type Output: Sized;

    fn compile_as(
        &self,
        compiler: &mut FunctionCompiler,
        target: CastTo,
    ) -> Result<Self::Output, Error>;
}

impl CompileAs for ScalarValue {
    type Output = Self;

    fn compile_as(&self, compiler: &mut FunctionCompiler, target: CastTo) -> Result<Self, Error> {
        // see https://gpuweb.github.io/gpuweb/wgsl/#value-constructor-builtin-function

        let input_type = self.ty;
        let output_type = input_type.cast(target);

        let value = match (input_type, output_type) {
            (ScalarType::Bool, ScalarType::Bool) => self.value,
            (ScalarType::Bool, ScalarType::Int(_signedness, _int_width)) => {
                compiler
                    .function_builder
                    .ins()
                    .uextend(output_type.ir_type(), self.value)
            }
            (ScalarType::Bool, ScalarType::Float(_float_width)) => {
                compiler
                    .function_builder
                    .ins()
                    .fcvt_from_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Int(_signedness, _int_width), ScalarType::Bool) => {
                compiler.function_builder.ins().icmp_imm(
                    ir::condcodes::IntCC::NotEqual,
                    self.value,
                    0,
                )
            }
            (
                ScalarType::Int(_signedness, input_int_width),
                ScalarType::Int(output_signedness, output_int_width),
            ) => {
                match input_int_width.cmp(&output_int_width) {
                    Ordering::Less => {
                        match output_signedness {
                            Signedness::Signed => {
                                compiler
                                    .function_builder
                                    .ins()
                                    .sextend(output_type.ir_type(), self.value)
                            }
                            Signedness::Unsigned => {
                                compiler
                                    .function_builder
                                    .ins()
                                    .uextend(output_type.ir_type(), self.value)
                            }
                        }
                    }
                    Ordering::Equal => self.value,
                    Ordering::Greater => {
                        compiler
                            .function_builder
                            .ins()
                            .ireduce(output_type.ir_type(), self.value)
                    }
                }
            }

            (ScalarType::Int(Signedness::Signed, _int_width), ScalarType::Float(_float_width)) => {
                compiler
                    .function_builder
                    .ins()
                    .fcvt_from_sint(output_type.ir_type(), self.value)
            }
            (
                ScalarType::Int(Signedness::Unsigned, _int_width),
                ScalarType::Float(_float_width),
            ) => {
                compiler
                    .function_builder
                    .ins()
                    .fcvt_from_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Float(float_width), ScalarType::Bool) => {
                let zero =
                    ScalarValue::compile_neg_zero(&mut compiler.function_builder, float_width);
                compiler.function_builder.ins().fcmp(
                    ir::condcodes::FloatCC::NotEqual,
                    self.value,
                    zero.value,
                )
            }
            (ScalarType::Float(_float_width), ScalarType::Int(Signedness::Signed, _int_width)) => {
                compiler
                    .function_builder
                    .ins()
                    .fcvt_to_sint(output_type.ir_type(), self.value)
            }
            (
                ScalarType::Float(_float_width),
                ScalarType::Int(Signedness::Unsigned, _int_width),
            ) => {
                compiler
                    .function_builder
                    .ins()
                    .fcvt_to_uint(output_type.ir_type(), self.value)
            }
            (ScalarType::Float(input_float_width), ScalarType::Float(output_float_width)) => {
                match input_float_width.cmp(&output_float_width) {
                    Ordering::Less => {
                        compiler
                            .function_builder
                            .ins()
                            .fpromote(output_type.ir_type(), self.value)
                    }
                    Ordering::Equal => self.value,
                    Ordering::Greater => {
                        compiler
                            .function_builder
                            .ins()
                            .fdemote(output_type.ir_type(), self.value)
                    }
                }
            }
        };

        Ok(Self {
            ty: output_type,
            value,
        })
    }
}

impl CompileAs for VectorValue {
    type Output = Self;

    fn compile_as(&self, compiler: &mut FunctionCompiler, target: CastTo) -> Result<Self, Error> {
        self.try_map_as_scalars(|scalar| scalar.compile_as(compiler, target))
    }
}

impl CompileAs for MatrixValue {
    type Output = Self;

    fn compile_as(&self, compiler: &mut FunctionCompiler, target: CastTo) -> Result<Self, Error> {
        self.try_map_as_scalars(|scalar| scalar.compile_as(compiler, target))
    }
}

impl CompileAs for Value {
    type Output = Self;

    fn compile_as(&self, compiler: &mut FunctionCompiler, target: CastTo) -> Result<Self, Error> {
        let value = match self {
            Value::Scalar(scalar_value) => scalar_value.compile_as(compiler, target)?.into(),
            Value::Vector(vector_value) => vector_value.compile_as(compiler, target)?.into(),
            Value::Matrix(matrix_value) => matrix_value.compile_as(compiler, target)?.into(),
            _ => {
                panic!("Invalid cast: {:?} as {target:?}", self.type_of(),)
            }
        };

        Ok(value)
    }
}
