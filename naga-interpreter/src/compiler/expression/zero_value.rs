use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use half::f16;

use crate::compiler::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    types::{
        MatrixType,
        ScalarType,
        Type,
        VectorType,
    },
    util::ieee16_from_f16,
    value::{
        MatrixValue,
        ScalarValue,
        Value,
        VectorValue,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct ZeroValueExpression {
    pub ty: naga::Handle<naga::Type>,
}

impl CompileExpression for ZeroValueExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let ty = compiler.context.types[self.ty];
        Value::compile_zero_value(ty, compiler)
    }
}

impl EvaluateExpression for ZeroValueExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

pub trait CompileZeroValue<Type>: Sized {
    fn compile_zero_value(ty: Type, compiler: &mut FunctionCompiler) -> Result<Self, Error>;
}

impl CompileZeroValue<ir::Type> for ir::Value {
    fn compile_zero_value(ty: ir::Type, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let base_ty = ty.lane_of();
        let mut value = if base_ty.is_int() {
            compiler.function_builder.ins().iconst(base_ty, 0)
        }
        else if base_ty == ir::types::F16 {
            compiler
                .function_builder
                .ins()
                .f16const(ieee16_from_f16(f16::ZERO))
        }
        else if base_ty == ir::types::F32 {
            compiler.function_builder.ins().f32const(0.0)
        }
        else {
            panic!("Invalid to zero {ty:?}");
        };

        if ty.lane_count() > 1 {
            value = compiler.function_builder.ins().splat(ty, value);
        }

        Ok(value)
    }
}

impl CompileZeroValue<ScalarType> for ScalarValue {
    fn compile_zero_value(ty: ScalarType, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = ir::Value::compile_zero_value(ty.ir_type(), compiler)?;
        Ok(Self { ty, value })
    }
}

impl CompileZeroValue<VectorType> for VectorValue {
    fn compile_zero_value(ty: VectorType, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let vectorized = compiler.context.simd_context.vector(ty);

        let value = ir::Value::compile_zero_value(vectorized.ty, compiler)?;
        let values = std::iter::repeat(value)
            .take(vectorized.count.into())
            .collect();

        Ok(Self { ty, values })
    }
}

impl CompileZeroValue<MatrixType> for MatrixValue {
    fn compile_zero_value(ty: MatrixType, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let vectorized = compiler.context.simd_context.matrix(ty);

        let value = ir::Value::compile_zero_value(vectorized.ty, compiler)?;
        let values = std::iter::repeat(value)
            .take(vectorized.count.into())
            .collect();

        Ok(Self { ty, values })
    }
}

impl CompileZeroValue<Type> for Value {
    fn compile_zero_value(ty: Type, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match ty {
            Type::Scalar(ty) => ScalarValue::compile_zero_value(ty, compiler)?.into(),
            Type::Vector(ty) => VectorValue::compile_zero_value(ty, compiler)?.into(),
            Type::Matrix(ty) => MatrixValue::compile_zero_value(ty, compiler)?.into(),
            Type::Struct(_ty) => todo!(),
            Type::Array(_ty) => todo!(),
            _ => panic!("Invalid to zero {ty:?}"),
        };
        Ok(value)
    }
}
