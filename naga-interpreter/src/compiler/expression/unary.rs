use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

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
pub struct UnaryExpression {
    pub operator: naga::UnaryOperator,
    pub operand: naga::Handle<naga::Expression>,
}

impl CompileExpression for UnaryExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        use naga::UnaryOperator::*;
        let input_value = compiler.compile_expression(self.operand)?;

        let output = match self.operator {
            Negate => input_value.compile_neg(compiler)?.into(),
            LogicalNot => input_value.compile_log_not(compiler)?.into(),
            BitwiseNot => input_value.compile_bit_not(compiler)?.into(),
        };

        Ok(output)
    }
}

impl EvaluateExpression for UnaryExpression {
    type Output = ConstantValue;

    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

macro_rules! define_unary_traits {
    {$(
        $trait:ident :: $method:ident
        $(auto $auto_kind:ident $auto_args:tt)*
    ;)*} => {
        $(
            pub trait $trait {
                type Output: Sized;

                fn $method(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error>;
            }

            $(
                define_unary_traits!(@impl_auto($trait, $method, $auto_kind, $auto_args));
            )?
        )*
    };
    (@impl_auto($trait:ident, $method:ident, Value, [$($variant:ident),*])) => {
        impl $trait for Value {
            type Output = Self;

            fn $method(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                let output = match self {
                    $(
                        Self::$variant(inner) => $trait::$method(inner, compiler)?.into(),
                    )*
                    _ => panic!("{} not implemented for {:?}", stringify!($trait), self.type_of()),
                };
                Ok(output)
            }
        }
    };
    (@impl_auto($trait:ident, $method:ident, map, [$($ty:ident),*])) => {
        $(
            impl $trait for $ty {
                type Output = Self;

                fn $method(&self, compiler: &mut FunctionCompiler) -> Result<Self::Output, Error> {
                    self.try_map_as_scalars(|scalar| $trait::$method(&scalar, compiler))
                }
            }
        )*
    }
}

define_unary_traits! {
    CompileNeg::compile_neg
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];

    CompileLogNot::compile_log_not
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];

    CompileBitNot::compile_bit_not
        auto Value[Scalar, Vector, Matrix]
        auto map[VectorValue, MatrixValue];
}

impl CompileNeg for ScalarValue {
    type Output = Self;

    fn compile_neg(&self, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Int(Signedness::Signed, _int_width) => {
                compiler.function_builder.ins().ineg(self.value)
            }
            ScalarType::Float(_float_width) => compiler.function_builder.ins().fneg(self.value),
            _ => panic!("negation is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}

impl CompileBitNot for ScalarValue {
    type Output = Self;

    fn compile_bit_not(&self, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Int(_signedness, _int_width) => {
                compiler.function_builder.ins().bnot(self.value)
            }
            _ => panic!("bitwise not is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}

impl CompileLogNot for ScalarValue {
    type Output = Self;

    fn compile_log_not(&self, compiler: &mut FunctionCompiler) -> Result<Self, Error> {
        let value = match self.ty {
            ScalarType::Bool => {
                compiler
                    .function_builder
                    .ins()
                    .icmp_imm(ir::condcodes::IntCC::Equal, self.value, 0)
            }
            _ => panic!("logical not is not valid for {:?}", self.ty),
        };

        Ok(self.with_ir_value(value))
    }
}
