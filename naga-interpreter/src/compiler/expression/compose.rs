use cranelift_codegen::ir::InstBuilder;

use crate::compiler::{
    Error,
    compiler::Context,
    constant::{
        ConstantArray,
        ConstantMatrix,
        ConstantScalar,
        ConstantStruct,
        ConstantValue,
        ConstantVector,
    },
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    simd::VectorIrType,
    types::{
        ArrayType,
        MatrixType,
        StructType,
        Type,
        VectorType,
    },
    value::{
        ArrayValue,
        AsIrValue,
        MatrixValue,
        ScalarValue,
        StructValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

#[derive(Clone, Debug)]
pub struct ComposeExpression {
    pub ty: naga::Handle<naga::Type>,
    pub components: Vec<naga::Handle<naga::Expression>>,
}

impl CompileExpression for ComposeExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let ty = compiler.context.types[self.ty];
        let components = self
            .components
            .iter()
            .map(|expression| expression.compile_expression(compiler))
            .collect::<Result<Vec<_>, Error>>()?;

        Value::compile_compose(compiler, ty, components)
    }
}

impl EvaluateExpression for ComposeExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        let ty = context.types[self.ty];

        let components = self
            .components
            .iter()
            .map(|handle| handle.evaluate_expression(context))
            .collect::<Result<Vec<ConstantValue>, Error>>()?;

        ConstantValue::evaluate_compose(context, ty, components)
    }
}

pub trait CompileCompose<Inner>: Sized + TypeOf {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: Self::Type,
        components: Vec<Inner>,
    ) -> Result<Self, Error>;
}

impl CompileCompose<ScalarValue> for VectorValue {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: VectorType,
        components: Vec<ScalarValue>,
    ) -> Result<Self, Error> {
        let vectorization = function_compiler.context.simd_context[ty];

        let values = match vectorization {
            VectorIrType::Plain { ty: _ } => {
                components
                    .into_iter()
                    .map(|component| component.as_ir_value())
                    .collect()
            }
            VectorIrType::Vector { ty: ir_ty } => {
                // fixme: this triggers a bug in cranelift when inserting into a vector type
                // that fits into a single lane. e.g. by composing a vec2i(1, 2)
                //
                // https://github.com/bytecodealliance/wasmtime/issues/12165
                // https://github.com/bytecodealliance/wasmtime/issues/12197

                const CRANELIFT_LOWERING_WORKAROUND: bool = false;
                let value = if components.len() == 2 && CRANELIFT_LOWERING_WORKAROUND {
                    //function_compiler.function_builder.ins().uunarrow(x, y)
                    todo!("workaround");
                }
                else {
                    let mut components = components.into_iter();
                    let first = components.next().unwrap();

                    let mut vector = function_compiler
                        .function_builder
                        .ins()
                        .splat(ir_ty, first.value);
                    let mut lane = 1;

                    for component in components {
                        vector = function_compiler.function_builder.ins().insertlane(
                            vector,
                            component.value,
                            lane as u8,
                        );

                        lane += 1;
                    }

                    vector
                };

                std::iter::once(value).collect()
            }
        };

        Ok(VectorValue { ty, values })
    }
}

#[allow(unused_variables)]
impl CompileCompose<ScalarValue> for MatrixValue {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: MatrixType,
        components: Vec<ScalarValue>,
    ) -> Result<Self, Error> {
        todo!("compose matrix from scalars");
    }
}

#[allow(unused_variables)]
impl CompileCompose<VectorValue> for MatrixValue {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: MatrixType,
        components: Vec<VectorValue>,
    ) -> Result<Self, Error> {
        todo!("compose matrix from vectors");
    }
}

impl CompileCompose<Value> for StructValue {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: StructType,
        components: Vec<Value>,
    ) -> Result<Self, Error> {
        let _ = function_compiler;
        Ok(Self {
            ty,
            members: components,
        })
    }
}

impl CompileCompose<Value> for ArrayValue {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: ArrayType,
        components: Vec<Value>,
    ) -> Result<Self, Error> {
        let _ = function_compiler;
        Ok(Self {
            ty,
            values: components,
        })
    }
}

impl CompileCompose<Value> for Value {
    fn compile_compose(
        function_compiler: &mut FunctionCompiler,
        ty: Type,
        components: Vec<Value>,
    ) -> Result<Self, Error> {
        let value = match ty {
            Type::Vector(vector_type) => {
                let components = components
                    .into_iter()
                    .map(|component| {
                        match component {
                            Value::Scalar(scalar_value) => scalar_value,
                            _ => {
                                panic!(
                                    "Compose is invalid for {ty:?} with components of {:?}",
                                    component.type_of()
                                )
                            }
                        }
                    })
                    .collect();
                VectorValue::compile_compose(function_compiler, vector_type, components)?.into()
            }
            Type::Matrix(matrix_type) => {
                match &components[0] {
                    Value::Scalar(_) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                match component {
                                    Value::Scalar(scalar_value) => scalar_value,
                                    _ => panic!("Mixed compose is invalid for matrices"),
                                }
                            })
                            .collect();
                        MatrixValue::compile_compose(function_compiler, matrix_type, components)?
                            .into()
                    }
                    Value::Vector(_) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                match component {
                                    Value::Vector(vector_value) => vector_value,
                                    _ => panic!("Mixed compose is invalid for matrices"),
                                }
                            })
                            .collect();
                        MatrixValue::compile_compose(function_compiler, matrix_type, components)?
                            .into()
                    }
                    _ => {
                        panic!(
                            "Compose is invalid for {ty:?} with components of {:?}",
                            components[0].type_of()
                        )
                    }
                }
            }
            Type::Struct(struct_type) => {
                StructValue::compile_compose(function_compiler, struct_type, components)?.into()
            }
            Type::Array(array_type) => {
                ArrayValue::compile_compose(function_compiler, array_type, components)?.into()
            }
            _ => panic!("Compose is invalid for {ty:?}"),
        };
        Ok(value)
    }
}

pub trait EvaluateCompose<Inner>: Sized + TypeOf {
    fn evaluate_compose(
        context: &Context,
        ty: Self::Type,
        components: Vec<Inner>,
    ) -> Result<Self, Error>;
}

impl EvaluateCompose<ConstantValue> for ConstantValue {
    fn evaluate_compose(
        context: &Context,
        ty: Type,
        components: Vec<ConstantValue>,
    ) -> Result<Self, Error> {
        let value = match ty {
            Type::Vector(vector_type) => {
                let components = components
                    .into_iter()
                    .map(|component| component.try_into().expect("expected scalar constant"))
                    .collect();
                <ConstantVector as EvaluateCompose<ConstantScalar>>::evaluate_compose(
                    context,
                    vector_type,
                    components,
                )?
                .into()
            }
            Type::Matrix(matrix_type) => {
                let inner_type = components[0].type_of();
                match inner_type {
                    Type::Scalar(scalar_type) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                component.try_into().expect("expected scalar constant")
                            })
                            .collect();
                        <ConstantMatrix as EvaluateCompose<ConstantScalar>>::evaluate_compose(
                            context,
                            matrix_type,
                            components,
                        )?
                        .into()
                    }
                    Type::Vector(vector_type) => {
                        let components = components
                            .into_iter()
                            .map(|component| {
                                component.try_into().expect("expected scalar constant")
                            })
                            .collect();
                        <ConstantMatrix as EvaluateCompose<ConstantVector>>::evaluate_compose(
                            context,
                            matrix_type,
                            components,
                        )?
                        .into()
                    }
                    _ => panic!("Invalid to compose matrix from {inner_type:?}"),
                }
            }
            Type::Struct(struct_type) => {
                <ConstantStruct as EvaluateCompose<ConstantValue>>::evaluate_compose(
                    context,
                    struct_type,
                    components,
                )?
                .into()
            }
            Type::Array(array_type) => {
                <ConstantArray as EvaluateCompose<ConstantValue>>::evaluate_compose(
                    context, array_type, components,
                )?
                .into()
            }
            _ => panic!("Compose is invalid for {ty:?}"),
        };

        Ok(value)
    }
}
