use arrayvec::ArrayVec;
use cranelift_codegen::ir::InstBuilder;

use crate::{
    Error,
    compiler::Context,
    constant::ConstantValue,
    expression::{
        CompileExpression,
        EvaluateExpression,
    },
    function::FunctionCompiler,
    simd::{
        MatrixIrType,
        VectorIrType,
    },
    types::{
        PointerType,
        PointerTypeBase,
        Type,
    },
    value::{
        ArrayValue,
        MatrixValue,
        PointerOffset,
        PointerValue,
        ScalarValue,
        StructValue,
        TypeOf,
        Value,
        VectorValue,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct AccessExpression {
    pub base: naga::Handle<naga::Expression>,
    pub index: naga::Handle<naga::Expression>,
}

impl CompileExpression for AccessExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let value = self.base.compile_expression(compiler)?;
        let index: ScalarValue = self.index.compile_expression(compiler)?.try_into()?;
        assert!(
            index.ty.is_integer(),
            "index value is not an integer: {:?}",
            index.ty
        );
        value.compile_access(compiler, &index)
    }
}

impl EvaluateExpression for AccessExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AccessIndexExpression {
    pub base: naga::Handle<naga::Expression>,
    pub index: u32,
}

impl CompileExpression for AccessIndexExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let value = self.base.compile_expression(compiler)?;
        value.compile_access(compiler, &self.index)
    }
}

impl EvaluateExpression for AccessIndexExpression {
    fn evaluate_expression(&self, context: &Context) -> Result<ConstantValue, Error> {
        todo!()
    }
}

pub trait CompileAccess<Index> {
    type Output;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &Index,
    ) -> Result<Self::Output, Error>;
}

impl CompileAccess<u32> for VectorValue {
    type Output = ScalarValue;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &u32,
    ) -> Result<Self::Output, Error> {
        assert!(
            *index < u32::from(self.ty.size),
            "vector index out of bounds"
        );

        let vectorization = compiler.context.simd_context.vector(self.ty);

        let index = u8::try_from(*index).expect("vector index overflow");
        let value_index = index / vectorization.lanes;
        let lane_index = index / vectorization.lanes;
        let mut value = self.values[usize::from(value_index)];

        if vectorization.ty.is_vector() {
            value = compiler
                .function_builder
                .ins()
                .extractlane(value, lane_index);
        }
        else {
            assert_eq!(lane_index, 0);
        }

        Ok(ScalarValue {
            ty: self.ty.scalar,
            value,
        })
    }
}

impl CompileAccess<ScalarValue> for VectorValue {
    type Output = ScalarValue;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &ScalarValue,
    ) -> Result<ScalarValue, Error> {
        let mut index = index.value;

        let value = match compiler.context.simd_context[self.ty] {
            VectorIrType::Plain { ty } => {
                let v0_or_v1 =
                    compiler
                        .function_builder
                        .ins()
                        .select(index, self.values[1], self.values[0]);
                index = compiler.function_builder.ins().ushr_imm(index, 1);
                match self.ty.size {
                    naga::VectorSize::Bi => v0_or_v1,
                    naga::VectorSize::Tri => {
                        compiler
                            .function_builder
                            .ins()
                            .select(index, self.values[2], v0_or_v1)
                    }
                    naga::VectorSize::Quad => {
                        let v2_or_v3 = compiler.function_builder.ins().select(
                            index,
                            self.values[3],
                            self.values[2],
                        );
                        compiler
                            .function_builder
                            .ins()
                            .select(index, v2_or_v3, v0_or_v1)
                    }
                }
            }
            VectorIrType::Vector { ty } => {
                let value = self.values[0];
                let v0 = compiler.function_builder.ins().extractlane(value, 0);
                let v1 = compiler.function_builder.ins().extractlane(value, 1);
                let v0_or_v1 = compiler.function_builder.ins().select(index, v1, v0);
                index = compiler.function_builder.ins().ushr_imm(index, 1);
                match self.ty.size {
                    naga::VectorSize::Bi => v0_or_v1,
                    naga::VectorSize::Tri => {
                        let v2 = compiler.function_builder.ins().extractlane(value, 2);
                        compiler.function_builder.ins().select(index, v2, v0_or_v1)
                    }
                    naga::VectorSize::Quad => {
                        let v2 = compiler.function_builder.ins().extractlane(value, 2);
                        let v3 = compiler.function_builder.ins().extractlane(value, 3);
                        let v2_or_v3 = compiler.function_builder.ins().select(index, v2, v3);
                        compiler
                            .function_builder
                            .ins()
                            .select(index, v2_or_v3, v0_or_v1)
                    }
                }
            }
        };

        Ok(ScalarValue {
            ty: self.ty.scalar,
            value,
        })
    }
}

impl CompileAccess<u32> for MatrixValue {
    type Output = VectorValue;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &u32,
    ) -> Result<Self::Output, Error> {
        assert!(
            *index < u32::from(self.ty.num_elements()),
            "matrix index out of bounds"
        );
        let index = u8::try_from(*index).expect("vector index overflow");
        let mut values = ArrayVec::new();

        match compiler.context.simd_context[self.ty] {
            MatrixIrType::Plain { ty } => {
                let index = u8::from(self.ty.rows) * index;
                values.extend(
                    self.values[usize::from(index)..][..usize::from(u8::from(self.ty.rows))]
                        .iter()
                        .copied(),
                );
            }
            MatrixIrType::ColumnVector { ty } => {
                values.push(self.values[usize::from(index)]);
            }
            MatrixIrType::FullVector { ty } => {
                let matrix_value = self.values[0];
                let index = u8::from(self.ty.rows) * index;
                for lane in 0..u8::from(self.ty.rows) {
                    let value = compiler
                        .function_builder
                        .ins()
                        .extractlane(matrix_value, index + lane);
                    values.push(value);
                }
            }
        }

        Ok(VectorValue {
            ty: self.ty.column_vector(),
            values,
        })
    }
}

impl CompileAccess<ScalarValue> for MatrixValue {
    type Output = VectorValue;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &ScalarValue,
    ) -> Result<VectorValue, Error> {
        todo!("dynamic matrix access")
    }
}

impl CompileAccess<u32> for StructValue {
    type Output = Value;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &u32,
    ) -> Result<Self::Output, Error> {
        let index = usize::try_from(*index).expect("struct member overflow");
        Ok(self.members[index].clone())
    }
}

impl CompileAccess<u32> for ArrayValue {
    type Output = Value;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &u32,
    ) -> Result<Self::Output, Error> {
        let index = usize::try_from(*index).expect("array index overflow");
        Ok(self.values[index].clone())
    }
}

impl CompileAccess<ScalarValue> for ArrayValue {
    type Output = Value;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &ScalarValue,
    ) -> Result<Value, Error> {
        todo!("dynamic array access")
    }
}

impl CompileAccess<u32> for PointerValue {
    type Output = PointerValue;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &u32,
    ) -> Result<PointerValue, Error> {
        let base_type = self.ty.base_type(compiler.context);

        let (base_type, offset) = match base_type {
            Type::Vector(vector_type) => {
                let offset = u32::from(vector_type.scalar.byte_width()) * *index;
                let base_type = PointerTypeBase::ScalarPointer(vector_type.scalar);
                (base_type, offset)
            }
            Type::Matrix(matrix_type) => {
                let offset = u32::from(matrix_type.column_stride())
                    * u32::from(matrix_type.scalar.byte_width())
                    * *index;
                let base_type = PointerTypeBase::VectorPointer(matrix_type.column_vector());
                (base_type, offset)
            }
            Type::Struct(struct_type) => {
                let index = usize::try_from(*index).expect("pointer index overflow");
                let member = &struct_type.members(compiler.context.source)[index];
                let base_type = PointerTypeBase::Pointer(member.ty);
                (base_type, member.offset)
            }
            Type::Array(array_type) => {
                let offset = array_type.stride * *index;
                let base_type = PointerTypeBase::Pointer(array_type.base_type);
                (base_type, offset)
            }
            _ => panic!("Invalid to access into {base_type:?}"),
        };

        Ok(PointerValue {
            ty: PointerType {
                base_type,
                address_space: self.ty.address_space,
            },
            inner: self.inner.with_offset(offset),
        })
    }
}

impl CompileAccess<ScalarValue> for PointerValue {
    type Output = Value;

    fn compile_access(
        &self,
        compiler: &mut FunctionCompiler,
        index: &ScalarValue,
    ) -> Result<Value, Error> {
        let base_type = self.ty.base_type(compiler.context);
        let index = index.value;

        let (stride, new_base_type) = match base_type {
            Type::Vector(vector_type) => {
                let stride = i64::from(vector_type.scalar.byte_width());
                let new_base_type = PointerTypeBase::ScalarPointer(vector_type.scalar);
                (stride, new_base_type)
            }
            Type::Matrix(matrix_type) => {
                let stride = i64::from(matrix_type.column_stride())
                    * i64::from(matrix_type.scalar.byte_width());
                let new_base_type = PointerTypeBase::VectorPointer(matrix_type.column_vector());
                (stride, new_base_type)
            }
            Type::Array(array_type) => {
                let stride = i64::from(array_type.stride);
                let new_base_type = PointerTypeBase::Pointer(array_type.base_type);
                (stride, new_base_type)
            }
            _ => {
                panic!(
                    "Invalid to dynamically access index into pointer: {:?}",
                    self.ty
                )
            }
        };

        let offset = compiler.function_builder.ins().imul_imm(index, stride);
        let value = self.with_dynamic_offset(
            compiler.context,
            &mut compiler.function_builder,
            offset,
            new_base_type,
        );

        Ok(value.into())
    }
}

macro_rules! impl_compile_access_for_value {
    ($index:ty {$($variant:ident,)*}) => {

        impl CompileAccess<$index> for Value {
            type Output = Value;

            fn compile_access(
                &self,
                compiler: &mut FunctionCompiler,
                index: &$index,
            ) -> Result<Value, Error> {
                let value = match self {
                    $(Value::$variant(value) => CompileAccess::compile_access(value, compiler, index)?.into(),)*
                    _ => panic!("Invalid to access {:?} with {index:?}", self.type_of()),
                };
                Ok(value)
            }
        }
    };
}

impl_compile_access_for_value!(u32 {
    Vector,
    Matrix,
    Struct,
    Array,
    Pointer,
});

impl_compile_access_for_value!(ScalarValue {
    Vector,
    Matrix,
    Array,
    Pointer,
});
