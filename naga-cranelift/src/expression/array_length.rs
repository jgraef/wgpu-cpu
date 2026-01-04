use cranelift_codegen::ir::{
    self,
    InstBuilder,
};

use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    types::{
        ArrayType,
        IntWidth,
        ScalarType,
        Signedness,
    },
    value::{
        PointerValue,
        PointerValueInner,
        ScalarValue,
        Value,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct ArrayLengthExpression {
    pub array: naga::Handle<naga::Expression>,
}

impl CompileExpression for ArrayLengthExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        let array_pointer: PointerValue = self.array.compile_expression(compiler)?.try_into()?;
        let base_type: ArrayType = array_pointer.ty.base_type(&compiler.context).try_into()?;

        let value = if let Some(size) = base_type.size {
            ScalarValue::compile_u32(&mut compiler.function_builder, size)
        }
        else {
            match array_pointer.inner {
                PointerValueInner::StaticPointer(pointer_range) => {
                    assert!(pointer_range.len.is_multiple_of(base_type.stride));
                    ScalarValue::compile_u32(
                        &mut compiler.function_builder,
                        pointer_range.len / base_type.stride,
                    )
                }
                PointerValueInner::DynamicPointer(pointer_range) => {
                    let mut value = compiler
                        .function_builder
                        .ins()
                        .udiv_imm(pointer_range.len, i64::from(base_type.stride));

                    // pointer_range.len is a u64 (on x86_64), so it usually has to be reduced to
                    // u32
                    let native_pointer_width = compiler.context.target_config.pointer_bytes();
                    let u32_width = IntWidth::I32.byte_width();
                    assert!(
                        native_pointer_width >= u32_width,
                        "not supported: native pointer width less than that of u32"
                    );
                    if native_pointer_width > u32_width {
                        // todo: we should make sure that we only ever create PointerRanges whose
                        // length doesn't exceed u32::MAX (i think we only create them in the
                        // runtime api functions).

                        value = compiler
                            .function_builder
                            .ins()
                            .ireduce(ir::types::I32, value);
                    }

                    ScalarValue {
                        ty: ScalarType::Int(Signedness::Unsigned, IntWidth::I32),
                        value,
                    }
                }
                PointerValueInner::StackLocation(stack_location) => {
                    let size = compiler.function_builder.func.sized_stack_slots
                        [stack_location.stack_slot]
                        .size;
                    assert!(size.is_multiple_of(base_type.stride));
                    ScalarValue::compile_u32(
                        &mut compiler.function_builder,
                        size / base_type.stride,
                    )
                }
                PointerValueInner::Handle(_) => panic!("Invalid to take length of handle pointer"),
            }
        };

        Ok(value.into())
    }
}
