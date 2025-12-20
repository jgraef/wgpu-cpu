use cranelift_codegen::ir::{
    InstBuilder,
    StackSlot,
    Value,
};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::context::Context;

#[derive(Clone, Copy, Debug)]
pub enum ValueExt {
    Value(Value),
    Stack { slot: StackSlot, offset: u32 },
}

impl ValueExt {
    pub fn value(&self) -> Value {
        match self {
            ValueExt::Value(value) => *value,
            _ => panic!("expected a plain value: {:?}", self),
        }
    }

    pub fn as_abi(&self, context: &Context, function_builder: &mut FunctionBuilder) -> Value {
        match self {
            ValueExt::Value(value) => *value,
            ValueExt::Stack { slot, offset } => {
                function_builder.ins().stack_addr(
                    context.pointer_type(),
                    *slot,
                    i32::try_from(*offset).expect("stack offset overflow"),
                )
            }
        }
    }

    /// self must be either a struct or array on the stack, or a pointer to a
    /// struct or array
    pub fn with_offset(&self, function_builder: &mut FunctionBuilder, offset: u32) -> Self {
        match self {
            ValueExt::Value(value) => {
                // pointer
                function_builder
                    .ins()
                    .iadd_imm(*value, i64::from(offset))
                    .into()
            }
            ValueExt::Stack {
                slot,
                offset: base_offset,
            } => {
                // on stack
                ValueExt::Stack {
                    slot: *slot,
                    offset: base_offset + offset,
                }
            }
        }
    }
}

impl From<Value> for ValueExt {
    fn from(value: Value) -> Self {
        Self::Value(value)
    }
}

impl From<StackSlot> for ValueExt {
    fn from(value: StackSlot) -> Self {
        Self::Stack {
            slot: value,
            offset: 0,
        }
    }
}
