#![allow(unused_variables)]

use crate::{
    Error,
    function::FunctionCompiler,
};

mod atomic;
mod barrier;
mod block;
mod call;
mod emit;
mod r#if;
mod image;
mod kill;
mod r#loop;
mod ray_query;
mod r#return;
mod store;
mod subgroup;
mod switch;
mod work_group;

pub use atomic::*;
pub use barrier::*;
pub use block::*;
pub use call::*;
pub use emit::*;
pub use r#if::*;
pub use image::*;
pub use kill::*;
pub use r#loop::*;
pub use ray_query::*;
pub use r#return::*;
pub use store::*;
pub use subgroup::*;
pub use switch::*;
pub use work_group::*;

#[derive(Clone, Copy, Debug)]
#[must_use]
pub enum ControlFlow {
    Continue,
    Diverged,
}

impl ControlFlow {
    pub fn is_continuing(&self) -> bool {
        matches!(self, Self::Continue)
    }

    pub fn is_diverged(&self) -> bool {
        matches!(self, Self::Diverged)
    }
}

pub trait CompileStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error>;
}

macro_rules! define_statement {
    ($($variant:ident($ty:ty),)*) => {
        #[derive(Clone, Debug)]
        pub enum Statement {
            $($variant($ty),)*
        }

        impl CompileStatement for Statement {
            fn compile_statement(
                &self,
                compiler: &mut FunctionCompiler,
            ) -> Result<ControlFlow, Error> {
                let control_flow = match self {
                    $(Self::$variant(statement) => CompileStatement::compile_statement(statement, compiler)?,)*
                };
                Ok(control_flow)
            }
        }

        $(
            impl From<$ty> for Statement {
                fn from(statement: $ty) -> Self {
                    Self::$variant(statement)
                }
            }

            impl TryFrom<Statement> for $ty {
                type Error = ();

                fn try_from(statement: Statement) -> Result<Self, Self::Error> {
                    match statement {
                        Statement::$variant(statement) => Ok(statement),
                        _ => Err(())
                    }
                }
            }
        )*
    };
}

define_statement!(
    Emit(EmitStatement),
    Block(BlockStatement),
    If(IfStatement),
    Switch(SwitchStatement),
    Loop(LoopStatement),
    Break(BreakStatement),
    Continue(ContinueStatement),
    Return(ReturnStatement),
    Kill(KillStatement),
    ControlBarrier(ControlBarrierStatement),
    MemoryBarrier(MemoryBarrierStatement),
    Store(StoreStatement),
    ImageStore(ImageStoreStatement),
    Atomic(AtomicStatement),
    ImageAtomic(ImageAtomicStatement),
    WorkGroupUniformLoad(WorkGroupUniformLoadStatement),
    Call(CallStatement),
    RayQuery(RayQueryStatement),
    SubgroupBallot(SubgroupBallotStatement),
    SubgroupGather(SubgroupGatherStatement),
    SubgroupCollectiveOperation(SubgroupCollectiveOperationStatement),
    CooperativeStore(CooperativeStoreStatement),
);

impl From<&naga::Statement> for Statement {
    fn from(value: &naga::Statement) -> Self {
        use naga::Statement::*;

        match value {
            Emit(range) => {
                EmitStatement {
                    expressions: range.clone(),
                }
                .into()
            }
            Block(block) => BlockStatement::from(block).into(),
            If {
                condition,
                accept,
                reject,
            } => {
                IfStatement {
                    condition: *condition,
                    accept: accept.into(),
                    reject: reject.into(),
                }
                .into()
            }
            Switch { selector, cases } => {
                SwitchStatement {
                    selector: *selector,
                    cases: cases.iter().map(|case| case.into()).collect(),
                }
                .into()
            }
            Loop {
                body,
                continuing,
                break_if,
            } => {
                LoopStatement {
                    body: body.into(),
                    continuing: continuing.into(),
                    break_if: *break_if,
                }
                .into()
            }
            Break => BreakStatement.into(),
            Continue => ContinueStatement.into(),
            Return { value } => ReturnStatement { value: *value }.into(),
            Kill => KillStatement.into(),
            ControlBarrier(barrier) => ControlBarrierStatement { barrier: *barrier }.into(),
            MemoryBarrier(barrier) => MemoryBarrierStatement { barrier: *barrier }.into(),
            Store { pointer, value } => {
                StoreStatement {
                    pointer: *pointer,
                    value: *value,
                }
                .into()
            }
            ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                ImageStoreStatement {
                    image: *image,
                    coordinate: *coordinate,
                    array_index: *array_index,
                    value: *value,
                }
                .into()
            }
            Atomic {
                pointer,
                fun,
                value,
                result,
            } => {
                AtomicStatement {
                    pointer: *pointer,
                    function: *fun,
                    value: *value,
                    result: *result,
                }
                .into()
            }
            ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => {
                ImageAtomicStatement {
                    image: *image,
                    coordinate: *coordinate,
                    array_index: *array_index,
                    function: *fun,
                    value: *value,
                }
                .into()
            }
            WorkGroupUniformLoad { pointer, result } => WorkGroupUniformLoadStatement {}.into(),
            Call {
                function,
                arguments,
                result,
            } => {
                CallStatement {
                    function: *function,
                    arguments: arguments.clone(),
                    result: *result,
                }
                .into()
            }
            RayQuery { query, fun } => {
                RayQueryStatement {
                    query: *query,
                    function: fun.clone(),
                }
                .into()
            }
            SubgroupBallot { result, predicate } => {
                SubgroupBallotStatement {
                    result: *result,
                    predicate: *predicate,
                }
                .into()
            }
            SubgroupGather {
                mode,
                argument,
                result,
            } => {
                SubgroupGatherStatement {
                    mode: *mode,
                    argument: *argument,
                    result: *result,
                }
                .into()
            }
            SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => {
                SubgroupCollectiveOperationStatement {
                    operation: *op,
                    collective_operation: *collective_op,
                    argument: *argument,
                    result: *result,
                }
                .into()
            }
            CooperativeStore { target, data } => {
                CooperativeStoreStatement {
                    target: *target,
                    data: *data,
                }
                .into()
            }
        }
    }
}
