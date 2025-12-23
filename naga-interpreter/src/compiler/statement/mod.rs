#![allow(unused_variables)]

use crate::compiler::{
    Error,
    function::FunctionCompiler,
};

mod atomic;
mod barrier;
mod block;
mod r#break;
mod call;
mod r#continue;
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
pub use r#break::*;
pub use call::*;
pub use r#continue::*;
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
            ) -> Result<(), Error> {
                match self {
                    $(Self::$variant(statement) => CompileStatement::compile_statement(statement, compiler)?,)*
                }
                Ok(())
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
);

pub trait CompileStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error>;
}

impl From<naga::Statement> for Statement {
    fn from(value: naga::Statement) -> Self {
        use naga::Statement::*;

        match value {
            Emit(range) => EmitStatement { expressions: range }.into(),
            Block(block) => BlockStatement { block }.into(),
            If {
                condition,
                accept,
                reject,
            } => {
                IfStatement {
                    condition,
                    accept: accept,
                    reject,
                }
                .into()
            }
            Switch { selector, cases } => SwitchStatement { selector, cases }.into(),
            Loop {
                body,
                continuing,
                break_if,
            } => {
                LoopStatement {
                    body,
                    continuing,
                    break_if,
                }
                .into()
            }
            Break => BreakStatement.into(),
            Continue => ContinueStatement.into(),
            Return { value } => ReturnStatement { value }.into(),
            Kill => KillStatement.into(),
            ControlBarrier(barrier) => ControlBarrierStatement { barrier }.into(),
            MemoryBarrier(barrier) => MemoryBarrierStatement { barrier }.into(),
            Store { pointer, value } => StoreStatement { pointer, value }.into(),
            ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => {
                ImageStoreStatement {
                    image,
                    coordinate,
                    array_index,
                    value,
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
                    pointer,
                    function: fun,
                    value,
                    result,
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
                    image,
                    coordinate,
                    array_index,
                    function: fun,
                    value,
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
                    function,
                    arguments,
                    result,
                }
                .into()
            }
            RayQuery { query, fun } => {
                RayQueryStatement {
                    query,
                    function: fun,
                }
                .into()
            }
            SubgroupBallot { result, predicate } => {
                SubgroupBallotStatement { result, predicate }.into()
            }
            SubgroupGather {
                mode,
                argument,
                result,
            } => {
                SubgroupGatherStatement {
                    mode,
                    argument,
                    result,
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
                    operation: op,
                    collective_operation: collective_op,
                    argument,
                    result,
                }
                .into()
            }
        }
    }
}
