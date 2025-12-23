use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct SubgroupBallotStatement {
    pub result: naga::Handle<naga::Expression>,
    pub predicate: Option<naga::Handle<naga::Expression>>,
}

impl CompileStatement for SubgroupBallotStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubgroupGatherStatement {
    pub mode: naga::GatherMode,
    pub argument: naga::Handle<naga::Expression>,
    pub result: naga::Handle<naga::Expression>,
}

impl CompileStatement for SubgroupGatherStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SubgroupCollectiveOperationStatement {
    pub operation: naga::SubgroupOperation,
    pub collective_operation: naga::CollectiveOperation,
    pub argument: naga::Handle<naga::Expression>,
    pub result: naga::Handle<naga::Expression>,
}

impl CompileStatement for SubgroupCollectiveOperationStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
