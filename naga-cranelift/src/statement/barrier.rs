use crate::{
    Error,
    function::FunctionCompiler,
    statement::CompileStatement,
};

#[derive(Clone, Copy, Debug)]
pub struct ControlBarrierStatement {
    pub barrier: naga::Barrier,
}

impl CompileStatement for ControlBarrierStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MemoryBarrierStatement {
    pub barrier: naga::Barrier,
}

impl CompileStatement for MemoryBarrierStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}
