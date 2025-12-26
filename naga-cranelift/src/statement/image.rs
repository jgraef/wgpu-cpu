use crate::{
    Error,
    function::FunctionCompiler,
    statement::{
        CompileStatement,
        ControlFlow,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct ImageStoreStatement {
    pub image: naga::Handle<naga::Expression>,
    pub coordinate: naga::Handle<naga::Expression>,
    pub array_index: Option<naga::Handle<naga::Expression>>,
    pub value: naga::Handle<naga::Expression>,
}

impl CompileStatement for ImageStoreStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageAtomicStatement {
    pub image: naga::Handle<naga::Expression>,
    pub coordinate: naga::Handle<naga::Expression>,
    pub array_index: Option<naga::Handle<naga::Expression>>,
    pub function: naga::AtomicFunction,
    pub value: naga::Handle<naga::Expression>,
}

impl CompileStatement for ImageAtomicStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<ControlFlow, Error> {
        todo!()
    }
}
