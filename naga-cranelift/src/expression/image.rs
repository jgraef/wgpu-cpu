use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    value::Value,
};

#[derive(Clone, Copy, Debug)]
pub struct ImageSampleExpression {
    pub image: naga::Handle<naga::Expression>,
    pub sampler: naga::Handle<naga::Expression>,
    pub gather: Option<naga::SwizzleComponent>,
    pub coordinate: naga::Handle<naga::Expression>,
    pub array_index: Option<naga::Handle<naga::Expression>>,
    pub offset: Option<naga::Handle<naga::Expression>>,
    pub level: naga::SampleLevel,
    pub depth_ref: Option<naga::Handle<naga::Expression>>,
    pub clamp_to_edge: bool,
}

impl CompileExpression for ImageSampleExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageLoadExpression {
    pub image: naga::Handle<naga::Expression>,
    pub coordinate: naga::Handle<naga::Expression>,
    pub array_index: Option<naga::Handle<naga::Expression>>,
    pub sample: Option<naga::Handle<naga::Expression>>,
    pub level: Option<naga::Handle<naga::Expression>>,
}

impl CompileExpression for ImageLoadExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ImageQueryExpression {
    pub image: naga::Handle<naga::Expression>,
    pub query: naga::ImageQuery,
}

impl CompileExpression for ImageQueryExpression {
    fn compile_expression(&self, compiler: &mut FunctionCompiler) -> Result<Value, Error> {
        todo!()
    }
}
