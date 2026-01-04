use crate::{
    Error,
    expression::CompileExpression,
    function::FunctionCompiler,
    types::{
        FloatWidth,
        IntWidth,
        ScalarType,
        Signedness,
    },
    value::{
        HandlePointer,
        PointerValue,
        ScalarValue,
        Value,
        VectorValue,
    },
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
        let image: PointerValue = self.image.compile_expression(compiler)?.try_into()?;
        let image: HandlePointer = image.try_into()?;

        let sampler: PointerValue = self.sampler.compile_expression(compiler)?.try_into()?;
        let sampler: HandlePointer = sampler.try_into()?;

        let coordinate: VectorValue = self.coordinate.compile_expression(compiler)?.try_into()?;

        let array_index = self
            .array_index
            .map(|array_index| {
                let array_index: ScalarValue =
                    array_index.compile_expression(compiler)?.try_into()?;
                assert_eq!(
                    array_index.ty,
                    ScalarType::Int(Signedness::Unsigned, IntWidth::I32)
                );
                Ok::<ScalarValue, Error>(array_index)
            })
            .transpose()?;

        let offset = self
            .offset
            .map(|offset| {
                let offset: ScalarValue = offset.compile_expression(compiler)?.try_into()?;
                assert_eq!(
                    offset.ty,
                    ScalarType::Int(Signedness::Unsigned, IntWidth::I32)
                );
                Ok::<ScalarValue, Error>(offset)
            })
            .transpose()?;

        let depth_ref = self
            .offset
            .map(|depth_ref| {
                let depth_ref: ScalarValue = depth_ref.compile_expression(compiler)?.try_into()?;
                assert_eq!(depth_ref.ty, ScalarType::Float(FloatWidth::F32));
                Ok::<ScalarValue, Error>(depth_ref)
            })
            .transpose()?;

        let texel = compiler.runtime_context.image_sample(
            compiler.context,
            &mut compiler.function_builder,
            compiler.abort_block,
            image,
            sampler,
            self.gather,
            coordinate,
            array_index,
            offset,
            self.level,
            depth_ref,
            self.clamp_to_edge,
        )?;

        Ok(texel.into())
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
