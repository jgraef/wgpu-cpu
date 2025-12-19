use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        InterStageLayout,
    },
};

pub trait Backend {
    type Module: Module;
    type Error: std::error::Error;

    fn create_module(
        &self,
        ir: naga::Module,
        info: naga::valid::ModuleInfo,
    ) -> Result<Self::Module, Self::Error>;
}

pub trait Module {
    fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound>;

    fn run_entry_point<I, O>(&self, entry_point: EntryPointIndex, input: I, output: O)
    where
        I: ShaderInput,
        O: ShaderOutput;

    fn inter_stage_layout(&self, entry_point: EntryPointIndex) -> Option<&InterStageLayout>;

    fn early_depth_test(&self, entry_point: EntryPointIndex) -> Option<naga::EarlyDepthTest>;
}
