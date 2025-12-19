pub mod bindings;
pub mod interpreter;
pub mod memory;
mod module;
#[cfg(test)]
mod tests;
pub mod variable;

pub use interpreter::Interpreter;
pub use module::{
    Error,
    InterpretedModule,
};

use crate::{
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
    },
    memory::NullMemory,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct InterpreterBackend;

impl crate::backend::Backend for InterpreterBackend {
    type Module = InterpretedModule;
    type Error = Error;

    fn create_module(
        &self,
        ir: naga::Module,
        info: naga::valid::ModuleInfo,
    ) -> Result<Self::Module, Self::Error> {
        InterpretedModule::new(ir, info)
    }
}

impl crate::backend::Module for InterpretedModule {
    fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        self.find_entry_point(name, stage)
    }

    fn run_entry_point<I, O>(&self, entry_point: EntryPointIndex, input: I, output: O)
    where
        I: crate::bindings::ShaderInput,
        O: crate::bindings::ShaderOutput,
    {
        let mut interpreter = Interpreter::new(self, NullMemory, entry_point);
        interpreter.run_entry_point(input, output);
    }

    fn inter_stage_layout(
        &self,
        entry_point: EntryPointIndex,
    ) -> Option<&crate::entry_point::InterStageLayout> {
        self.inner.entry_points[entry_point]
            .inter_stage_layout
            .as_ref()
    }

    fn early_depth_test(&self, entry_point: EntryPointIndex) -> Option<naga::EarlyDepthTest> {
        self.inner.entry_points[entry_point].early_depth_test
    }
}
