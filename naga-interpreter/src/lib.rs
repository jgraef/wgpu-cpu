#![allow(unused_variables)]

pub mod bindings;
mod interpreter;
pub mod memory;
mod module;
#[cfg(test)]
mod tests;
mod util;

pub use interpreter::Interpreter;
pub use module::{
    EntryPointIndex,
    Error as ModuleError,
    ShaderModule,
};
