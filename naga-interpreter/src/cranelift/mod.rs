mod compiler;
mod module;
#[cfg(test)]
mod tests;

pub use compiler::{
    Compiler,
    Error as CompilerError,
};
