use crate::compiler::{
    Error,
    function::FunctionCompiler,
    statement::{
        BlockStatement,
        CompileStatement,
    },
};

#[derive(Clone, Debug)]
pub struct SwitchStatement {
    pub selector: naga::Handle<naga::Expression>,
    pub cases: Vec<SwitchCase>,
}

impl CompileStatement for SwitchStatement {
    fn compile_statement(&self, compiler: &mut FunctionCompiler) -> Result<(), Error> {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct SwitchCase {
    /// Value, upon which the case is considered true.
    pub value: naga::SwitchValue,
    /// Body of the case.
    pub body: BlockStatement,
    /// If true, the control flow continues to the next case in the list,
    /// or default.
    pub fall_through: bool,
}

impl From<&naga::SwitchCase> for SwitchCase {
    fn from(case: &naga::SwitchCase) -> Self {
        Self {
            value: case.value,
            body: (&case.body).into(),
            fall_through: case.fall_through,
        }
    }
}
