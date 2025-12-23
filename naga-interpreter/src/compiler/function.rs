use std::{
    fmt::Debug,
    ops::Range,
};

use cranelift_codegen::ir::{
    self,
};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{
    FuncId,
    Module,
};

use crate::{
    compiler::{
        Error,
        compiler::Context,
        expression::CompileExpression,
        simd::SimdImmediates,
        statement::{
            BlockStatement,
            CompileStatement,
        },
        types::{
            PointerType,
            Type,
        },
        util::alignment_log2,
        value::{
            StackLocation,
            Store,
            Value,
        },
    },
    util::{
        CoArena,
        SparseCoArena,
    },
};

#[derive(Clone, Debug)]
pub struct FunctionArgument {
    /// Range of block inputs that this argument corresponds to
    pub block_inputs: Range<usize>,

    /// Type of argument
    pub ty: Type,
}

#[derive(Clone, Copy, Debug)]
pub struct LocalVariable<'source> {
    pub name: Option<&'source str>,
    pub ty: Type,
    pub pointer_type: PointerType,
    pub stack_slot: ir::StackSlot,
}

#[derive(Clone, Debug)]
pub struct FunctionDeclaration {
    pub function_id: FuncId,
    pub signature: ir::Signature,
    pub arguments: Vec<FunctionArgument>,
    pub return_type: Option<Type>,
}

#[derive(Clone, Copy, Debug)]
pub struct ImportedFunction<'compiler> {
    pub function_ref: ir::FuncRef,
    pub declaration: &'compiler FunctionDeclaration,
}

#[derive(derive_more::Debug)]
pub struct FunctionCompiler<'source, 'compiler> {
    pub context: &'compiler Context<'source>,
    pub function: &'source naga::Function,
    pub declaration: &'compiler FunctionDeclaration,
    pub entry_block: ir::Block,
    pub local_variables: CoArena<naga::LocalVariable, LocalVariable<'source>>,
    pub simd_immediates: SimdImmediates,
    pub imported_functions: SparseCoArena<naga::Function, ImportedFunction<'compiler>>,
    #[debug(skip)]
    pub function_builder: FunctionBuilder<'compiler>,
    pub emitted_expression: SparseCoArena<naga::Expression, Value>,
    pub source_locations: Vec<naga::Span>,
    pub loop_switch_stack: LoopSwitchStack,
}

impl<'source, 'compiler> FunctionCompiler<'source, 'compiler> {
    pub fn new(
        compiler_context: &'compiler Context<'source>,
        cl_context: &'compiler mut cranelift_codegen::Context,
        fb_context: &'compiler mut cranelift_frontend::FunctionBuilderContext,
        function: &'source naga::Function,
        declaration: &'compiler FunctionDeclaration,
    ) -> Result<Self, Error> {
        cl_context.clear();

        if compiler_context.config.collect_debug_info {
            cl_context.func.dfg.collect_debug_info();
        }

        let simd_immediates = compiler_context.simd_context.simd_immediates();

        cl_context.func.signature = declaration.signature.clone();

        let mut function_builder = FunctionBuilder::new(&mut cl_context.func, fb_context);

        let entry_block = function_builder.create_block();
        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);
        function_builder.seal_block(entry_block);

        // local variables
        let local_variables =
            CoArena::try_from_arena(&function.local_variables, |handle, variable| {
                let type_layout = compiler_context.layouter[variable.ty];
                let stack_slot_key = ir::StackSlotKey::new(handle.index().try_into().unwrap());

                let stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
                    kind: ir::StackSlotKind::ExplicitSlot,
                    size: type_layout.size,
                    align_shift: alignment_log2(type_layout.alignment),
                    key: Some(stack_slot_key),
                });

                let pointer_type =
                    PointerType::from_naga(variable.ty, naga::AddressSpace::Function)?;

                Ok::<_, Error>(LocalVariable {
                    name: variable.name.as_deref(),
                    ty: compiler_context.types[variable.ty],
                    pointer_type,
                    stack_slot,
                })
            })?;

        Ok(Self {
            context: compiler_context,
            function,
            declaration,
            entry_block,
            local_variables,
            simd_immediates,
            imported_functions: Default::default(),
            function_builder,
            emitted_expression: Default::default(),
            source_locations: vec![],
            loop_switch_stack: Default::default(),
        })
    }

    /// Emits code that initializes all local variables.
    ///
    /// This must be called before the function body is compiled.
    pub fn initialize_local_variables(&mut self) -> Result<(), Error> {
        for (handle, variable) in self.function.local_variables.iter() {
            if let Some(init) = variable.init {
                let variable = self.local_variables[handle];
                let value = init.compile_expression(self)?;
                value.store(
                    self.context,
                    &mut self.function_builder,
                    StackLocation::from(variable.stack_slot),
                )?;
            }
        }

        Ok(())
    }

    /// Imports all functions that are called from this functions.
    ///
    /// Do this before compiling the function body. Otherwise compilation of any
    /// function call will fail.
    pub fn import_functions<Output>(
        &mut self,
        function_declarations: &'compiler SparseCoArena<naga::Function, FunctionDeclaration>,
        output: &mut Output,
    ) -> Result<(), Error>
    where
        Output: Module,
    {
        let mut importer = FunctionImporter {
            function_refs: &mut self.imported_functions,
            compiler_context: self.context,
            function_declarations,
            output,
            caller: &mut self.function_builder.func,
        };
        importer.import_functions(&self.function.body)
    }

    /// Finish compilation of the function
    pub fn finish(self) {
        assert!(self.loop_switch_stack.is_empty());
        self.function_builder.finalize();
    }

    /// Set the current source location
    pub fn set_source_span(&mut self, span: naga::Span) {
        if self.context.config.collect_debug_info {
            let id = self
                .source_locations
                .len()
                .try_into()
                .expect("source location id overflow");

            self.source_locations.push(span);
            self.function_builder.set_srcloc(ir::SourceLoc::new(id));
        }
    }
}

/// Helper to call [`Module::declare_func_in_func`] on all functions called from
/// another function.
pub struct FunctionImporter<'source, 'compiler, 'function, Output> {
    pub function_refs: &'function mut SparseCoArena<naga::Function, ImportedFunction<'compiler>>,
    pub compiler_context: &'compiler Context<'source>,
    pub function_declarations: &'compiler SparseCoArena<naga::Function, FunctionDeclaration>,
    pub output: &'function mut Output,
    pub caller: &'function mut ir::Function,
}

impl<'source, 'compiler, 'function, Output> FunctionImporter<'source, 'compiler, 'function, Output>
where
    Output: Module,
{
    pub fn import_functions(&mut self, block: &naga::Block) -> Result<(), Error> {
        use naga::Statement::*;

        for statement in block {
            match statement {
                Block(block) => self.import_functions(block)?,
                If {
                    condition: _,
                    accept,
                    reject,
                } => {
                    self.import_functions(accept)?;
                    self.import_functions(reject)?;
                }
                Switch { selector: _, cases } => {
                    for case in cases {
                        self.import_functions(&case.body)?;
                    }
                }
                Loop {
                    body,
                    continuing,
                    break_if: _,
                } => {
                    self.import_functions(body)?;
                    self.import_functions(continuing)?;
                }
                Call {
                    function,
                    arguments: _,
                    result: _,
                } => {
                    if !self.function_refs.contains(*function) {
                        let declaration =
                            self.function_declarations.get(*function).ok_or_else(|| {
                                let name = self.compiler_context.source.functions[*function]
                                    .name
                                    .clone();
                                Error::UndeclaredFunctionCall {
                                    name,
                                    handle: *function,
                                }
                            })?;

                        let function_ref = self
                            .output
                            .declare_func_in_func(declaration.function_id, self.caller);

                        self.function_refs.insert(
                            *function,
                            ImportedFunction {
                                function_ref,
                                declaration,
                            },
                        );
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }
}

pub fn compile_function<'source, 'compiler, Output>(
    context: &'compiler Context<'source>,
    cl_context: &'compiler mut cranelift_codegen::Context,
    fb_context: &'compiler mut cranelift_frontend::FunctionBuilderContext,
    function_declarations: &SparseCoArena<naga::Function, FunctionDeclaration>,
    output: &'compiler mut Output,
    function: &'source naga::Function,
    declaration: &FunctionDeclaration,
) -> Result<(), Error>
where
    Output: Module,
{
    let mut function_compiler =
        FunctionCompiler::new(context, cl_context, fb_context, function, declaration)?;

    function_compiler.initialize_local_variables()?;
    function_compiler.import_functions(function_declarations, output)?;
    let body = BlockStatement::from(&function.body);
    body.compile_statement(&mut function_compiler)?;
    function_compiler.finish();

    output.define_function(declaration.function_id, cl_context)?;

    Ok(())
}

#[derive(Clone, Debug, Default)]
pub struct LoopSwitchStack {
    pub stack: Vec<LoopSwitchState>,
}

impl LoopSwitchStack {
    pub fn push_loop(&mut self, continuing_block: ir::Block, exit_block: ir::Block) {
        self.stack.push(LoopSwitchState::LoopBody {
            continuing_block,
            exit_block,
        });
    }

    pub fn pop_loop(
        &mut self,
        expected_continuing_block: ir::Block,
        expected_exit_block: ir::Block,
    ) {
        match self.pop() {
            LoopSwitchState::LoopBody {
                continuing_block,
                exit_block,
            } => {
                assert_eq!(continuing_block, expected_continuing_block);
                assert_eq!(exit_block, expected_exit_block);
            }
            state => panic!("unexpected loop switch state: {state:?}"),
        }
    }

    pub fn push_continuing(&mut self) {
        self.stack.push(LoopSwitchState::LoopContinuing);
    }

    pub fn pop_continuing(&mut self) {
        match self.pop() {
            LoopSwitchState::LoopContinuing => {}
            state => panic!("unexpected loop switch state: {state:?}"),
        }
    }

    pub fn push_switch(&mut self, exit_block: ir::Block) {
        self.stack.push(LoopSwitchState::SwitchBody { exit_block });
    }

    pub fn pop_switch(&mut self, expected_exit_block: ir::Block) {
        match self.pop() {
            LoopSwitchState::SwitchBody { exit_block } => {
                assert_eq!(exit_block, expected_exit_block);
            }
            state => panic!("unexpected loop switch state: {state:?}"),
        }
    }

    pub fn pop(&mut self) -> LoopSwitchState {
        self.stack.pop().expect("not in loop")
    }

    pub fn top(&self) -> LoopSwitchState {
        *self.stack.last().expect("not in loop")
    }

    pub fn get_break_block(&self) -> ir::Block {
        match self.top() {
            LoopSwitchState::LoopBody {
                continuing_block: _,
                exit_block,
            }
            | LoopSwitchState::SwitchBody { exit_block } => exit_block,
            state => panic!("unexpected loop switch state: {state:?}"),
        }
    }

    pub fn get_continuing_block(&self) -> ir::Block {
        match self.top() {
            LoopSwitchState::LoopBody {
                continuing_block,
                exit_block: _,
            } => continuing_block,
            state => panic!("unexpected loop switch state: {state:?}"),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum LoopSwitchState {
    LoopBody {
        continuing_block: ir::Block,
        exit_block: ir::Block,
    },
    LoopContinuing,
    SwitchBody {
        exit_block: ir::Block,
    },
}
