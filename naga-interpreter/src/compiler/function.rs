use std::{
    fmt::Debug,
    ops::Range,
};

use cranelift_codegen::{
    entity::EntityRef,
    ir::{
        self,
        FuncRef,
        InstBuilder as _,
        ValueLabel,
    },
};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{
    FuncId,
    Module,
};

use crate::{
    compiler::{
        Error,
        compiler::{
            Context,
            FuncBuilderExt,
        },
        expression::{
            CompileExpression,
            Expression,
        },
        simd::SimdImmediates,
        types::{
            PointerType,
            Type,
        },
        util::alignment_log2,
        value::{
            AsIrValue,
            AsIrValues,
            PointerValue,
            ScalarValue,
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
    pub function_ref: FuncRef,
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
        })
    }

    /// Emits code that initializes all local variables.
    ///
    /// This must be called before the function body is compiled.
    pub fn initialize_local_variables(&mut self) -> Result<(), Error> {
        for (handle, variable) in self.function.local_variables.iter() {
            if let Some(init) = variable.init {
                let variable = self.local_variables[handle];
                let value = self.compile_expression(init)?;
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

    /// Compile a [`naga::Block`]
    pub fn compile_block(&mut self, naga_block: &naga::ir::Block) -> Result<(), Error> {
        for (statement, span) in naga_block.span_iter() {
            self.set_source_span(*span);
            self.compile_statement(statement)?;
        }
        Ok(())
    }

    /// Compile a [`naga::Statement`]
    pub fn compile_statement(&mut self, statement: &naga::Statement) -> Result<(), Error> {
        #![allow(unused_variables)]
        use naga::Statement::*;

        match statement {
            Emit(range) => {
                self.compile_emit(range.clone())?;
            }
            Block(naga_block) => {
                // I don't think we actually have to emit a block in cranelift IR.
                // We only have to emit blocks, if we want to jump to them from multiple other
                // blocks, or as an entry point for functions.

                /*let current_cl_block = self.function_builder.current_block().unwrap();
                let new_cl_block = self.function_builder.create_block();

                self.function_builder.switch_to_block(new_cl_block);
                self.compile_block(naga_block);
                self.function_builder.switch_to_block(current_cl_block);

                self.function_builder.ins().jump(new_cl_block, []);
                self.function_builder.seal_block(new_cl_block);*/

                self.compile_block(naga_block)?;
            }
            If {
                condition,
                accept,
                reject,
            } => self.compile_if(*condition, accept, reject)?,
            Switch { selector, cases } => todo!(),
            Loop {
                body,
                continuing,
                break_if,
            } => todo!(),
            Break => todo!(),
            Continue => todo!(),
            Return { value } => {
                self.compile_return(*value)?;
            }
            Kill => todo!(),
            ControlBarrier(barrier) => todo!(),
            MemoryBarrier(barrier) => todo!(),
            Store { pointer, value } => self.compile_store(*pointer, *value)?,
            ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => todo!(),
            Atomic {
                pointer,
                fun,
                value,
                result,
            } => todo!(),
            ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => todo!(),
            WorkGroupUniformLoad { pointer, result } => todo!(),
            Call {
                function,
                arguments,
                result,
            } => self.compile_call(*function, &arguments, *result)?,
            RayQuery { query, fun } => todo!(),
            SubgroupBallot { result, predicate } => todo!(),
            SubgroupGather {
                mode,
                argument,
                result,
            } => todo!(),
            SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => todo!(),
        }

        Ok(())
    }

    pub fn compile_expression(
        &mut self,
        handle: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        let value = if let Some(value) = self.emitted_expression.get(handle) {
            value.clone()
        }
        else {
            let expression = &self.function.expressions[handle];
            // todo: do that here or in the constructor?
            let expression = Expression::from_naga(&self.context.types, expression);

            let span = &self.function.expressions.get_span(handle);
            self.set_source_span(*span);

            let value = expression.compile_expression(self)?;

            if self.context.config.collect_debug_info
                && self.function.named_expressions.contains_key(&handle)
            {
                value.as_ir_values().for_each(|ir_value| {
                    self.function_builder
                        .set_val_label(ir_value, ValueLabel::new(handle.index()));
                });
            }

            self.emitted_expression.insert(handle, value.clone());

            value
        };

        Ok(value)
    }

    pub fn compile_emit(
        &mut self,
        expressions: naga::Range<naga::Expression>,
    ) -> Result<(), Error> {
        for expression in expressions {
            self.compile_expression(expression)?;
        }
        Ok(())
    }

    pub fn compile_call(
        &mut self,
        function: naga::Handle<naga::Function>,
        arguments: &[naga::Handle<naga::Expression>],
        result: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let argument_values = arguments
            .iter()
            .map(|argument| self.compile_expression(*argument))
            .collect::<Result<Vec<_>, Error>>()?;

        // todo: error would be nicer
        let imported_function = self
            .imported_functions
            .get(function)
            .unwrap_or_else(|| panic!("Function not imported: {function:?}"));

        let result_value = self.function_builder.call_(
            self.context,
            imported_function.function_ref,
            argument_values,
            imported_function.declaration.return_type,
        );

        if let Some(result) = result {
            if let Some(result_value) = result_value {
                self.emitted_expression.insert(result, result_value);
            }
            else {
                panic!("Expected function to return a value");
            }
        }

        Ok(())
    }

    pub fn compile_return(
        &mut self,
        expression: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let mut return_values = vec![];

        if let Some(expression) = expression {
            let value = self.compile_expression(expression)?;
            return_values.extend(value.as_ir_values());
        }

        self.function_builder.ins().return_(&return_values);

        // fixme: return ControlFlow to stop compiling this block. this is a bit tricky
        // because we also return Results for now we'll just switch to a new
        // block for the rest. this block will not be jumped to, but we still do the
        // work compiling it.
        let void_block = self.function_builder.create_block();
        self.function_builder.seal_block(void_block);
        self.function_builder.switch_to_block(void_block);
        self.function_builder.set_cold_block(void_block); // very cold lol

        Ok(())
    }

    pub fn compile_store(
        &mut self,
        pointer: naga::Handle<naga::Expression>,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<(), Error> {
        let pointer: PointerValue = self.compile_expression(pointer)?.try_into()?;
        let value: Value = self.compile_expression(expression)?;
        pointer.deref_store(self.context, &mut self.function_builder, &value)
    }

    pub fn compile_if(
        &mut self,
        condition: naga::Handle<naga::Expression>,
        accept: &naga::Block,
        reject: &naga::Block,
    ) -> Result<(), Error> {
        let condition_value: ScalarValue = self.compile_expression(condition)?.try_into()?;
        let condition_value = condition_value.as_ir_value();

        let accept_block = self.function_builder.create_block();
        let reject_block = self.function_builder.create_block();
        let continue_block = self.function_builder.create_block();

        self.function_builder
            .ins()
            .brif(condition_value, accept_block, [], reject_block, []);

        self.function_builder.seal_block(accept_block);
        self.function_builder.seal_block(reject_block);

        self.function_builder.switch_to_block(accept_block);
        self.compile_block(accept)?;
        self.function_builder.ins().jump(continue_block, []);

        self.function_builder.switch_to_block(reject_block);
        self.compile_block(reject)?;
        self.function_builder.ins().jump(continue_block, []);

        self.function_builder.seal_block(continue_block);
        self.function_builder.switch_to_block(continue_block);

        Ok(())
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
    function_compiler.compile_block(&function.body)?;
    function_compiler.finish();

    output.define_function(declaration.function_id, cl_context)?;

    Ok(())
}
