//! Module compiler
//!
//! [`Compiler`] compiles a [`naga::Module`] into a
//! [`cranelift_module::Module`].
//!
//! # Example
//!
//! This JIT-compiles a WGSL module into a set of entry points that can be
//! called from Rust.
//!
//! ```
//! # use naga_cranelift::{compiler::Compiler, system_isa};
//! # use cranelift_jit::{JITModule, JITBuilder};
//! # use cranelift_codegen::settings::Configurable;
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # let source = r#"
//! # @vertex
//! # fn main() -> @builtin(position) vec4f {
//! #   return vec4f();
//! # }
//! # "#;
//! // Parse the naga::Module
//! let module = naga::front::wgsl::parse_str(&source)?;
//! let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
//! let info = validator.validate(&module)?;
//!
//! // Create a JIT module output
//! let isa = system_isa()?;
//! let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
//! let mut jit_module = JITModule::new(jit_builder);
//!
//! let mut compiler = Compiler::new(&module, &info, &mut jit_module, Default::default())?;
//! compiler.declare_all_functions()?;
//! compiler.compile_all_functions()?;
//! let entry_points = compiler.compile_all_entry_points()?;
//! #
//! # Ok(())
//! # }
//! ```

use cranelift_codegen::{
    ir::{
        self,
        InstBuilder as _,
    },
    isa,
};
use cranelift_frontend::{
    FunctionBuilder,
    FunctionBuilderContext,
};
use cranelift_module::{
    Linkage,
    Module,
};

use crate::{
    Error,
    bindings::{
        InterStageLayout,
        collect_user_defined_inter_stage_layout_from_function_arguments,
        collect_user_defined_inter_stage_layout_from_function_result,
    },
    constant::{
        ConstantValue,
        WriteConstant,
    },
    expression::{
        ConstantExpression,
        EvaluateExpression,
    },
    function::{
        ABORT_CODE_TYPE,
        FunctionArgument,
        FunctionDeclaration,
        FunctionResult,
        compile_function,
    },
    product::CompiledEntryPoint,
    runtime::{
        RuntimeContextValue,
        RuntimeEntryPointBuilder,
    },
    simd::SimdContext,
    types::{
        AsIrTypes,
        Type,
    },
    util::{
        CoArena,
        SparseCoArena,
        alignment_log2,
    },
    variable::{
        GlobalVariable,
        GlobalVariableInner,
        GlobalVariablesLayouter,
        PrivateMemoryLayout,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Config {
    /// Calling convention used for internal function
    ///
    /// This calling convention will be used for all functions. Technically the
    /// shader entry points will have this calling convention too, but the
    /// function returned as entry point from the compiler is another function
    /// that sets everything up. This outer entry point will have the default
    /// calling convention of the target.
    ///
    /// If this is not set, the default calling convention of the target will be
    /// used.
    pub calling_convention: Option<isa::CallConv>,

    pub collect_debug_info: bool,
}

/// Immutable data shared during compilation of a [`naga::Module`].
#[derive(derive_more::Debug)]
pub struct Context<'source> {
    /// Compiler config
    pub config: Config,

    /// The [naga source module](naga::Module)
    pub source: &'source naga::Module,

    /// [`ModuleInfo`](naga::valid::ModuleInfo) returned by validation.
    pub info: &'source naga::valid::ModuleInfo,

    /// Used for getting size and alignment of naga's types.
    pub layouter: naga::proc::Layouter,

    /// Target ISA information that we need during compilation.
    ///
    /// This contains the default calling convention and pointer type.
    #[debug(skip)]
    pub target_config: isa::TargetFrontendConfig,

    /// Information on how we can use SIMD for vectors and matrices.
    pub simd_context: SimdContext,

    /// Maps naga's types to our types
    pub types: CoArena<naga::Type, Type>,
}

impl<'source> Context<'source> {
    pub fn new(
        source: &'source naga::Module,
        info: &'source naga::valid::ModuleInfo,
        isa: &dyn isa::TargetIsa,
        config: Config,
    ) -> Result<Self, Error> {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(source.to_ctx())?;

        let target_config = isa.frontend_config();
        let simd_context = SimdContext::new(isa);
        tracing::debug!(?simd_context);

        let types = CoArena::try_from_unique_arena(&source.types, |handle, _ty| {
            Type::from_naga(&source, handle)
        })?;

        Ok(Self {
            source,
            info,
            layouter,
            target_config,
            simd_context,
            types,
            config,
        })
    }

    pub fn internal_calling_convention(&self) -> isa::CallConv {
        self.config
            .calling_convention
            .unwrap_or(self.target_config.default_call_conv)
    }

    pub fn pointer_type(&self) -> ir::Type {
        self.target_config.pointer_type()
    }
}

/// Compiler backend
///
/// This compiles [naga IR](naga::ir) to [cranelift IR](ir). The compiler is
/// generic over the [output module](Module).
#[derive(derive_more::Debug)]
pub struct Compiler<'source, Output> {
    context: Context<'source>,

    /// Contains reusable state that is used by cranelift for compilation
    #[debug(skip)]
    cl_context: cranelift_codegen::Context,

    /// This is used by cranelift to compile functions and can be reused
    #[debug(skip)]
    fb_context: FunctionBuilderContext,

    /// Contains signatures and function IDs of functions declared by the shader
    /// module
    function_declarations: SparseCoArena<naga::Function, FunctionDeclaration>,

    #[debug(skip)]
    output: Output,

    global_expressions: CoArena<naga::Expression, ConstantValue>,
    global_variables: SparseCoArena<naga::GlobalVariable, GlobalVariable>,
    private_memory_stack_slot_data: Option<ir::StackSlotData>,
}

impl<'source, Output> Compiler<'source, Output>
where
    Output: Module,
{
    /// Create a new compiler.
    pub fn new(
        source: &'source naga::Module,
        // note: we don't use this at the moment. but you can only get this by validating a module,
        // so we know that `source` is valid
        info: &'source naga::valid::ModuleInfo,
        output: Output,
        config: Config,
    ) -> Result<Self, Error> {
        let isa = output.isa();
        tracing::debug!(target = %isa.triple());

        let context = Context::new(source, info, isa, config)?;

        let global_expressions =
            CoArena::try_from_arena(&source.global_expressions, |_handle, expression| {
                let expression = ConstantExpression::try_from(expression.clone())?;
                expression.evaluate_expression(&context)
            })?;

        Ok(Self {
            context,
            cl_context: output.make_context(),
            fb_context: FunctionBuilderContext::new(),
            function_declarations: Default::default(),
            output,
            global_expressions,
            global_variables: Default::default(),
            private_memory_stack_slot_data: None,
        })
    }
}

impl<'source, Output> Compiler<'source, Output>
where
    Output: Module,
{
    pub fn declare_all_functions(&mut self) -> Result<(), Error> {
        for (handle, function) in self.context.source.functions.iter() {
            let declaration = self.declare_function(function)?;
            self.function_declarations.insert(handle, declaration);
        }

        Ok(())
    }

    pub fn compile_all_functions(&mut self) -> Result<(), Error> {
        for (handle, function) in self.context.source.functions.iter() {
            if let Some(declaration) = self.function_declarations.get(handle) {
                compile_function(
                    &self.context,
                    &mut self.cl_context,
                    &mut self.fb_context,
                    &self.function_declarations,
                    &mut self.output,
                    function,
                    declaration,
                    &self.global_variables,
                )?;
            }
        }

        Ok(())
    }

    pub fn compile_all_entry_points(&mut self) -> Result<Vec<CompiledEntryPoint>, Error> {
        self.context
            .source
            .entry_points
            .iter()
            .map(|entry_point| self.compile_entry_point(entry_point))
            .collect()
    }

    pub fn compile_function(
        &mut self,
        function: &'source naga::Function,
        declaration: &FunctionDeclaration,
    ) -> Result<(), Error> {
        compile_function(
            &self.context,
            &mut self.cl_context,
            &mut self.fb_context,
            &self.function_declarations,
            &mut self.output,
            function,
            declaration,
            &self.global_variables,
        )
    }

    pub fn declare_function(
        &mut self,
        function: &'source naga::Function,
    ) -> Result<FunctionDeclaration, Error> {
        let mut signature = ir::Signature::new(self.context.internal_calling_convention());

        // functions only return an abort code
        signature.returns.push(ir::AbiParam::new(ABORT_CODE_TYPE));

        // implicit first argument is the runtime context pointer
        signature
            .params
            .push(ir::AbiParam::new(self.context.pointer_type()));

        // return values
        let return_type = function.result.as_ref().map(|result| {
            let return_type = self.context.types[result.ty];

            signature
                .params
                .push(ir::AbiParam::new(self.context.pointer_type()));

            let type_layout = self.context.layouter[result.ty];
            let stack_slot_data = ir::StackSlotData {
                kind: ir::StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment_log2(type_layout.alignment),
                key: None,
            };

            FunctionResult {
                ty: return_type,
                stack_slot_data,
            }
        });

        // arguments
        let mut arguments = Vec::with_capacity(function.arguments.len());
        for argument in &function.arguments {
            let start = signature.params.len();
            signature.params.extend(
                self.context.types[argument.ty]
                    .as_ir_types(&self.context)
                    .map(ir::AbiParam::new),
            );
            let end = signature.params.len();

            let ty = self.context.types[argument.ty];

            arguments.push(FunctionArgument {
                block_inputs: start..end,
                ty,
            })
        }

        let function_id = if let Some(name) = &function.name {
            self.output
                .declare_function(&name, Linkage::Local, &signature)?
        }
        else {
            self.output.declare_anonymous_function(&signature)?
        };

        let declaration = FunctionDeclaration {
            function_id,
            signature,
            arguments,
            return_type,
        };

        Ok(declaration)
    }

    pub fn compile_entry_point(
        &mut self,
        entry_point: &'source naga::EntryPoint,
    ) -> Result<CompiledEntryPoint, Error> {
        // compile entry point function
        let entry_point_declaration = self.declare_function(&entry_point.function)?;
        self.compile_function(&entry_point.function, &entry_point_declaration)?;

        // build shim
        self.output.clear_context(&mut self.cl_context);

        let main_function_ref = self.output.declare_func_in_func(
            entry_point_declaration.function_id,
            &mut self.cl_context.func,
        );

        self.cl_context
            .func
            .signature
            .params
            .push(ir::AbiParam::new(self.context.pointer_type()));
        self.cl_context
            .func
            .signature
            .returns
            .push(ir::AbiParam::new(ABORT_CODE_TYPE));

        let mut function_builder =
            FunctionBuilder::new(&mut self.cl_context.func, &mut self.fb_context);

        let entry_block = function_builder.create_block();
        let exit_block = function_builder.create_block();

        // compile entry block first. this determines that it's actually the entry block
        // for the function
        function_builder.switch_to_block(entry_block);
        function_builder.append_block_params_for_function_params(entry_block);

        let runtime_context = {
            let block_params = function_builder.block_params(entry_block);
            // note: the order of these arguments must be synchronized with the call to the
            // compiled code in
            // [`EntryPoint::function`](super::product::EntryPoint::function).
            let runtime_pointer = block_params[0];
            RuntimeContextValue::new(&self.context, &mut function_builder, runtime_pointer)
        };

        function_builder.seal_block(entry_block);

        let mut shim_builder =
            RuntimeEntryPointBuilder::new(&self.context, function_builder, runtime_context);

        // allocates space for global variables on stack. then calls into the runtime to
        // initialize that memory. then stashes the pointer to that stack slot into the
        // runtime context struct.
        //
        // don't forget this! otherwise the global variables pointers will be dangling
        // and the pointer in the context will be null
        //
        // todo: I think we should move the global variables / private memory stuff from
        // context into the state and set everything up in one place so that we can't
        // mess this up (draumatized from my first segfault lol).
        if let Some(private_memory_stack_slot_data) = self.private_memory_stack_slot_data.clone() {
            shim_builder
                .compile_private_data_initialization(private_memory_stack_slot_data, exit_block)?;
        }

        let mut argument_values = Vec::with_capacity(entry_point.function.arguments.len() + 2);
        argument_values.push(runtime_context.pointer);

        let mut output_layout = vec![];
        let mut result_pointer = None;

        if let Some(result) = &entry_point.function.result {
            let (pointer, layout) = shim_builder.allocate_result(result)?;
            result_pointer = Some(pointer);
            output_layout = layout;
            argument_values.push(pointer.pointer.value);
        }

        let input_layout = shim_builder.load_arguments(
            &entry_point.function.arguments,
            &mut argument_values,
            exit_block,
        )?;

        let inst = shim_builder
            .function_builder
            .ins()
            .call(main_function_ref, &argument_values);
        let abort_code = shim_builder.function_builder.inst_results(inst)[0];
        let continue_block = shim_builder.function_builder.create_block();
        shim_builder.function_builder.ins().brif(
            abort_code,
            exit_block,
            [&ir::BlockArg::Value(abort_code)],
            continue_block,
            [],
        );
        shim_builder.function_builder.seal_block(continue_block);
        shim_builder
            .function_builder
            .switch_to_block(continue_block);

        if let Some(pointer) = result_pointer {
            shim_builder.pass_result_to_runtime(pointer, exit_block);
        }

        let mut function_builder = shim_builder.function_builder;
        let abort_code = function_builder.ins().iconst(ABORT_CODE_TYPE, 0);
        function_builder
            .ins()
            .jump(exit_block, [&ir::BlockArg::Value(abort_code)]);
        function_builder.seal_block(exit_block);

        // create exit block
        function_builder.append_block_param(exit_block, ABORT_CODE_TYPE);
        function_builder.switch_to_block(exit_block);
        let abort_code = function_builder.block_params(exit_block)[0];
        function_builder.ins().return_(&[abort_code]);

        function_builder.finalize();

        let shim_function = self
            .output
            .declare_anonymous_function(&self.cl_context.func.signature)?;

        self.output
            .define_function(shim_function, &mut self.cl_context)?;

        let inter_stage_layout = match entry_point.stage {
            naga::ShaderStage::Vertex => {
                Some(InterStageLayout::Vertex {
                    output: collect_user_defined_inter_stage_layout_from_function_result(
                        &self.context.source,
                        &self.context.layouter,
                        &entry_point.function.result,
                    ),
                })
            }
            naga::ShaderStage::Fragment => {
                Some(InterStageLayout::Fragment {
                    input: collect_user_defined_inter_stage_layout_from_function_arguments(
                        &self.context.source,
                        &self.context.layouter,
                        &entry_point.function.arguments,
                    ),
                })
            }
            _ => None,
        };

        Ok(CompiledEntryPoint {
            name: entry_point.name.clone(),
            stage: entry_point.stage,
            inter_stage_layout,
            early_depth_test: entry_point.early_depth_test,
            function_id: shim_function,
            input_layout,
            output_layout,
        })
    }

    pub fn layout_private_memory(&mut self) -> PrivateMemoryLayout {
        let mut layouter = GlobalVariablesLayouter::new(&self.context.layouter);

        assert!(self.global_variables.is_empty());

        for (handle, global_variable) in self.context.source.global_variables.iter() {
            if let Some(binding) = global_variable.binding {
                self.global_variables.insert(
                    handle,
                    GlobalVariable {
                        ty: global_variable.ty,
                        address_space: global_variable.space,
                        inner: GlobalVariableInner::Resource { binding },
                    },
                );
            }
            else {
                self.global_variables.insert(
                    handle,
                    GlobalVariable {
                        ty: global_variable.ty,
                        address_space: global_variable.space,
                        // offset and len will be patched later when we finalize the layout
                        inner: GlobalVariableInner::Memory { offset: 0, len: 0 },
                    },
                );

                layouter.push(handle, global_variable, |expression, data| {
                    let value = &self.global_expressions[expression];
                    value.write_into(&self.context, data);
                });
            }
        }

        let layout = layouter.finish();

        let mut no_private_memory_allocated = true;
        for (handle, entry) in layout.iter() {
            let global_variable = self.global_variables.get_mut(handle).unwrap();
            match &mut global_variable.inner {
                GlobalVariableInner::Memory { offset, len } => {
                    *offset = entry.offset;
                    *len = entry.len;
                    no_private_memory_allocated = false;
                }
                GlobalVariableInner::Resource { binding: _ } => unreachable!(),
            }
        }

        if !self.global_variables.is_empty() {
            self.private_memory_stack_slot_data = layout.private_memory_layout.stack_slot_data();

            if self.private_memory_stack_slot_data.is_none() {
                assert!(
                    no_private_memory_allocated,
                    "memory variables emitted, but no memory will be allocated"
                );
            }
        }

        layout.private_memory_layout
    }
}
