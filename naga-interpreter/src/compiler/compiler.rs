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
//! # use naga_interpreter::compiler::{compiler::Compiler, system_isa};
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
        AbiParam,
        InstBuilder,
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
    compiler::{
        Error,
        expression::EvaluateExpression,
        function::{
            FunctionArgument,
            FunctionDeclaration,
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
        value::{
            AsIrValues,
            FromIrValues,
            Value,
        },
    },
    entry_point::EntryPoints,
    util::{
        CoArena,
        SparseCoArena,
    },
};

const SHIM_FUNCTION_NAME: &str = "__naga_rt0";

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

        //let global_expressions =
        //    CoArena::try_from_arena(&source.global_expressions, |handle, expression|
        // todo!())?;

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
    pub function_declarations: SparseCoArena<naga::Function, FunctionDeclaration>,

    #[debug(skip)]
    output: Output,
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

        Ok(Self {
            context,
            cl_context: output.make_context(),
            fb_context: FunctionBuilderContext::new(),
            function_declarations: Default::default(),
            output,
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
                )?;
            }
        }

        Ok(())
    }

    pub fn compile_all_entry_points(&mut self) -> Result<EntryPoints<CompiledEntryPoint>, Error> {
        let mut entry_points = EntryPoints::with_capacity(self.context.source.entry_points.len());

        for entry_point in &self.context.source.entry_points {
            let entry_point_data = self.compile_entry_point(entry_point)?;
            entry_points.push(entry_point, entry_point_data);
        }

        Ok(entry_points)
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
        )
    }

    pub fn declare_function(
        &mut self,
        function: &'source naga::Function,
    ) -> Result<FunctionDeclaration, Error> {
        //let mut signature = self.output.make_signature();
        let mut signature = ir::Signature::new(self.context.internal_calling_convention());

        // return values
        let return_type = function.result.as_ref().map(|result| {
            let return_type = self.context.types[result.ty];
            signature
                .returns
                .extend(return_type.as_ir_types(&self.context).map(AbiParam::new));
            return_type
        });

        // arguments
        let mut arguments = Vec::with_capacity(function.arguments.len());
        for argument in &function.arguments {
            let start = signature.params.len();
            signature.params.extend(
                self.context.types[argument.ty]
                    .as_ir_types(&self.context)
                    .map(AbiParam::new),
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
            .push(AbiParam::new(self.context.pointer_type()));
        self.cl_context
            .func
            .signature
            .params
            .push(AbiParam::new(self.context.pointer_type()));

        let mut function_builder =
            FunctionBuilder::new(&mut self.cl_context.func, &mut self.fb_context);

        let entry_block = function_builder.create_block();
        let panic_block = function_builder.create_block();

        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);

        let runtime = {
            let block_params = function_builder.block_params(entry_block);
            // note: the order of these arguments must be synchronized with the call to the
            // compiled code in `module.rs`.
            let vtable_pointer = block_params[0];
            let data_pointer = block_params[1];
            RuntimeContextValue::new(
                &self.context,
                &mut function_builder,
                vtable_pointer,
                data_pointer,
            )
        };

        function_builder.seal_block(entry_block);

        let mut shim_builder =
            RuntimeEntryPointBuilder::new(&self.context, function_builder, runtime, panic_block);

        let (arguments, input_layout) =
            shim_builder.compile_arguments_shim(&entry_point.function.arguments)?;

        let return_type = entry_point
            .function
            .result
            .as_ref()
            .map(|function_result| Type::from_naga(&self.context.source, function_result.ty))
            .transpose()?;

        let return_value = shim_builder.function_builder.call_(
            &self.context,
            main_function_ref,
            arguments,
            return_type,
        );

        let output_layout = entry_point
            .function
            .result
            .as_ref()
            .map(|result| shim_builder.compile_result_shim(result, return_value.unwrap()))
            .transpose()?
            .unwrap_or_default();

        let mut function_builder = shim_builder.function_builder;

        // return from the block we're in
        function_builder.ins().return_(&[]);

        // the panic block will just return
        // we don't need to return any error code since errors are written into a field
        // in the runtime data.
        function_builder.switch_to_block(panic_block);
        function_builder.ins().return_(&[]);
        function_builder.seal_block(panic_block);

        function_builder.finalize();

        //println!("{:#?}", self.state.cl_context.func);

        let shim_function = self.output.declare_function(
            SHIM_FUNCTION_NAME,
            Linkage::Local,
            &self.cl_context.func.signature,
        )?;

        self.output
            .define_function(shim_function, &mut self.cl_context)?;

        Ok(CompiledEntryPoint {
            function_id: shim_function,
            input_layout,
            output_layout,
        })
    }

    // wip
    #[allow(unused)]
    fn compile_global_variables(&mut self) -> Result<(), Error> {
        //let mut global_data = vec![];
        let mut offset = 0;

        let global_variables = CoArena::try_from_arena(
            &self.context.source.global_variables,
            |handle, global_variable| {
                let ty = self.context.types[global_variable.ty];

                if let Some(init) = global_variable.init {
                    assert!(global_variable.binding.is_none());
                    let init = init.evaluate_expression(&self.context)?;
                    let type_layout = self.context.layouter[global_variable.ty];

                    dbg!(init);
                    dbg!(type_layout);
                    todo!("write constant value into global data");
                }

                if let Some(binding) = global_variable.binding {
                    assert!(global_variable.init.is_none());
                }

                Ok::<_, Error>(GlobalVariable {
                    offset,
                    ty,
                    address_space: global_variable.space,
                })
            },
        )?;

        // todo: store coarena for function compiler
        todo!("return struct containing global data and layout");
    }
}

// better name?
// todo
#[allow(unused)]
#[derive(Clone, Debug)]
pub struct PrivateMemory {
    data: Vec<u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalVariable {
    pub address_space: naga::AddressSpace,
    pub offset: i32,
    pub ty: Type,
}

pub trait FuncBuilderExt {
    fn call_(
        &mut self,
        context: &Context,
        func_ref: ir::FuncRef,
        arguments: impl IntoIterator<Item = Value>,
        return_type: Option<Type>,
    ) -> Option<Value>;

    fn switch_to_void_block(&mut self);
}

impl<'a> FuncBuilderExt for FunctionBuilder<'a> {
    fn call_(
        &mut self,
        context: &Context,
        func_ref: ir::FuncRef,
        arguments: impl IntoIterator<Item = Value>,
        return_type: Option<Type>,
    ) -> Option<Value> {
        let mut values = vec![];
        for argument in arguments {
            values.extend(argument.as_ir_values());
        }

        let inst = self.ins().call(func_ref, &values);

        return_type.map(|return_type| {
            Value::from_ir_values_iter(
                context,
                return_type,
                self.inst_results(inst).iter().copied(),
            )
        })
    }

    fn switch_to_void_block(&mut self) {
        // todo: return early from compiling a block with ControlFlow instead.
        let void_block = self.create_block();
        self.seal_block(void_block);
        self.set_cold_block(void_block); // very cold lol
        self.switch_to_block(void_block);
    }
}
