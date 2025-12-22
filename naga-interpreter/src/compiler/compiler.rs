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
    FuncId,
    Linkage,
    Module,
};

use crate::{
    compiler::{
        Error,
        bindings::ShimBuilder,
        function::{
            FunctionCompiler,
            FunctionName,
        },
        module::CompiledEntryPoint,
        simd::SimdContext,
        types::Type,
        value::{
            AsIrValues,
            FromIrValues,
            Value,
        },
    },
    entry_point::EntryPoints,
    util::CoArena,
};

#[derive(derive_more::Debug)]
pub struct Context<'source> {
    pub source: &'source naga::Module,
    #[allow(unused)]
    pub info: &'source naga::valid::ModuleInfo,
    pub layouter: naga::proc::Layouter,
    #[debug(skip)]
    pub target_config: isa::TargetFrontendConfig,
    pub simd_context: SimdContext,
    pub types: CoArena<naga::Type, Type>,
}

impl<'source> Context<'source> {
    pub fn calling_convention(&self) -> isa::CallConv {
        self.target_config.default_call_conv
    }

    pub fn pointer_type(&self) -> ir::Type {
        self.target_config.pointer_type()
    }
}

#[derive(derive_more::Debug)]
pub struct State {
    #[debug(skip)]
    pub(super) fb_context: FunctionBuilderContext,

    #[debug(skip)]
    pub(super) cl_context: cranelift_codegen::Context,

    next_anonymous_function_id: usize,
}

impl State {
    pub fn anonymous_function_name(&mut self) -> FunctionName {
        let id = self.next_anonymous_function_id;
        self.next_anonymous_function_id += 1;
        FunctionName::Anonymous(id)
    }
}

#[derive(derive_more::Debug)]
pub struct Compiler<'source, Output> {
    context: Context<'source>,
    state: State,

    #[debug(skip)]
    output: Output,
}

impl<'source, Output> Compiler<'source, Output>
where
    Output: Module,
{
    pub fn new(
        source: &'source naga::Module,
        info: &'source naga::valid::ModuleInfo,
        output: Output,
    ) -> Result<Self, Error> {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(source.to_ctx())?;

        let cl_context = output.make_context();
        let target_config = output.target_config();

        let isa = output.isa();
        tracing::debug!(target = %isa.triple());

        let vector_registers = SimdContext::new(isa);
        tracing::debug!(?vector_registers);

        let types = CoArena::try_from_unique_arena(&source.types, |handle, _ty| {
            Type::from_naga(&source, handle)
        })?;

        Ok(Self {
            context: Context {
                source,
                info,
                layouter,
                target_config,
                simd_context: vector_registers,
                types,
            },
            state: State {
                fb_context: FunctionBuilderContext::new(),
                cl_context,
                next_anonymous_function_id: 1,
            },
            output,
        })
    }
}

impl<'source, Output> Compiler<'source, Output>
where
    Output: Module,
{
    pub fn compile_all_entry_points(&mut self) -> Result<EntryPoints<CompiledEntryPoint>, Error> {
        let mut entry_points = EntryPoints::with_capacity(self.context.source.entry_points.len());

        for entry_point in &self.context.source.entry_points {
            let entry_point_data = self.compile_entry_point(entry_point)?;
            entry_points.push(entry_point, entry_point_data);
        }

        Ok(entry_points)
    }

    pub fn compile_entry_point(
        &mut self,
        entry_point: &'source naga::EntryPoint,
    ) -> Result<CompiledEntryPoint, Error> {
        // compile entry point function
        let main_function_id = self.compile_function(&entry_point.function)?;

        // build shim
        self.output.clear_context(&mut self.state.cl_context);

        let main_function_ref = self
            .output
            .declare_func_in_func(main_function_id, &mut self.state.cl_context.func);

        self.state
            .cl_context
            .func
            .signature
            .params
            .push(AbiParam::new(self.context.pointer_type()));
        self.state
            .cl_context
            .func
            .signature
            .params
            .push(AbiParam::new(self.context.pointer_type()));

        let mut function_builder =
            FunctionBuilder::new(&mut self.state.cl_context.func, &mut self.state.fb_context);

        let entry_block = function_builder.create_block();
        let panic_block = function_builder.create_block();

        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);

        let shim_vtable = function_builder.block_params(entry_block)[0];
        let shim_data = function_builder.block_params(entry_block)[1];

        function_builder.seal_block(entry_block);

        let mut shim_builder = ShimBuilder::new(
            &self.context,
            function_builder,
            shim_vtable,
            shim_data,
            panic_block,
        );

        let (arguments, input_layout) =
            shim_builder.compile_arguments_shim(&entry_point.function.arguments)?;

        let return_type = entry_point
            .function
            .result
            .as_ref()
            .map(|function_result| Type::from_naga(&self.context.source, function_result.ty))
            .transpose()?;

        let return_value = shim_builder.function_builder.call_(
            main_function_ref,
            arguments,
            return_type,
            &self.context,
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

        // the panic block will also just return
        function_builder.switch_to_block(panic_block);
        function_builder.ins().return_(&[]);
        function_builder.seal_block(panic_block);

        function_builder.finalize();

        //println!("{:#?}", self.state.cl_context.func);

        let shim_function = self.output.declare_function(
            "__naga_interpreter_shim", // this name is not accuarate anymore, isn't it :D
            Linkage::Local,
            &self.state.cl_context.func.signature,
        )?;

        self.output
            .define_function(shim_function, &mut self.state.cl_context)?;

        Ok(CompiledEntryPoint {
            function_id: shim_function,
            input_layout,
            output_layout,
        })
    }

    pub fn compile_function(&mut self, function: &'source naga::Function) -> Result<FuncId, Error> {
        self.output.clear_context(&mut self.state.cl_context);

        let mut function_compiler =
            FunctionCompiler::new(&self.context, &mut self.state, function)?;

        let function_id = function_compiler.declare(&mut self.output)?;

        function_compiler.initialize_local_variables()?;
        function_compiler.compile_block(&function.body)?;

        function_compiler.finish();

        self.output
            .define_function(function_id, &mut self.state.cl_context)?;

        Ok(function_id)
    }
}

pub trait FuncBuilderExt {
    fn call_(
        &mut self,
        func_ref: ir::FuncRef,
        arguments: impl IntoIterator<Item = Value>,
        return_type: Option<Type>,
        context: &Context,
    ) -> Option<Value>;
}

impl<'a> FuncBuilderExt for FunctionBuilder<'a> {
    fn call_(
        &mut self,
        func_ref: ir::FuncRef,
        arguments: impl IntoIterator<Item = Value>,
        return_type: Option<Type>,
        context: &Context,
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
}
