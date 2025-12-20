use cranelift_codegen::ir::{
    AbiParam,
    InstBuilder,
    types,
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
        context::Context,
        function::{
            FunctionCompiler,
            FunctionName,
        },
        module::CompiledEntryPoint,
    },
    entry_point::EntryPoints,
    util::typifier_from_function,
};

#[derive(derive_more::Debug)]
pub struct CodegenState {
    #[debug(skip)]
    pub(super) fb_context: FunctionBuilderContext,

    #[debug(skip)]
    pub(super) cl_context: cranelift_codegen::Context,

    next_anonymous_function_id: usize,
}

impl CodegenState {
    pub fn anonymous_function_name(&mut self) -> FunctionName {
        let id = self.next_anonymous_function_id;
        self.next_anonymous_function_id += 1;
        FunctionName::Anonymous(id)
    }
}

#[derive(derive_more::Debug)]
pub struct Compiler<'source, Output> {
    context: Context<'source>,
    state: CodegenState,

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
        println!("target: {}", isa.triple());
        println!(
            "dynamic_vector_bytes i8: {}, i32: {}, f16: {}, f32: {}",
            isa.dynamic_vector_bytes(types::I8),
            isa.dynamic_vector_bytes(types::I32),
            isa.dynamic_vector_bytes(types::F16),
            isa.dynamic_vector_bytes(types::F32)
        );

        Ok(Self {
            context: Context {
                source,
                info,
                layouter,
                target_config,
            },
            state: CodegenState {
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

        let output = {
            let inst = shim_builder
                .function_builder
                .ins()
                .call(main_function_ref, &arguments);
            let results = shim_builder.function_builder.inst_results(inst);
            assert!(results.len() <= 1);
            results.get(0).copied()
        };

        let output_layout = entry_point
            .function
            .result
            .as_ref()
            .map(|result| {
                shim_builder.compile_result_shim(
                    result,
                    output.expect(
                        "compiled entry point doesn't return anything, but in naga IR it does.",
                    ),
                )
            })
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

        let typifier = typifier_from_function(&self.context.source, function);

        let mut function_compiler =
            FunctionCompiler::new(&self.context, &mut self.state, &typifier, function)?;

        let function_id = function_compiler.declare(&mut self.output)?;

        function_compiler.compile_block(&function.body)?;

        function_compiler.finish();

        self.output
            .define_function(function_id, &mut self.state.cl_context)?;

        Ok(function_id)
    }
}
