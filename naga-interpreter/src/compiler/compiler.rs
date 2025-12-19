use std::{
    borrow::Cow,
    convert::Infallible,
    fmt::Debug,
};

use cranelift_codegen::{
    ir::{
        AbiParam,
        Block,
        Immediate,
        InstBuilder,
        MemFlags,
        StackSlotData,
        StackSlotKind,
        Type,
        Value,
        condcodes::{
            FloatCC,
            IntCC,
        },
        immediates::{
            Ieee16,
            Imm64,
            V128Imm,
        },
        types,
    },
    isa::{
        CallConv,
        TargetFrontendConfig,
    },
    settings::Configurable,
};
use cranelift_frontend::{
    FunctionBuilder,
    FunctionBuilderContext,
    Variable,
};
use cranelift_jit::{
    JITBuilder,
    JITModule,
};
use cranelift_module::{
    FuncId,
    Linkage,
    Module,
};
use half::f16;

use crate::{
    compiler::{
        Error,
        bindings::ShimBuilder,
        module::{
            CompiledEntryPoint,
            CompiledModule,
        },
    },
    entry_point::EntryPoints,
    util::{
        SparseCoArena,
        typifier_from_function,
    },
};

#[derive(derive_more::Debug)]
pub struct Compiler<'module> {
    context: Context<'module>,

    #[debug(skip)]
    function_builder_context: FunctionBuilderContext,

    #[debug(skip)]
    cl_context: cranelift_codegen::Context,

    #[debug(skip)]
    jit_module: JITModule,

    next_anonymous_function_id: usize,
}

impl<'module> Compiler<'module> {
    pub fn new(
        module: &'module naga::Module,
        info: &'module naga::valid::ModuleInfo,
    ) -> Result<Self, Error> {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx())?;

        let mut flag_builder = cranelift_codegen::settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        let isa_builder = cranelift_native::builder().unwrap_or_else(|message| {
            panic!("host machine is not supported: {message}");
        });
        let isa = isa_builder
            .finish(cranelift_codegen::settings::Flags::new(flag_builder))
            .unwrap();

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
        let jit_module = JITModule::new(jit_builder);

        let cl_context = jit_module.make_context();

        let target_config = jit_module.target_config();

        let source_module = Context {
            module,
            info,
            layouter,
            target_config,
        };

        Ok(Self {
            context: source_module,
            function_builder_context: FunctionBuilderContext::new(),
            cl_context,
            jit_module,
            next_anonymous_function_id: 1,
        })
    }
}

impl<'module> Compiler<'module> {
    pub fn compile(mut self) -> Result<CompiledModule, Error> {
        let mut entry_points = EntryPoints::with_capacity(self.context.module.entry_points.len());

        for entry_point in &self.context.module.entry_points {
            let entry_point_data = self.compile_entry_point(entry_point)?;
            entry_points.push(entry_point, entry_point_data);
        }

        self.jit_module.finalize_definitions()?;

        Ok(CompiledModule::new(self.jit_module, entry_points))
    }

    pub fn anonymous_function_name(&mut self) -> String {
        let id = self.next_anonymous_function_id;
        self.next_anonymous_function_id += 1;
        format!("__naga_interpreter_anonymous_{id}")
    }

    pub fn compile_entry_point(
        &mut self,
        entry_point: &naga::EntryPoint,
    ) -> Result<CompiledEntryPoint, Error> {
        // compile entry point function
        let main_function_id = self.compile_function(&entry_point.function)?;

        // build shim
        self.jit_module.clear_context(&mut self.cl_context);

        let main_function_ref = self
            .jit_module
            .declare_func_in_func(main_function_id, &mut self.cl_context.func);

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

        let mut function_builder = FunctionBuilder::new(
            &mut self.cl_context.func,
            &mut self.function_builder_context,
        );

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

        println!("{:#?}", self.cl_context.func);

        let shim_function = self.jit_module.declare_function(
            "__naga_interpreter_shim", // this name is not accuarate anymore, isn't it :D
            Linkage::Local,
            &self.cl_context.func.signature,
        )?;

        self.jit_module
            .define_function(shim_function, &mut self.cl_context)?;

        Ok(CompiledEntryPoint {
            function_id: shim_function,
            input_layout,
            output_layout,
        })
    }

    pub fn compile_function(&mut self, function: &naga::Function) -> Result<FuncId, Error> {
        self.jit_module.clear_context(&mut self.cl_context);

        let function_name = function
            .name
            .as_ref()
            .map_or_else(|| Cow::Owned(self.anonymous_function_name()), Cow::Borrowed);

        let typifier = typifier_from_function(&self.context.module, function);

        // some immediates that we might use
        const VECTOR_REDUCE_SHUFFLE_MASKS: [V128Imm; 2] = [
            V128Imm([1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            V128Imm([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ];
        let vector_reduce_shuffle_masks = VECTOR_REDUCE_SHUFFLE_MASKS
            .map(|mask| self.cl_context.func.dfg.immediates.push(mask.into()));
        let matrix_transpose_shuffle_masks = {
            let vector_sizes = [2, 3, 4];
            vector_sizes.map(|columns| {
                vector_sizes.map(|rows| {
                    let mask = make_transpose_shuffle_mask(columns, rows);
                    self.cl_context.func.dfg.immediates.push(mask.into())
                })
            })
        };

        // function result
        if let Some(result) = &function.result {
            let ty = &self.context.module.types[result.ty];

            self.cl_context
                .func
                .signature
                .returns
                .push(self.context.abi_param(&ty.inner)?);
        }

        // function arguments
        for argument in &function.arguments {
            let ty = &self.context.module.types[argument.ty];

            self.cl_context
                .func
                .signature
                .params
                .push(self.context.abi_param(&ty.inner)?);
        }

        let mut function_builder = FunctionBuilder::new(
            &mut self.cl_context.func,
            &mut self.function_builder_context,
        );

        let entry_block = function_builder.create_block();

        function_builder.append_block_params_for_function_params(entry_block);
        function_builder.switch_to_block(entry_block);

        let mut function_compiler = FunctionCompiler {
            context: &self.context,
            function,
            typifier: &typifier,
            function_builder,
            entry_block,
            emitted_expression: Default::default(),
            //local_variables,
            vector_reduce_shuffle_masks,
            matrix_transpose_shuffle_masks,
        };

        function_compiler.compile_block(&function.body)?;

        function_compiler.function_builder.seal_block(entry_block);
        function_compiler.function_builder.finalize();

        let function_id = self.jit_module.declare_function(
            &function_name,
            Linkage::Local,
            &self.cl_context.func.signature,
        )?;

        self.jit_module
            .define_function(function_id, &mut self.cl_context)?;

        Ok(function_id)
    }
}

#[derive(derive_more::Debug)]
pub struct Context<'module> {
    pub module: &'module naga::Module,
    #[allow(unused)]
    pub info: &'module naga::valid::ModuleInfo,
    pub layouter: naga::proc::Layouter,
    #[debug(skip)]
    pub target_config: TargetFrontendConfig,
}

impl<'module> Context<'module> {
    pub fn abi_param(&self, ty: &naga::TypeInner) -> Result<AbiParam, Error> {
        Ok(AbiParam::new(self.abi_ty(ty)?))
    }

    pub fn abi_ty(&self, ty: &naga::TypeInner) -> Result<Type, Error> {
        // structs have to be lowered for the ABI. we can just pass a
        // pointer or struct offset? wgsl function arguments are not
        // variables. they can only be loaded via the FunctionCall
        // expression, so we don't have to worry about COW or anything
        // like that.
        //
        // https://users.rust-lang.org/t/help-trying-to-transfer-a-structure-from-cranelift-to-rust/106429/4

        use naga::TypeInner::*;

        let ty = match ty {
            Scalar(scalar) => self.scalar_type(*scalar)?,
            Vector { size, scalar } => self.vector_type(*scalar, *size)?,
            Matrix {
                columns,
                rows,
                scalar,
            } => self.matrix_type(*scalar, *columns, *rows)?,
            Atomic(scalar) => self.scalar_type(*scalar)?,
            Pointer { base: _, space: _ } => self.target_config.pointer_type(),
            ValuePointer {
                size: _,
                scalar: _,
                space: _,
            } => self.target_config.pointer_type(),
            Array {
                base: _,
                size: _,
                stride: _,
            } => self.target_config.pointer_type(),
            Struct {
                members: _,
                span: _,
            } => self.target_config.pointer_type(),
            Image {
                dim: _,
                arrayed: _,
                class: _,
            } => self.target_config.pointer_type(),
            Sampler { comparison: _ } => self.target_config.pointer_type(),
            AccelerationStructure { vertex_return: _ } => self.target_config.pointer_type(),
            RayQuery { vertex_return: _ } => self.target_config.pointer_type(),
            BindingArray { base: _, size: _ } => self.target_config.pointer_type(),
        };

        Ok(ty)
    }

    pub fn calling_convention(&self) -> CallConv {
        self.target_config.default_call_conv
    }

    pub fn pointer_type(&self) -> Type {
        self.target_config.pointer_type()
    }

    pub fn scalar_type(&self, scalar: naga::Scalar) -> Result<Type, Error> {
        use naga::ScalarKind::*;

        match scalar.kind {
            Sint | Uint => {
                match scalar.width {
                    4 => Some(types::I32),
                    _ => None,
                }
            }
            Float => {
                match scalar.width {
                    2 => Some(types::F16),
                    4 => Some(types::F32),
                    _ => None,
                }
            }
            Bool => {
                match scalar.width {
                    1 => Some(types::I8),
                    _ => None,
                }
            }
            AbstractInt | AbstractFloat => {
                panic!("Abstract types must not appear in naga IR")
            }
        }
        .ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Scalar(scalar),
            }
        })
    }

    pub fn vector_type(&self, scalar: naga::Scalar, size: naga::VectorSize) -> Result<Type, Error> {
        let lane = self.scalar_type(scalar)?;
        lane.by(size.into()).ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Vector { size, scalar },
            }
        })
    }

    pub fn matrix_type(
        &self,
        scalar: naga::Scalar,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Type, Error> {
        let lane = self.scalar_type(scalar)?;
        let column_lanes = match columns {
            naga::VectorSize::Bi => 2,
            naga::VectorSize::Tri => 4, // this is intentional, for alignment
            naga::VectorSize::Quad => 4,
        };
        let row_lanes = u32::from(rows);
        lane.by(column_lanes * row_lanes).ok_or_else(|| {
            Error::UnsupportedType {
                ty: naga::TypeInner::Matrix {
                    columns,
                    rows,
                    scalar,
                },
            }
        })
    }
}

#[derive(derive_more::Debug)]
struct FunctionCompiler<'module, 'compiler> {
    context: &'compiler Context<'module>,
    function: &'module naga::Function,
    typifier: &'compiler naga::front::Typifier,
    #[debug(skip)]
    function_builder: FunctionBuilder<'compiler>,
    entry_block: Block,
    emitted_expression: SparseCoArena<naga::Expression, Variable>,
    //local_variables: CoArena<naga::LocalVariable, Variable>,
    vector_reduce_shuffle_masks: [Immediate; 2],
    matrix_transpose_shuffle_masks: [[Immediate; 3]; 3],
}

impl<'module, 'compiler> FunctionCompiler<'module, 'compiler> {
    pub fn expression_ty(&self, expression: naga::Handle<naga::Expression>) -> &naga::TypeInner {
        self.typifier[expression].inner_with(&self.context.module.types)
    }

    pub fn compile_block(&mut self, naga_block: &naga::Block) -> Result<(), Error> {
        for statement in naga_block {
            self.compile_statement(statement)?;
        }
        Ok(())
    }

    pub fn compile_statement(&mut self, statement: &naga::Statement) -> Result<(), Error> {
        #![allow(unused_variables)]

        match statement {
            naga::Statement::Emit(range) => {
                self.compile_emit(range.clone())?;
            }
            naga::Statement::Block(naga_block) => {
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
            naga::Statement::If {
                condition,
                accept,
                reject,
            } => todo!(),
            naga::Statement::Switch { selector, cases } => todo!(),
            naga::Statement::Loop {
                body,
                continuing,
                break_if,
            } => todo!(),
            naga::Statement::Break => todo!(),
            naga::Statement::Continue => todo!(),
            naga::Statement::Return { value } => {
                self.compile_return(*value)?;
            }
            naga::Statement::Kill => todo!(),
            naga::Statement::ControlBarrier(barrier) => todo!(),
            naga::Statement::MemoryBarrier(barrier) => todo!(),
            naga::Statement::Store { pointer, value } => todo!(),
            naga::Statement::ImageStore {
                image,
                coordinate,
                array_index,
                value,
            } => todo!(),
            naga::Statement::Atomic {
                pointer,
                fun,
                value,
                result,
            } => todo!(),
            naga::Statement::ImageAtomic {
                image,
                coordinate,
                array_index,
                fun,
                value,
            } => todo!(),
            naga::Statement::WorkGroupUniformLoad { pointer, result } => todo!(),
            naga::Statement::Call {
                function,
                arguments,
                result,
            } => todo!(),
            naga::Statement::RayQuery { query, fun } => todo!(),
            naga::Statement::SubgroupBallot { result, predicate } => todo!(),
            naga::Statement::SubgroupGather {
                mode,
                argument,
                result,
            } => todo!(),
            naga::Statement::SubgroupCollectiveOperation {
                op,
                collective_op,
                argument,
                result,
            } => todo!(),
        }

        Ok(())
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

    pub fn compile_return(
        &mut self,
        expression: Option<naga::Handle<naga::Expression>>,
    ) -> Result<(), Error> {
        let return_value = expression
            .map(|expression| self.compile_expression(expression))
            .transpose()?;
        self.function_builder.ins().return_(return_value.as_slice());
        Ok(())
    }

    pub fn compile_expression(
        &mut self,
        expression: naga::Handle<naga::Expression>,
    ) -> Result<Value, Error> {
        #![allow(unused_variables)]

        let value = if let Some(variable) = self.emitted_expression.get(expression) {
            self.function_builder.use_var(*variable)
        }
        else {
            let output_type = self.typifier[expression].inner_with(&self.context.module.types);
            let expression = &self.function.expressions[expression];

            match expression {
                naga::Expression::Literal(literal) => {
                    self.compile_literal(*literal, output_type)?
                }
                naga::Expression::Constant(handle) => todo!(),
                naga::Expression::Override(handle) => todo!(),
                naga::Expression::ZeroValue(handle) => self.compile_zero(output_type)?,
                naga::Expression::Compose { ty, components } => {
                    self.compile_compose(*ty, components, output_type)?
                }
                naga::Expression::Access { base, index } => todo!(),
                naga::Expression::AccessIndex { base, index } => todo!(),
                naga::Expression::Splat { size, value } => todo!(),
                naga::Expression::Swizzle {
                    size,
                    vector,
                    pattern,
                } => todo!(),
                naga::Expression::FunctionArgument(function_argument) => {
                    self.function_builder.block_params(self.entry_block)
                        [*function_argument as usize]
                }
                naga::Expression::GlobalVariable(handle) => todo!(),
                naga::Expression::LocalVariable(handle) => todo!(),
                naga::Expression::Load { pointer } => todo!(),
                naga::Expression::ImageSample {
                    image,
                    sampler,
                    gather,
                    coordinate,
                    array_index,
                    offset,
                    level,
                    depth_ref,
                    clamp_to_edge,
                } => todo!(),
                naga::Expression::ImageLoad {
                    image,
                    coordinate,
                    array_index,
                    sample,
                    level,
                } => todo!(),
                naga::Expression::ImageQuery { image, query } => todo!(),
                naga::Expression::Unary { op, expr } => {
                    self.compile_unary_operator(*op, *expr, output_type)?
                }
                naga::Expression::Binary { op, left, right } => {
                    self.compile_binary_operator(*op, *left, *right, output_type)?
                }
                naga::Expression::Select {
                    condition,
                    accept,
                    reject,
                } => todo!(),
                naga::Expression::Derivative { axis, ctrl, expr } => todo!(),
                naga::Expression::Relational { fun, argument } => todo!(),
                naga::Expression::Math {
                    fun,
                    arg,
                    arg1,
                    arg2,
                    arg3,
                } => todo!(),
                naga::Expression::As {
                    expr,
                    kind,
                    convert,
                } => self.compile_as(*expr, *kind, *convert, output_type)?,
                naga::Expression::CallResult(handle) => todo!(),
                naga::Expression::AtomicResult { ty, comparison } => todo!(),
                naga::Expression::WorkGroupUniformLoadResult { ty } => todo!(),
                naga::Expression::ArrayLength(handle) => todo!(),
                naga::Expression::RayQueryVertexPositions { query, committed } => todo!(),
                naga::Expression::RayQueryProceedResult => todo!(),
                naga::Expression::RayQueryGetIntersection { query, committed } => todo!(),
                naga::Expression::SubgroupBallotResult => todo!(),
                naga::Expression::SubgroupOperationResult { ty } => todo!(),
            }
        };

        Ok(value)
    }

    pub fn compile_as(
        &mut self,
        input_expression: naga::Handle<naga::Expression>,
        kind: naga::ScalarKind,
        convert: Option<u8>,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        use naga::TypeInner::*;

        // these are not used actually
        let _ = (kind, convert);

        let input_value = self.compile_expression(input_expression)?;
        let input_ty = self.expression_ty(input_expression);

        match (input_ty, output_ty) {
            (naga::TypeInner::Scalar(input_scalar), naga::TypeInner::Scalar(output_scalar)) => {
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            (
                Vector {
                    size: input_size,
                    scalar: input_scalar,
                },
                Vector {
                    size: output_size,
                    scalar: output_scalar,
                },
            ) => {
                assert_eq!(input_size, output_size);
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            (
                Matrix {
                    columns: input_columns,
                    rows: input_rows,
                    scalar: input_scalar,
                },
                Matrix {
                    columns: output_columns,
                    rows: output_rows,
                    scalar: output_scalar,
                },
            ) => {
                assert_eq!(input_columns, output_columns);
                assert_eq!(input_rows, output_rows);
                self.compile_as_scalar(input_value, *input_scalar, *output_scalar)
            }
            _ => panic!("Invalid cast from {input_ty:?} to {output_ty:?}"),
        }
    }

    pub fn compile_as_scalar(
        &mut self,
        input_value: Value,
        input_scalar: naga::Scalar,
        output_scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::ScalarKind::*;

        // https://gpuweb.github.io/gpuweb/wgsl/#value-constructor-builtin-function

        let output_ty = self.context.scalar_type(output_scalar)?;

        let output_value = match (input_scalar.kind, output_scalar.kind) {
            (Sint, Uint) | (Uint, Sint) => input_value,
            (Sint, Bool) | (Uint, Bool) => {
                self.function_builder
                    .ins()
                    .icmp_imm(IntCC::NotEqual, input_value, 0)
            }
            (Sint, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_sint(output_ty, input_value)
            }
            (Uint, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_uint(output_ty, input_value)
            }
            (Float, Sint) => {
                self.function_builder
                    .ins()
                    .fcvt_to_sint(output_ty, input_value)
            }
            (Float, Uint) => {
                self.function_builder
                    .ins()
                    .fcvt_to_uint(output_ty, input_value)
            }
            (Float, Bool) => {
                let zero = match input_scalar.width {
                    2 => {
                        self.function_builder
                            .ins()
                            .f16const(ieee16_from_f16(f16::NEG_ZERO))
                    }
                    4 => self.function_builder.ins().f32const(0.0),
                    _ => panic!("invalid float width: {}", input_scalar.width),
                };
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, input_value, zero)
            }
            (Float, Float) if input_scalar.width < output_scalar.width => {
                self.function_builder.ins().fpromote(output_ty, input_value)
            }
            (Float, Float) if input_scalar.width > output_scalar.width => {
                self.function_builder.ins().fdemote(output_ty, input_value)
            }
            (Bool, Sint) | (Bool, Uint) => {
                self.function_builder.ins().uextend(output_ty, input_value)
            }
            (Bool, Float) => {
                self.function_builder
                    .ins()
                    .fcvt_from_uint(output_ty, input_value)
            }
            _ => panic!("unhandled type conversion: {input_scalar:?} to {output_scalar:?}"),
        };

        Ok(output_value)
    }

    pub fn compile_unary_operator(
        &mut self,
        operator: naga::UnaryOperator,
        input_expression: naga::Handle<naga::Expression>,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        use naga::TypeInner::*;

        let _ = output_ty;

        let input_value = self.compile_expression(input_expression)?;
        let input_ty = self.expression_ty(input_expression);

        match input_ty {
            Scalar(scalar) => self.compile_unary_scalar(operator, input_value, *scalar),
            Vector { size: _, scalar } => self.compile_unary_scalar(operator, input_value, *scalar),
            Matrix {
                columns: _,
                rows: _,
                scalar,
            } => self.compile_unary_scalar(operator, input_value, *scalar),
            _ => panic!("invalid unary operator {operator:?} on {input_ty:?}"),
        }
    }

    pub fn compile_unary_scalar(
        &mut self,
        operator: naga::UnaryOperator,
        input_value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::{
            ScalarKind::{
                Bool,
                Float,
                Sint,
                Uint,
            },
            UnaryOperator::*,
        };

        let output_value = match (scalar.kind, operator) {
            (Sint, Negate) => self.function_builder.ins().ineg(input_value),
            (Sint, BitwiseNot) => self.function_builder.ins().bnot(input_value),
            (Uint, BitwiseNot) => self.function_builder.ins().bnot(input_value),
            (Float, Negate) => self.function_builder.ins().fneg(input_value),
            (Bool, LogicalNot) => {
                self.function_builder
                    .ins()
                    .icmp_imm(IntCC::Equal, input_value, 0)
            }
            _ => panic!("invalid unary operator {operator:?} on {scalar:?}"),
        };

        Ok(output_value)
    }

    pub fn compile_binary_operator(
        &mut self,
        operator: naga::BinaryOperator,
        left_expression: naga::Handle<naga::Expression>,
        right_expression: naga::Handle<naga::Expression>,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        use naga::{
            BinaryOperator::*,
            TypeInner::*,
        };

        let _ = output_ty;

        let left_value = self.compile_expression(left_expression)?;
        let right_value = self.compile_expression(right_expression)?;
        let left_ty = self.typifier[left_expression].inner_with(&self.context.module.types);
        let right_ty = self.typifier[right_expression].inner_with(&self.context.module.types);

        let output_value = match (left_ty, right_ty) {
            (Scalar(left_scalar), Scalar(right_scalar)) => {
                assert_eq!(left_scalar, right_scalar);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Vector {
                    size: left_size,
                    scalar: left_scalar,
                },
                Vector {
                    size: right_size,
                    scalar: right_scalar,
                },
            ) => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_size, right_size);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if operator == Add || operator == Subtract => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_columns, right_columns);
                assert_eq!(left_rows, right_rows);
                self.compile_scalar_binary_operator(
                    operator,
                    left_value,
                    right_value,
                    *left_scalar,
                )?
            }
            (
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
                Scalar(scalar),
            ) if operator == Multiply => {
                let scalar_splat = self.function_builder.ins().splat(
                    self.context
                        .matrix_type(*scalar, *matrix_columns, *matrix_rows)?,
                    left_value,
                );
                self.compile_scalar_multiply(left_value, scalar_splat, *scalar)?
            }
            (
                Scalar(scalar),
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, scalar);
                let scalar_splat = self.function_builder.ins().splat(
                    self.context
                        .matrix_type(*scalar, *matrix_columns, *matrix_rows)?,
                    left_value,
                );
                self.compile_scalar_multiply(scalar_splat, right_value, *scalar)?
            }
            (
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
                Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, vector_scalar);
                assert_eq!(matrix_rows, vector_size);

                // transpose matrix with shuffle
                let matrix =
                    self.compile_matrix_transpose(left_value, *matrix_columns, *matrix_rows)?;
                // then vector * matrix
                self.compile_vector_matrix_multiply(
                    right_value,
                    matrix,
                    *matrix_scalar,
                    *matrix_rows,
                    *matrix_columns,
                )?
            }
            (
                Vector {
                    size: vector_size,
                    scalar: vector_scalar,
                },
                Matrix {
                    columns: matrix_columns,
                    rows: matrix_rows,
                    scalar: matrix_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(matrix_scalar, vector_scalar);
                assert_eq!(matrix_columns, vector_size);

                self.compile_vector_matrix_multiply(
                    left_value,
                    right_value,
                    *matrix_scalar,
                    *matrix_rows,
                    *matrix_columns,
                )?
            }
            (
                Matrix {
                    columns: left_columns,
                    rows: left_rows,
                    scalar: left_scalar,
                },
                Matrix {
                    columns: right_columns,
                    rows: right_rows,
                    scalar: right_scalar,
                },
            ) if operator == Multiply => {
                assert_eq!(left_scalar, right_scalar);
                assert_eq!(left_rows, right_columns);
                todo!("matrix * matrix");
            }
            _ => panic!("invalid unary operator {operator:?} on {left_ty:?} and {right_ty:?}"),
        };

        Ok(output_value)
    }

    pub fn compile_scalar_binary_operator(
        &mut self,
        operator: naga::BinaryOperator,
        left_value: Value,
        right_value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::{
            BinaryOperator::*,
            ScalarKind::{
                Bool,
                Float,
                Sint,
                Uint,
            },
        };

        let output = match (scalar.kind, operator) {
            (Sint, Add) | (Uint, Add) => self.function_builder.ins().iadd(left_value, right_value),
            (Float, Add) => self.function_builder.ins().fadd(left_value, right_value),
            (Sint, Subtract) | (Uint, Subtract) => {
                self.function_builder.ins().isub(left_value, right_value)
            }
            (Float, Subtract) => self.function_builder.ins().fsub(left_value, right_value),
            (Sint, Multiply) | (Uint, Multiply) => {
                self.function_builder.ins().imul(left_value, right_value)
            }
            (Float, Multiply) => self.function_builder.ins().fmul(left_value, right_value),
            (Sint, Divide) => self.function_builder.ins().sdiv(left_value, right_value),
            (Uint, Divide) => self.function_builder.ins().udiv(left_value, right_value),
            (Float, Divide) => self.function_builder.ins().fdiv(left_value, right_value),
            (Sint, Modulo) => self.function_builder.ins().srem(left_value, right_value),
            (Uint, Modulo) => self.function_builder.ins().urem(left_value, right_value),
            (Float, Modulo) => {
                // https://www.w3.org/TR/WGSL/#arithmetic-expr
                // > If T is a floating point type, the result is equal to:
                // > e1 - e2 * trunc(e1 / e2).
                let x = self.function_builder.ins().fdiv(left_value, right_value);
                let x = self.function_builder.ins().trunc(x);
                let x = self.function_builder.ins().fmul(right_value, x);
                self.function_builder.ins().fsub(left_value, x)
            }
            (Uint, Equal) | (Sint, Equal) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::Equal, left_value, right_value)
            }
            (Float, Equal) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::Equal, left_value, right_value)
            }
            (Uint, NotEqual) | (Sint, NotEqual) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::NotEqual, left_value, right_value)
            }
            (Float, NotEqual) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::NotEqual, left_value, right_value)
            }
            (Uint, Less) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, left_value, right_value)
            }
            (Sint, Less) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, left_value, right_value)
            }
            (Float, Less) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::LessThan, left_value, right_value)
            }
            (Uint, LessEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedLessThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Sint, LessEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::SignedLessThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Float, LessEqual) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::LessThanOrEqual, left_value, right_value)
            }
            (Uint, Greater) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedGreaterThan,
                    left_value,
                    right_value,
                )
            }
            (Sint, Greater) => {
                self.function_builder
                    .ins()
                    .icmp(IntCC::SignedGreaterThan, left_value, right_value)
            }
            (Float, Greater) => {
                self.function_builder
                    .ins()
                    .fcmp(FloatCC::GreaterThan, left_value, right_value)
            }
            (Uint, GreaterEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::UnsignedGreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Sint, GreaterEqual) => {
                self.function_builder.ins().icmp(
                    IntCC::SignedGreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Float, GreaterEqual) => {
                self.function_builder.ins().fcmp(
                    FloatCC::GreaterThanOrEqual,
                    left_value,
                    right_value,
                )
            }
            (Uint, And) | (Sint, And) | (Bool, LogicalAnd) => {
                self.function_builder.ins().band(left_value, right_value)
            }
            (Uint, ExclusiveOr) | (Sint, ExclusiveOr) => {
                self.function_builder.ins().bxor(left_value, right_value)
            }
            (Uint, InclusiveOr) | (Sint, InclusiveOr) | (Bool, LogicalOr) => {
                self.function_builder.ins().bor(left_value, right_value)
            }
            (Uint, ShiftLeft) | (Sint, ShiftLeft) => {
                self.function_builder.ins().ishl(left_value, right_value)
            }
            (Uint, ShiftRight) => self.function_builder.ins().ushr(left_value, right_value),
            (Sint, ShiftRight) => self.function_builder.ins().sshr(left_value, right_value),
            _ => panic!("invalid unary operator {operator:?} on {scalar:?}"),
        };

        Ok(output)
    }

    pub fn compile_scalar_multiply(
        &mut self,
        left_value: Value,
        right_value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::ScalarKind::{
            Float,
            Sint,
            Uint,
        };

        let output = match scalar.kind {
            Sint | Uint => self.function_builder.ins().imul(left_value, right_value),
            Float => self.function_builder.ins().fmul(left_value, right_value),
            _ => panic!("Invalid scalar for multiplication: {scalar:?}"),
        };

        Ok(output)
    }

    pub fn compile_vector_matrix_multiply(
        &mut self,
        vector: Value,
        matrix: Value,
        scalar: naga::Scalar,
        rows: naga::VectorSize,
        columns: naga::VectorSize,
    ) -> Result<Value, Error> {
        let mut output = self.compile_vector_zero(scalar, rows)?;

        for column in 0..u8::from(columns) {
            let column_vector = self.function_builder.ins().extractlane(matrix, column);
            let column_mul = self.compile_scalar_multiply(vector, column_vector, scalar)?;
            let column_sum = self.compile_vector_sum(column_mul, scalar)?;
            output = self
                .function_builder
                .ins()
                .insertlane(output, column_sum, column);
        }

        Ok(output)
    }

    pub fn compile_vector_sum(
        &mut self,
        value: Value,
        scalar: naga::Scalar,
    ) -> Result<Value, Error> {
        use naga::ScalarKind::{
            Float,
            Sint,
            Uint,
        };

        let reduce_output = match scalar.kind {
            Sint | Uint => {
                self.compile_vector_reduce(value, |function_builder, accu, shuffled| {
                    Ok(function_builder.ins().iadd(accu, shuffled))
                })?
            }
            Float => {
                self.compile_vector_reduce(value, |function_builder, accu, shuffled| {
                    Ok(function_builder.ins().fadd(accu, shuffled))
                })?
            }
            _ => panic!("Invalid scalar for multiplication: {scalar:?}"),
        };

        let output = self.function_builder.ins().extractlane(reduce_output, 0);
        Ok(output)
    }

    pub fn compile_vector_reduce(
        &mut self,
        mut value: Value,
        mut op: impl FnMut(&mut FunctionBuilder, Value, Value) -> Result<Value, Error>,
    ) -> Result<Value, Error> {
        // (sum example)
        //               v0      v1      v2      v3
        //
        // shuffle:       1       0       3       2
        // add:     (v0+v1) (v1+v0) (v2+v3) (v3+v2)
        // shuffle:       2       3       0       1
        // add:     (v0+v1+v2+v2), ...

        for i in 0..1 {
            let shuffled = self.function_builder.ins().shuffle(
                value,
                value,
                self.vector_reduce_shuffle_masks[i],
            );
            value = op(&mut self.function_builder, value, shuffled)?;
        }

        Ok(value)
    }

    pub fn compile_literal(
        &mut self,
        literal: naga::Literal,
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        let _ = output_ty;

        let output = match literal {
            naga::Literal::F64(value) => self.function_builder.ins().f64const(value),
            naga::Literal::F32(value) => self.function_builder.ins().f32const(value),
            naga::Literal::F16(value) => {
                self.function_builder.ins().f16const(ieee16_from_f16(value))
            }
            naga::Literal::U32(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I32, Imm64::new(value.into()))
            }
            naga::Literal::I32(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I32, Imm64::new(value.into()))
            }
            naga::Literal::U64(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I64, Imm64::new(value as i64))
            }
            naga::Literal::I64(value) => {
                self.function_builder
                    .ins()
                    .iconst(types::I64, Imm64::new(value))
            }
            naga::Literal::Bool(value) => {
                self.function_builder.ins().iconst(types::I8, value as i64)
            }
            _ => panic!("abstract literal: {literal:?}"),
        };

        Ok(output)
    }

    pub fn compile_zero(&mut self, output_type: &naga::TypeInner) -> Result<Value, Error> {
        use naga::TypeInner::*;

        let output = match output_type {
            Scalar(scalar) => self.compile_scalar_zero(*scalar)?,
            Vector { size, scalar } => self.compile_vector_zero(*scalar, *size)?,
            Matrix {
                columns,
                rows,
                scalar,
            } => self.compile_matrix_zero(*scalar, *columns, *rows)?,
            Atomic(scalar) => self.compile_scalar_zero(*scalar)?,
            Array {
                base,
                size: _,
                stride: _,
            } => {
                self.compile_stack_zero(naga::proc::TypeLayout {
                    size: output_type.size(self.context.module.to_ctx()),
                    alignment: self.context.layouter[*base].alignment,
                })?
            }
            Struct { members, span } => {
                self.compile_stack_zero(naga::proc::TypeLayout {
                    size: *span,
                    alignment: self.context.layouter[members[0].ty].alignment,
                })?
            }
            _ => panic!("type can't be zeroed: {output_type:?}"),
        };

        Ok(output)
    }

    pub fn compile_scalar_zero(&mut self, scalar: naga::Scalar) -> Result<Value, Error> {
        use naga::ScalarKind::*;

        let output = match (scalar.kind, scalar.width) {
            (Sint, 4) | (Uint, 4) => self.function_builder.ins().iconst(types::I64, 0),
            (Float, 2) => {
                self.function_builder
                    .ins()
                    .f16const(ieee16_from_f16(f16::ZERO))
            }
            (Float, 4) => self.function_builder.ins().f32const(0.0),
            (Bool, 1) => self.function_builder.ins().iconst(types::I8, 0),
            _ => panic!("Invalid scalar type: {scalar:?}"),
        };

        Ok(output)
    }

    pub fn compile_vector_zero(
        &mut self,
        scalar: naga::Scalar,
        size: naga::VectorSize,
    ) -> Result<Value, Error> {
        // todo: use vconst
        let zero = self.compile_scalar_zero(scalar)?;
        let vector_ty = self.context.vector_type(scalar, size)?;
        let output = self.function_builder.ins().splat(vector_ty, zero);
        Ok(output)
    }

    pub fn compile_matrix_zero(
        &mut self,
        scalar: naga::Scalar,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Value, Error> {
        // todo: use vconst
        let zero = self.compile_scalar_zero(scalar)?;
        let vector_ty = self.context.matrix_type(scalar, columns, rows)?;
        let output = self.function_builder.ins().splat(vector_ty, zero);
        Ok(output)
    }

    pub fn compile_stack_zero(
        &mut self,
        type_layout: naga::proc::TypeLayout,
    ) -> Result<Value, Error> {
        let alignment = alignment_log2(type_layout.alignment);
        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment,
                key: None,
            });

        let output =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        self.function_builder.emit_small_memset(
            self.context.target_config,
            output,
            0,
            type_layout.size.into(),
            alignment,
            MemFlags::new(),
        );

        Ok(output)
    }

    pub fn compile_matrix_transpose(
        &mut self,
        value: Value,
        columns: naga::VectorSize,
        rows: naga::VectorSize,
    ) -> Result<Value, Error> {
        let output = self.function_builder.ins().shuffle(
            value,
            value,
            self.matrix_transpose_shuffle_masks[columns as u8 as usize - 1]
                [rows as u8 as usize - 1],
        );

        Ok(output)
    }

    pub fn compile_compose(
        &mut self,
        ty: naga::Handle<naga::Type>,
        components: &[naga::Handle<naga::Expression>],
        output_ty: &naga::TypeInner,
    ) -> Result<Value, Error> {
        #![allow(unused_variables)]

        let _ = output_ty;
        let ty = &self.context.module.types[ty];

        let components = components
            .into_iter()
            .copied()
            .map(|expression| {
                let value = self.compile_expression(expression)?;
                let ty = self.typifier[expression].inner_with(&self.context.module.types);
                Ok::<_, Error>((value, ty))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let output = match &ty.inner {
            naga::TypeInner::Vector { size, scalar } => {
                let mut output = self.compile_vector_zero(*scalar, *size)?;

                for (i, (value, _ty)) in components.into_iter().enumerate() {
                    output = self
                        .function_builder
                        .ins()
                        .insertlane(output, value, i as u8);
                }

                output
            }
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar: _,
            } => {
                match &components[0].1 {
                    naga::TypeInner::Scalar(scalar) => {
                        let mut output = self.compile_matrix_zero(*scalar, *columns, *rows)?;
                        let lanes = MatrixLanes::new(*columns, *rows);

                        for (i, (value, _ty)) in components.into_iter().enumerate() {
                            output = self.function_builder.ins().insertlane(
                                output,
                                value,
                                lanes.lane_flat(i as u8),
                            );
                        }

                        output
                    }
                    naga::TypeInner::Vector { size, scalar } => {
                        assert_eq!(*size, *rows);

                        let mut output = self.compile_matrix_zero(*scalar, *columns, *rows)?;
                        let lanes = MatrixLanes::new(*columns, *rows);

                        lanes.for_each(|lane, column, row| {
                            if let Some((value, _ty)) = components.get(usize::from(column)) {
                                let value = self.function_builder.ins().extractlane(*value, row);
                                output =
                                    self.function_builder.ins().insertlane(output, value, lane);
                            }
                        });

                        output
                    }
                    _ => panic!("Invalid compose: {ty:?} from {:?}", components[0].1),
                }
            }
            naga::TypeInner::Array { base, size, stride } => {
                /*let type_layout = self.context.layouter[*base];
                let array = self.compile_stack_zero(type_layout)?;

                let offset = 0;
                for (value, ty) in components {

                    self.function_builder.

                    offset += *stride;
                    todo!();
                }

                array*/
                todo!();
            }
            naga::TypeInner::Struct { members, span } => {
                todo!();
            }
            _ => panic!("Invalid compose: {ty:?}"),
        };

        Ok(output)
    }
}

fn ieee16_from_f16(x: f16) -> Ieee16 {
    Ieee16::with_bits(x.to_bits())
}

#[derive(Debug)]
struct MatrixLanes {
    columns: u8,
    rows: u8,
    row_stride: u8,
}

impl MatrixLanes {
    pub fn new(columns: impl Into<u8>, rows: impl Into<u8>) -> Self {
        let columns = columns.into();
        let rows = rows.into();

        let row_stride: u8 = match columns {
            2 => 2,
            3 => 4,
            4 => 4,
            _ => unreachable!("invalid matrix size: {columns}x{rows}"),
        };

        Self {
            columns,
            rows,
            row_stride,
        }
    }

    pub fn lane(&self, column: u8, row: u8) -> u8 {
        assert!(column < self.columns as u8);
        assert!(row < self.rows as u8);
        row * self.row_stride + column
    }

    pub fn lane_flat(&self, i: u8) -> u8 {
        let column = i / self.rows;
        let row = i % self.rows;
        self.lane(column, row)
    }

    pub fn for_each(&self, mut f: impl FnMut(u8, u8, u8)) {
        self.try_for_each(|lane, row, column| {
            f(lane, row, column);
            Ok::<(), Infallible>(())
        })
        .unwrap_or_else(|e| match e {})
    }

    pub fn try_for_each<E>(&self, mut f: impl FnMut(u8, u8, u8) -> Result<(), E>) -> Result<(), E> {
        for row in 0..self.columns {
            for column in 0..self.rows {
                f(self.lane(row, column), row, column)?;
            }
        }
        Ok(())
    }
}

fn make_transpose_shuffle_mask(columns: u8, rows: u8) -> V128Imm {
    let mut mask = [0; 16];

    let lanes = MatrixLanes::new(columns, rows);

    lanes.for_each(|lane, row, column| {
        mask[usize::from(lane)] = lanes.lane(row, column);
    });

    V128Imm(mask)
}

pub(super) fn alignment_log2(alignment: naga::proc::Alignment) -> u8 {
    const ALIGNMENTS: [naga::proc::Alignment; 5] = [
        naga::proc::Alignment::ONE,
        naga::proc::Alignment::TWO,
        naga::proc::Alignment::FOUR,
        naga::proc::Alignment::EIGHT,
        naga::proc::Alignment::SIXTEEN,
    ];
    ALIGNMENTS
        .iter()
        .enumerate()
        .find(|(_i, x)| **x == alignment)
        .map(|(i, _x)| i)
        .unwrap()
        .try_into()
        .unwrap()
}
