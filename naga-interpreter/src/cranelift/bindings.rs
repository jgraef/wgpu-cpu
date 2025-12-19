use std::{
    any::Any,
    ffi::{
        c_int,
        c_void,
    },
    panic::AssertUnwindSafe,
};

use cranelift_codegen::ir::{
    AbiParam,
    Block,
    InstBuilder,
    MemFlags,
    SigRef,
    Signature,
    StackSlotData,
    StackSlotKind,
    Type,
    Value,
    types,
};
use cranelift_frontend::FunctionBuilder;

use crate::{
    bindings::{
        IoBindingVisitor,
        ShaderInput,
        ShaderOutput,
        VisitIoBindings,
    },
    cranelift::{
        Error,
        compiler::{
            Context,
            alignment_log2,
        },
    },
};

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct ShimVtable {
    // the size of the stack slot is not strictly needed, but let's keep it somewhat safe :D
    pub copy_inputs_to: unsafe fn(*mut c_void, *mut u8, usize) -> c_int,
    pub copy_outputs_from: unsafe fn(*mut c_void, *const u8, usize) -> c_int,
}

impl ShimVtable {
    pub const fn new<I, O>() -> Self
    where
        I: ShaderInput,
        O: ShaderOutput,
    {
        unsafe fn copy_inputs_to<I, O>(data: *mut c_void, target: *mut u8, len: usize) -> i32
        where
            I: ShaderInput,
        {
            let data = unsafe {
                // SAFETY: It is unsafe to pass anything but a pointer to ShimData<I, O> to this
                // function. The lifetime of the ShimData is ensured by the code calling into
                // the generated code.
                &mut *(data as *mut ShimData<I, O>)
            };

            let target = std::ptr::slice_from_raw_parts_mut(target, len);
            let target = unsafe {
                // SAFETY: The `target` pointer with `len` length must correspond to a valid
                // `&mut [u8]` produced by the compiled code
                &mut *target
            };

            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                data.copy_inputs_to(target);
            }));

            data.panic = result;
            data.panic.is_ok() as i32
        }

        unsafe fn copy_outputs_from<I, O>(data: *mut c_void, source: *const u8, len: usize) -> i32
        where
            O: ShaderOutput,
        {
            let data = unsafe {
                // SAFETY: It is unsafe to pass anything but a pointer to ShimData<I, O> to this
                // function. The lifetime of the ShimData is ensured by the code calling into
                // the generated code.
                &mut *(data as *mut ShimData<I, O>)
            };

            let source = std::ptr::slice_from_raw_parts(source, len);
            let source = unsafe {
                // SAFETY: The `source` pointer with `len` length must correspond to a valid
                // `&[u8]` produced by the compiled code
                &*source
            };

            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                data.copy_outputs_from(source);
            }));

            data.panic = result;
            data.panic.is_ok() as i32
        }

        ShimVtable {
            copy_inputs_to: copy_inputs_to::<I, O>,
            copy_outputs_from: copy_outputs_from::<I, O>,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShimVtableSignatures {
    pub copy_inputs_to: SigRef,
    pub copy_outputs_from: SigRef,
}

impl ShimVtableSignatures {
    pub fn new(function_builder: &mut FunctionBuilder, context: &Context) -> Self {
        let data = AbiParam::new(context.pointer_type());

        let copy_inputs_to = function_builder.import_signature(Signature {
            params: vec![
                data,
                AbiParam::new(context.pointer_type()),
                AbiParam::new(context.pointer_type()),
            ],
            returns: vec![AbiParam::new(types::I32)],
            call_conv: context.calling_convention(),
        });

        let copy_outputs_from = function_builder.import_signature(Signature {
            params: vec![
                data,
                AbiParam::new(context.pointer_type()),
                AbiParam::new(context.pointer_type()),
            ],
            returns: vec![AbiParam::new(types::I32)],
            call_conv: context.calling_convention(),
        });

        Self {
            copy_inputs_to,
            copy_outputs_from,
        }
    }
}

#[derive(Debug)]
pub struct ShimData<'layout, I, O> {
    pub input: I,
    pub input_layout: &'layout [BindingStackLayout],
    pub output: O,
    pub output_layout: &'layout [BindingStackLayout],
    pub panic: Result<(), Box<dyn Any + Send + 'static>>,
}

impl<'layout, I, O> ShimData<'layout, I, O>
where
    I: ShaderInput,
{
    pub fn copy_inputs_to(&mut self, target: &mut [u8]) {
        for layout in self.input_layout {
            self.input.write_into(
                &layout.binding,
                &layout.ty,
                &mut target[layout.offset..][..layout.size],
            );
        }
    }
}

impl<'layout, I, O> ShimData<'layout, I, O>
where
    O: ShaderOutput,
{
    pub fn copy_outputs_from(&mut self, source: &[u8]) {
        for layout in self.output_layout {
            self.output.read_from(
                &layout.binding,
                &layout.ty,
                &source[layout.offset..][..layout.size],
            );
        }
    }
}

#[derive(Clone, Debug)]
pub struct BindingStackLayout {
    pub binding: naga::Binding,
    pub ty: naga::Type,
    pub offset: usize,
    pub size: usize,
}

#[derive(Clone, Debug)]
pub struct CollectBindingStackLayouts<'module> {
    pub layouter: &'module naga::proc::Layouter,
    pub layouts: Vec<BindingStackLayout>,
}

impl<'module> VisitIoBindings for CollectBindingStackLayouts<'module> {
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        let _ = (name, top_level);

        let type_layout = self.layouter[ty_handle];
        let size = type_layout.size;

        self.layouts.push(BindingStackLayout {
            binding: binding.clone(),
            ty: ty.clone(),
            offset: offset as usize,
            size: size as usize,
        });
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ShimVtableCallCompiler {
    pub shim_vtable: Value,
    pub shim_data: Value,
    pub shim_vtable_signatures: ShimVtableSignatures,
    pub function_pointer_type: Type,
}

impl ShimVtableCallCompiler {
    pub fn new(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        shim_vtable: Value,
        shim_data: Value,
    ) -> Self {
        let shim_vtable_signatures = ShimVtableSignatures::new(function_builder, context);
        Self {
            shim_vtable,
            shim_data,
            shim_vtable_signatures,
            function_pointer_type: context.pointer_type(),
        }
    }
    pub fn compile_call(
        &self,
        function_builder: &mut FunctionBuilder,
        vtable_offset: usize,
        signature: SigRef,
        arguments: impl IntoIterator<Item = Value>,
    ) -> Value {
        let vtable_offset =
            i32::try_from(vtable_offset).expect("shim vtable offset does not fit into an i32");

        let arguments = std::iter::once(self.shim_data)
            .chain(arguments)
            .collect::<Vec<_>>();

        let func_ptr = function_builder.ins().load(
            self.function_pointer_type,
            MemFlags::new(),
            self.shim_vtable,
            vtable_offset,
        );
        let inst = function_builder
            .ins()
            .call_indirect(signature, func_ptr, &arguments);
        let results = function_builder.inst_results(inst);
        assert_eq!(results.len(), 1);
        results[0]
    }
}

macro_rules! call_shim_vtable {
    ($compiler: expr, $function_builder:expr, $func:ident($($arg:expr),*)) => {
        {
            let vtable_offset = memoffset::offset_of!(ShimVtable, $func);
            let signature = $compiler.shim_vtable_signatures.$func;
            $compiler.compile_call($function_builder, vtable_offset, signature, [$($arg),*])
        }
    };
}

pub struct ShimBuilder<'module, 'compiler> {
    context: &'compiler Context<'module>,
    pub function_builder: FunctionBuilder<'compiler>,
    shim_vtable_caller: ShimVtableCallCompiler,
    panic_block: Block,
}

impl<'module, 'compiler> ShimBuilder<'module, 'compiler> {
    pub fn new(
        context: &'compiler Context<'module>,
        mut function_builder: FunctionBuilder<'compiler>,
        shim_vtable: Value,
        shim_data: Value,
        panic_block: Block,
    ) -> Self {
        let shim_vtable_caller =
            ShimVtableCallCompiler::new(context, &mut function_builder, shim_vtable, shim_data);

        Self {
            context,
            function_builder,
            shim_vtable_caller,
            panic_block,
        }
    }

    pub fn compile_arguments_shim(
        &mut self,
        arguments: &[naga::FunctionArgument],
    ) -> Result<(Vec<Value>, Vec<BindingStackLayout>), Error> {
        let mut arguments = arguments.iter();
        let mut collect_binding_stack_layouts = CollectBindingStackLayouts {
            layouter: &self.context.layouter,
            layouts: vec![],
        };
        let mut visitor = IoBindingVisitor::new(
            &self.context.module.types,
            &mut collect_binding_stack_layouts,
        );

        let mut pass_arguments = vec![];

        let type_layout = {
            let Some(first) = arguments.next()
            else {
                return Ok((vec![], vec![]));
            };

            let argument_type = &self.context.module.types[first.ty];
            let mut type_layout = self.context.layouter[first.ty];
            visitor.visit_function_argument(first, 0);
            pass_arguments.push(PassBy::new(&self.context, &argument_type.inner, 0)?);

            for argument in arguments {
                let argument_type = &self.context.module.types[argument.ty];
                let argument_type_layout = &self.context.layouter[argument.ty];

                let offset = type_layout.alignment.round_up(type_layout.size);
                let len = argument_type_layout.size;

                type_layout.size += len;

                visitor.visit_function_argument(argument, offset);
                pass_arguments.push(PassBy::new(&self.context, &argument_type.inner, offset)?);
            }

            type_layout
        };

        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment_log2(type_layout.alignment),
                key: None,
            });

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        let len = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(type_layout.size));

        let result = call_shim_vtable!(
            self.shim_vtable_caller,
            &mut self.function_builder,
            copy_inputs_to(stack_slot_pointer, len)
        );

        // check result (error?)
        // 1=ok, 0=panic
        let continue_block = self.function_builder.create_block();
        self.function_builder
            .ins()
            .brif(result, continue_block, [], self.panic_block, []);
        self.function_builder.switch_to_block(continue_block);
        self.function_builder.seal_block(continue_block);

        let argument_values = pass_arguments
            .into_iter()
            .map(|pass| {
                match pass {
                    PassBy::Reference { offset } => {
                        self.function_builder
                            .ins()
                            .iadd_imm(stack_slot_pointer, i64::from(offset))
                    }
                    PassBy::Value { offset, ty } => {
                        self.function_builder.ins().load(
                            ty,
                            MemFlags::new(),
                            stack_slot_pointer,
                            i32::try_from(offset).expect("stack offset overflow"),
                        )
                    }
                }
            })
            .collect();

        Ok((argument_values, collect_binding_stack_layouts.layouts))
    }

    pub fn compile_result_shim(
        &mut self,
        result: &naga::FunctionResult,
        value: Value,
    ) -> Result<Vec<BindingStackLayout>, Error> {
        let type_layout = self.context.layouter[result.ty];
        let result_type = &self.context.module.types[result.ty];

        let mut collect_binding_stack_layouts = CollectBindingStackLayouts {
            layouter: &self.context.layouter,
            layouts: vec![],
        };
        let mut visitor = IoBindingVisitor::new(
            &self.context.module.types,
            &mut collect_binding_stack_layouts,
        );
        visitor.visit_function_result(result, 0);
        let pass_result = PassBy::new(&self.context, &result_type.inner, 0)?;

        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(StackSlotData {
                kind: StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment_log2(type_layout.alignment),
                key: None,
            });

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        let len = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(type_layout.size));

        match pass_result {
            PassBy::Reference { offset: _ } => {}
            PassBy::Value { offset, ty: _ } => {
                self.function_builder.ins().store(
                    MemFlags::new(),
                    value,
                    stack_slot_pointer,
                    i32::try_from(offset).expect("stack offset overflow"),
                );
            }
        }

        let result = call_shim_vtable!(
            self.shim_vtable_caller,
            &mut self.function_builder,
            copy_outputs_from(stack_slot_pointer, len)
        );

        // check result (error?)
        // 1=ok, 0=panic
        let continue_block = self.function_builder.create_block();
        self.function_builder
            .ins()
            .brif(result, continue_block, [], self.panic_block, []);
        self.function_builder.switch_to_block(continue_block);
        self.function_builder.seal_block(continue_block);

        Ok(collect_binding_stack_layouts.layouts)
    }
}

#[derive(Debug)]
enum PassBy {
    Reference { offset: u32 },
    Value { offset: u32, ty: Type },
}

impl PassBy {
    pub fn new(context: &Context, ty: &naga::TypeInner, offset: u32) -> Result<Self, Error> {
        let pass_by = match ty {
            naga::TypeInner::Scalar(scalar) => {
                let ty = context.scalar_type(*scalar)?;
                PassBy::Value { offset, ty }
            }
            naga::TypeInner::Vector { size, scalar } => {
                let ty = context.vector_type(*scalar, *size)?;
                PassBy::Value { offset, ty }
            }
            naga::TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => {
                let ty = context.matrix_type(*scalar, *columns, *rows)?;
                PassBy::Value { offset, ty }
            }
            naga::TypeInner::Struct {
                members: _,
                span: _,
            } => PassBy::Reference { offset },
            _ => panic!("Invalid binding argument/result type: {ty:?}"),
        };
        Ok(pass_by)
    }
}
