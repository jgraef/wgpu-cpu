use std::{
    any::Any,
    ffi::{
        c_int,
        c_void,
    },
    panic::AssertUnwindSafe,
};

use cranelift_codegen::ir::{
    self,
    InstBuilder,
};
use cranelift_frontend::FunctionBuilder;

use crate::{
    bindings::{
        IoBindingVisitor,
        ShaderInput,
        ShaderOutput,
        VisitIoBindings,
    },
    compiler::{
        Error,
        compiler::Context,
        types::Type,
        util::alignment_log2,
        value::{
            Load,
            Location,
            StackLocation,
            Store,
            Value,
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
    pub copy_inputs_to: ir::SigRef,
    pub copy_outputs_from: ir::SigRef,
}

impl ShimVtableSignatures {
    pub fn new(function_builder: &mut FunctionBuilder, context: &Context) -> Self {
        let data = ir::AbiParam::new(context.pointer_type());

        let copy_inputs_to = function_builder.import_signature(ir::Signature {
            params: vec![
                data,
                ir::AbiParam::new(context.pointer_type()),
                ir::AbiParam::new(context.pointer_type()),
            ],
            returns: vec![ir::AbiParam::new(ir::types::I32)],
            call_conv: context.calling_convention(),
        });

        let copy_outputs_from = function_builder.import_signature(ir::Signature {
            params: vec![
                data,
                ir::AbiParam::new(context.pointer_type()),
                ir::AbiParam::new(context.pointer_type()),
            ],
            returns: vec![ir::AbiParam::new(ir::types::I32)],
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
pub struct CollectBindingStackLayouts<'source> {
    pub layouter: &'source naga::proc::Layouter,
    pub layouts: Vec<BindingStackLayout>,
}

impl<'source> VisitIoBindings for CollectBindingStackLayouts<'source> {
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
    pub shim_vtable: ir::Value,
    pub shim_data: ir::Value,
    pub shim_vtable_signatures: ShimVtableSignatures,
    pub function_pointer_type: ir::Type,
}

impl ShimVtableCallCompiler {
    pub fn new(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        shim_vtable: ir::Value,
        shim_data: ir::Value,
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
        signature: ir::SigRef,
        arguments: impl IntoIterator<Item = ir::Value>,
    ) -> ir::Value {
        let vtable_offset =
            i32::try_from(vtable_offset).expect("shim vtable offset does not fit into an i32");

        let arguments = std::iter::once(self.shim_data)
            .chain(arguments)
            .collect::<Vec<_>>();

        let func_ptr = function_builder.ins().load(
            self.function_pointer_type,
            ir::MemFlags::new(),
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

pub struct ShimBuilder<'source, 'compiler> {
    context: &'compiler Context<'source>,
    pub function_builder: FunctionBuilder<'compiler>,
    shim_vtable_caller: ShimVtableCallCompiler,
    panic_block: ir::Block,
}

impl<'source, 'compiler> ShimBuilder<'source, 'compiler> {
    pub fn new(
        context: &'compiler Context<'source>,
        mut function_builder: FunctionBuilder<'compiler>,
        shim_vtable: ir::Value,
        shim_data: ir::Value,
        panic_block: ir::Block,
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
        let mut argument_values = vec![];

        let mut collect_binding_stack_layouts = CollectBindingStackLayouts {
            layouter: &self.context.layouter,
            layouts: vec![],
        };
        let mut visitor = IoBindingVisitor::new(
            &self.context.source.types,
            &mut collect_binding_stack_layouts,
        );

        let type_layout = {
            let Some(first) = arguments.next()
            else {
                return Ok((vec![], vec![]));
            };

            let argument_type = Type::from_naga(&self.context.source, first.ty)?;
            let mut type_layout = self.context.layouter[first.ty];
            visitor.visit_function_argument(first, 0);
            argument_values.push((argument_type, 0));

            for argument in arguments {
                let argument_type = Type::from_naga(&self.context.source, argument.ty)?;
                let argument_type_layout = &self.context.layouter[argument.ty];

                let offset = type_layout.alignment.round_up(type_layout.size);
                let len = argument_type_layout.size;

                type_layout.size += len;

                visitor.visit_function_argument(argument, offset);
                argument_values.push((argument_type, offset));
            }

            type_layout
        };

        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(ir::StackSlotData {
                kind: ir::StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment_log2(type_layout.alignment),
                key: None,
            });

        let len = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(type_layout.size));

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

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

        let argument_values = argument_values
            .into_iter()
            .map(|(ty, offset)| {
                Value::load(
                    self.context,
                    &mut self.function_builder,
                    ty,
                    StackLocation::from(stack_slot)
                        .with_offset(offset.try_into().expect("stack offset overflow")),
                )
            })
            .collect::<Result<Vec<Value>, Error>>()?;

        Ok((argument_values, collect_binding_stack_layouts.layouts))
    }

    pub fn compile_result_shim(
        &mut self,
        result: &naga::FunctionResult,
        value: Value,
    ) -> Result<Vec<BindingStackLayout>, Error> {
        let type_layout = self.context.layouter[result.ty];

        let mut collect_binding_stack_layouts = CollectBindingStackLayouts {
            layouter: &self.context.layouter,
            layouts: vec![],
        };
        let mut visitor = IoBindingVisitor::new(
            &self.context.source.types,
            &mut collect_binding_stack_layouts,
        );
        visitor.visit_function_result(result, 0);

        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(ir::StackSlotData {
                kind: ir::StackSlotKind::ExplicitSlot,
                size: type_layout.size,
                align_shift: alignment_log2(type_layout.alignment),
                key: None,
            });

        let len = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(type_layout.size));

        value.store(
            self.context,
            &mut self.function_builder,
            StackLocation::from(stack_slot),
        )?;

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

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
