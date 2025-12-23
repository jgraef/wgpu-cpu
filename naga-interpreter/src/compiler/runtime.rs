use std::{
    any::Any,
    fmt::Debug,
    marker::PhantomData,
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
            PointerOffset,
            StackLocation,
            Store,
            Value,
        },
    },
};

pub trait Runtime: Sized {
    type Error;

    fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), Self::Error>;
    fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), Self::Error>;
}

impl<R> Runtime for &mut R
where
    R: Runtime,
{
    type Error = R::Error;

    fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), Self::Error> {
        R::copy_inputs_to(self, target)
    }

    fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), Self::Error> {
        R::copy_outputs_from(self, source)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum RuntimeStatusCode {
    Abort = 0,
    Ok = 1,
}

#[derive(derive_more::Debug)]
pub enum AbortPayload<E> {
    RuntimeError(E),
    Panic(#[debug(skip)] Box<dyn Any + Send + 'static>),
}

#[derive(Debug)]
pub struct RuntimeContextData<R>
where
    R: Runtime,
{
    pub runtime: R,
    pub abort_payload: Option<AbortPayload<R::Error>>,
}

impl<R> RuntimeContextData<R>
where
    R: Runtime,
{
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            abort_payload: None,
        }
    }

    pub fn into_inner(self) -> Result<R, R::Error> {
        if let Some(abort_payload) = self.abort_payload {
            match abort_payload {
                AbortPayload::RuntimeError(runtime_error) => Err(runtime_error),
                AbortPayload::Panic(payload) => {
                    std::panic::resume_unwind(payload);
                }
            }
        }
        else {
            Ok(self.runtime)
        }
    }

    /// # Safety
    ///
    /// The provided `*mut DynRuntimeContextData` must correspond to a valid
    /// `&mut Self`.
    pub unsafe fn from_pointer_mut<'a>(pointer: *mut DynRuntimeContextData) -> &'a mut Self {
        unsafe { &mut *(pointer as *mut Self) }
    }

    pub fn as_pointer_mut(&mut self) -> *mut DynRuntimeContextData {
        self as *mut Self as *mut DynRuntimeContextData
    }

    pub fn with_runtime(
        &mut self,
        mut f: impl FnMut(&mut R) -> Result<(), R::Error>,
    ) -> RuntimeStatusCode {
        // any panics in the bindings implementations will be catched here and
        // propagated manually to where the entry point was called
        match std::panic::catch_unwind(AssertUnwindSafe(|| f(&mut self.runtime))) {
            Ok(Ok(())) => RuntimeStatusCode::Ok,
            Ok(Err(runtime_error)) => {
                self.abort_payload = Some(AbortPayload::RuntimeError(runtime_error));
                RuntimeStatusCode::Abort
            }
            Err(panic) => {
                self.abort_payload = Some(AbortPayload::Panic(panic));
                RuntimeStatusCode::Abort
            }
        }
    }
}

/// Type representing the struct pointed to by an opaque runtime data pointer.
///
/// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub struct DynRuntimeContextData {
    _data: (),
    _marker: PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

impl Debug for DynRuntimeContextData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DynRuntimeContextData")
            .field(&(self as *const _ as *const u8))
            .finish()
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct RuntimeContext {
    /// Pointer to the underlying runtime context struct.
    pub data: *mut DynRuntimeContextData,

    /// Contains function pointers for the runtime API
    pub vtable: RuntimeVtable,
}

impl RuntimeContext {
    pub fn new<R>(data: &mut RuntimeContextData<R>) -> Self
    where
        R: Runtime,
    {
        Self {
            data: data.as_pointer_mut(),
            vtable: RuntimeVtable::new::<R>(),
        }
    }
}

/// Vtable containing function pointers for the runtime API.
///
/// These functions can be called from the compiled shader code. The first
/// argument is always a pointer to runtime context struct.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct RuntimeVtable {
    /// Copy shader parameters to stack.
    ///
    /// The runtime entry point will allocate stack space for all arguments to
    /// the shader and call this function with it. It can then call the actual
    /// shader entry point with the arguments.
    ///
    /// The provided arguments are:
    /// 1. Pointer to the runtime context data
    /// 2. Pointer to the allocated stack space
    /// 3. Size of allocated stack space
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the arguments on the stack. (see [`BindingStackLayout`]).
    pub copy_inputs_to:
        unsafe extern "C" fn(*mut DynRuntimeContextData, *mut u8, usize) -> RuntimeStatusCode,

    /// Copy shader return value from stack
    ///
    /// The runtime entry point will write the return value of the shader to a
    /// location on its stack and call this function with it.
    ///
    /// The provided arguments are:
    /// 1. Pointer to the runtime context data
    /// 2. Pointer to the stack space with the return value
    /// 3. Size of the stack space with the return value
    ///
    /// The [`Runtime`] implementation is responsible for kowing the correct
    /// layout of the return value on the stack. (see [`BindingStackLayout`]).
    pub copy_outputs_from:
        unsafe extern "C" fn(*mut DynRuntimeContextData, *const u8, usize) -> RuntimeStatusCode,
}

impl RuntimeVtable {
    pub const fn new<R>() -> Self
    where
        R: Runtime,
    {
        unsafe extern "C" fn copy_inputs_to<R>(
            data: *mut DynRuntimeContextData,
            target: *mut u8,
            len: usize,
        ) -> RuntimeStatusCode
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut R`
                RuntimeContextData::<R>::from_pointer_mut(data)
            };

            let target = std::ptr::slice_from_raw_parts_mut(target, len);
            let target = unsafe {
                // SAFETY: The `target` pointer with `len` length must correspond to a valid
                // `&mut [u8]` produced by the compiled code
                &mut *target
            };

            data.with_runtime(|runtime| runtime.copy_inputs_to(target))
        }

        unsafe extern "C" fn copy_outputs_from<R>(
            data: *mut DynRuntimeContextData,
            source: *const u8,
            len: usize,
        ) -> RuntimeStatusCode
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut R`
                RuntimeContextData::<R>::from_pointer_mut(data)
            };

            let source = std::ptr::slice_from_raw_parts(source, len);
            let source = unsafe {
                // SAFETY: The `source` pointer with `len` length must correspond to a valid
                // `&[u8]` produced by the compiled code
                &*source
            };

            data.with_runtime(|runtime| runtime.copy_outputs_from(source))
        }

        RuntimeVtable {
            copy_inputs_to: copy_inputs_to::<R>,
            copy_outputs_from: copy_outputs_from::<R>,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeMethodSignatures {
    pub copy_inputs_to: ir::SigRef,
    pub copy_outputs_from: ir::SigRef,
}

impl RuntimeMethodSignatures {
    pub fn new(function_builder: &mut FunctionBuilder, context: &Context) -> Self {
        let data = ir::AbiParam::new(context.pointer_type());

        let copy_inputs_to = function_builder.import_signature(ir::Signature {
            params: vec![
                data,
                ir::AbiParam::new(context.pointer_type()),
                ir::AbiParam::new(context.pointer_type()),
            ],
            returns: vec![ir::AbiParam::new(ir::types::I32)],
            call_conv: context.target_config.default_call_conv,
        });

        let copy_outputs_from = function_builder.import_signature(ir::Signature {
            params: vec![
                data,
                ir::AbiParam::new(context.pointer_type()),
                ir::AbiParam::new(context.pointer_type()),
            ],
            returns: vec![ir::AbiParam::new(ir::types::I32)],
            call_conv: context.target_config.default_call_conv,
        });

        Self {
            copy_inputs_to,
            copy_outputs_from,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RuntimeContextValue {
    pub runtime_pointer: ir::Value,
    pub method_signatures: RuntimeMethodSignatures,
}

impl RuntimeContextValue {
    pub fn new(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        runtime_pointer: ir::Value,
    ) -> Self {
        let method_signatures = RuntimeMethodSignatures::new(function_builder, context);
        Self {
            runtime_pointer,
            method_signatures,
        }
    }
}

/// Emits code to call a runtime method
///
/// Can be created with the [`runtime_method`] macro.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeMethod<'a> {
    /// The IR value for the runtime.
    ///
    /// Also contains all the runtime API method signatures.
    pub runtime_value: &'a RuntimeContextValue,

    /// Offset of the function pointer in the [`RuntimeVtable`].
    pub vtable_offset: i32,

    /// Function signature
    pub signature: ir::SigRef,
}

impl<'a> RuntimeMethod<'a> {
    pub fn call(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        arguments: impl IntoIterator<Item = ir::Value>,
    ) -> ir::Value {
        let data_pointer_offset: i32 = memoffset::offset_of!(RuntimeContext, data)
            .try_into()
            .expect("pointer offset overflow");
        let vtable_offset: i32 = memoffset::offset_of!(RuntimeContext, vtable)
            .try_into()
            .expect("pointer offset overflow");

        // todo: these can probably be relaxed
        let memory_flags = ir::MemFlags::new();

        let data_pointer = function_builder.ins().load(
            context.pointer_type(),
            memory_flags,
            self.runtime_value.runtime_pointer,
            data_pointer_offset,
        );

        let function_pointer = function_builder.ins().load(
            context.pointer_type(),
            memory_flags,
            self.runtime_value.runtime_pointer,
            self.vtable_offset + vtable_offset,
        );

        let arguments = std::iter::once(data_pointer)
            .chain(arguments)
            .collect::<Vec<_>>();

        let inst =
            function_builder
                .ins()
                .call_indirect(self.signature, function_pointer, &arguments);
        let results = function_builder.inst_results(inst);
        assert_eq!(results.len(), 1);
        results[0]
    }
}

/// Creates a [`RuntimeMethod`] struct which can be used to emit code to call a
/// runtime API method.
///
/// The first argument is the IR value for the pointer to the [`RuntimeContext`]
/// struct.
macro_rules! runtime_method {
    ($runtime:expr, $func:ident) => {{
        let vtable_offset = memoffset::offset_of!(RuntimeVtable, $func);
        let vtable_offset = i32::try_from(vtable_offset).expect("runtime vtable offset overflow");
        let signature = $runtime.method_signatures.$func;
        RuntimeMethod {
            runtime_value: &$runtime,
            vtable_offset,
            signature,
        }
    }};
}

pub struct RuntimeEntryPointBuilder<'source, 'compiler> {
    pub context: &'compiler Context<'source>,
    pub function_builder: FunctionBuilder<'compiler>,
    pub runtime: RuntimeContextValue,
    pub panic_block: ir::Block,
}

impl<'source, 'compiler> RuntimeEntryPointBuilder<'source, 'compiler> {
    pub fn new(
        context: &'compiler Context<'source>,
        function_builder: FunctionBuilder<'compiler>,
        runtime: RuntimeContextValue,
        panic_block: ir::Block,
    ) -> Self {
        Self {
            context,
            function_builder,
            runtime,
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

        let result = runtime_method!(self.runtime, copy_inputs_to).call(
            self.context,
            &mut self.function_builder,
            [stack_slot_pointer, len],
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

        let result = runtime_method!(self.runtime, copy_outputs_from).call(
            self.context,
            &mut self.function_builder,
            [stack_slot_pointer, len],
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

#[derive(Debug, thiserror::Error)]
pub enum DefaultRuntimeError {
    // todo
}

#[derive(Debug)]
pub struct DefaultRuntime<'layout, I, O> {
    pub input: I,
    pub input_layout: &'layout [BindingStackLayout],
    pub output: O,
    pub output_layout: &'layout [BindingStackLayout],
}

impl<'layout, I, O> Runtime for DefaultRuntime<'layout, I, O>
where
    I: ShaderInput,
    O: ShaderOutput,
{
    type Error = DefaultRuntimeError;

    fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), DefaultRuntimeError> {
        for layout in self.input_layout {
            self.input.write_into(
                &layout.binding,
                &layout.ty,
                &mut target[layout.offset..][..layout.size],
            );
        }

        Ok(())
    }

    fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), DefaultRuntimeError> {
        for layout in self.output_layout {
            self.output.read_from(
                &layout.binding,
                &layout.ty,
                &source[layout.offset..][..layout.size],
            );
        }

        Ok(())
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
