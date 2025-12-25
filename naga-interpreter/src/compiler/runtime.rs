use std::{
    any::Any,
    fmt::Debug,
    marker::PhantomData,
    panic::AssertUnwindSafe,
};

use cranelift_codegen::ir::{
    self,
    BlockArg,
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
        function::ABORT_CODE_TYPE,
        types::Type,
        util::alignment_log2,
        value::{
            AsIrValues,
            Load,
            Pointer,
            PointerOffset,
            PointerRange,
            StackLocation,
            Value,
        },
        variable::PrivateMemoryLayout,
    },
};

/// Runtime API
///
/// This is used by the generated code to access shader inputs/outputs,
/// initialize global variables, etc.
///
/// The runtime implementation is responsible for knowing the layouts of the
/// different buffers, but it'll get passed a slice that is safe to access.
///
/// The runtime can return errors or panic. Both will be handled properly and
/// propagated to the call of the entry point (TODO: propagation itself doesn't
/// work right now).
pub trait Runtime: Sized {
    /// Error type returned by the runtme.
    ///
    /// When the runtime returns an error it is stored in [`RuntimeData`] and
    /// the shader aborts execution. When it returns from the entrypoint the
    /// call to [`EntryPoint::function`](super::product::EntryPoint::function)
    /// will return that error.
    type Error;

    /// Copy shader parameters to stack.
    ///
    /// The runtime entry point will allocate stack space for all arguments to
    /// the shader and call this function with it. It can then call the actual
    /// shader entry point with the arguments.
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the arguments on the stack. (see [`BindingStackLayout`]).
    ///
    /// Note that this method is only called once per shader invocation on
    /// startup. It's fine to drop any references to the input data after this
    /// is called.
    fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), Self::Error>;

    /// Copy shader return value from stack.
    ///
    /// The runtime entry point will write the return value of the shader to a
    /// location on its stack and call this function with it.
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the return value on the stack. (see [`BindingStackLayout`]).
    fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), Self::Error>;

    /// Initialize the global variables.
    ///
    /// The runtime entry point will allocate space for its global variables on
    /// its stack and call this function to populate these with initial values.
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the global variables. (see
    /// [`GlobalVariableLayout`](super::variable::GlobalVariableLayout)).
    ///
    /// # TODO
    ///
    /// Global variables usually live in the [private address
    /// space](naga::AddressSpace::Private), but they might also live in others,
    /// such as the [work group address space](naga::AddressSpace::WorkGroup).
    /// Only private global variables are implemented right now.
    fn initialize_global_variables(&mut self, private_data: &mut [u8]) -> Result<(), Self::Error>;

    /// # TODO
    ///
    /// Is this usable and safe?
    fn buffer(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error>;

    /// # TODO
    ///
    /// Is this usable and safe?
    fn buffer_mut(&mut self, binding: naga::ResourceBinding) -> Result<&mut [u8], Self::Error>;
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

    fn initialize_global_variables(&mut self, private_data: &mut [u8]) -> Result<(), Self::Error> {
        R::initialize_global_variables(self, private_data)
    }

    fn buffer(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
        R::buffer(self, binding)
    }

    fn buffer_mut(&mut self, binding: naga::ResourceBinding) -> Result<&mut [u8], Self::Error> {
        R::buffer_mut(self, binding)
    }
}

/// Result code returned by a runtime API method to the shader.
///
/// Implementors of [`Runtime`] can ignore this, because this is returned by the
/// code calling into the [`Runtime`] implementation.
///
/// # Note
///
/// This must be synchronized with the code generated in
/// [`RuntimeMethod::call`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i8)]
pub enum RuntimeResult {
    Abort = 1,
    Ok = 0,
}

/// Payload propagated to the entry point call site when the shader aborts
/// execution.
///
/// This is stored in [`RuntimeData`] when the [`Runtime`] implementation
/// returns an [`Err`] or panics.
#[derive(derive_more::Debug)]
pub enum AbortPayload<E> {
    /// Call to a [`Runtime`] method returned an error.
    RuntimeError(E),

    /// Call to a [`Runtime`] method panicked.
    Panic(#[debug(skip)] Box<dyn Any + Send + 'static>),

    /// The shader executed a [`Kill statement`](naga::Statement::Kill)
    Kill,
}

/// All of the runtime data.
///
/// This contains the runtime itself and a place to store an [`AbortPayload`].
#[derive(Debug)]
pub struct RuntimeData<R>
where
    R: Runtime,
{
    pub runtime: R,
    pub abort_payload: Option<AbortPayload<R::Error>>,
}

impl<R> RuntimeData<R>
where
    R: Runtime,
{
    pub fn new(runtime: R) -> Self {
        Self {
            runtime,
            abort_payload: None,
        }
    }

    /// # Safety
    ///
    /// The provided `*mut DynRuntimeData` must correspond to a valid `&mut
    /// Self`.
    pub unsafe fn from_pointer_mut<'a>(pointer: *mut DynRuntimeData) -> &'a mut Self {
        unsafe { &mut *(pointer as *mut Self) }
    }

    pub fn as_pointer_mut(&mut self) -> *mut DynRuntimeData {
        self as *mut Self as *mut DynRuntimeData
    }

    /// Calls the provided closure with the runtime and handles errors and
    /// panics.
    ///
    /// This is used by the implementations for the methods in the
    /// [`RuntimeVtable`].
    pub fn with_runtime(
        &mut self,
        mut f: impl FnMut(&mut R) -> Result<(), R::Error>,
    ) -> RuntimeResult {
        // rust can't unwind when called from external code, so we have to handle this
        // ourselves. any panics and errors in the runtime implementation will
        // be catched here and stored in the RuntimeData. then a RuntimeResult is
        // returned to indicate to the caller (RuntimeMethod::call) to abort the shader
        // program.
        match std::panic::catch_unwind(AssertUnwindSafe(|| f(&mut self.runtime))) {
            Ok(Ok(())) => RuntimeResult::Ok,
            Ok(Err(runtime_error)) => {
                self.abort_payload = Some(AbortPayload::RuntimeError(runtime_error));
                RuntimeResult::Abort
            }
            Err(panic) => {
                self.abort_payload = Some(AbortPayload::Panic(panic));
                RuntimeResult::Abort
            }
        }
    }
}

/// Type representing the struct pointed to by an opaque runtime data pointer.
///
/// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub struct DynRuntimeData {
    _data: (),
    _marker: PhantomData<(*mut u8, std::marker::PhantomPinned)>,
}

/// A pointer to this struct is passed through the compiled shader code. This
/// allows the compiled code to always call the runtime and it provides some
/// auxilary "global" storage for e.g. private memory.
#[derive(Debug)]
#[repr(C)]
pub struct RuntimeContext {
    /// Opaque pointer to the underlying [`RuntimeData`].
    pub runtime: *mut DynRuntimeData,

    /// Contains function pointers for the runtime API.
    pub vtable: RuntimeVtable,

    /// The compiled shader will store a reference to its stack allocated
    /// private data here.
    pub private_memory: *mut u8,
}

impl RuntimeContext {
    /// Creates a [`RuntimeContext`] for the specific [`Runtime`].
    ///
    /// This produces the opaque runtime pointer and vtable. It also initializes
    /// the `private_memory` pointer to `null`.
    ///
    /// # Safety
    ///
    /// Since this struct contains a pointer to the provided [`RuntimeData`],
    /// the runtime data must be valid for as long as this might get used. The
    /// method is safe because a shader can only be called with a runtime after
    /// agreeing to all the safety contracts.
    pub fn new<R>(data: &mut RuntimeData<R>) -> Self
    where
        R: Runtime,
    {
        Self {
            runtime: data.as_pointer_mut(),
            vtable: RuntimeVtable::new::<R>(),
            private_memory: std::ptr::null_mut(),
        }
    }
}

/// Vtable containing function pointers for the runtime API.
///
/// These functions can be called from the compiled shader code. The first
/// argument is always a pointer to runtime context struct.
///
/// These functions correspond to methods of the [`Runtime`] trait and they are
/// light wrappers that produce safe variants for passed-in pointers and handle
/// errors and panics.
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
    pub copy_inputs_to: unsafe extern "C" fn(*mut DynRuntimeData, *mut u8, usize) -> RuntimeResult,

    /// Copy shader return value from stack.
    ///
    /// The runtime entry point will write the return value of the shader to a
    /// location on its stack and call this function with it.
    ///
    /// The provided arguments are:
    /// 1. Pointer to the runtime context data
    /// 2. Pointer to the stack space with the return value
    /// 3. Size of the stack space with the return value
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the return value on the stack. (see [`BindingStackLayout`]).
    pub copy_outputs_from:
        unsafe extern "C" fn(*mut DynRuntimeData, *const u8, usize) -> RuntimeResult,

    /// Initialize the global variables.
    ///
    /// The runtime entry point will allocate space for its global variables on
    /// its stack and call this function to populate these with initial values.
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the global variables. (see
    /// [`GlobalVariableLayout`](super::variable::GlobalVariableLayout)).
    pub initialize_global_variables:
        unsafe extern "C" fn(*mut DynRuntimeData, *mut u8, usize) -> RuntimeResult,

    /// Terminates shader execution
    ///
    /// This is called by the shader for [`Kill
    /// statements`](naga::Statement::Kill).
    ///
    /// This API method doesn't call into the [`Runtime`] implementation, but
    /// only sets the abort payload and returns a [`RuntimeResult`] telling the
    /// shader to abort execution.
    pub kill: unsafe extern "C" fn(*mut DynRuntimeData) -> RuntimeResult,

    pub buffer: unsafe extern "C" fn(
        *mut DynRuntimeData,
        group: u32,
        binding: u32,
        access: u32,
        pointer_out: &mut *const u8,
        len_out: &mut usize,
    ) -> RuntimeResult,
}

impl RuntimeVtable {
    /// Creates a [`RuntimeVtable`] for a type implementing [`Runtime`].
    ///
    /// This is a safe way to create a vtable for a [`Runtime`] implementation.
    ///
    /// Note that the method is `const` and thus a `&'static RuntimeVtable` can
    /// be obtained.
    pub const fn new<R>() -> Self
    where
        R: Runtime,
    {
        unsafe extern "C" fn copy_inputs_to<R>(
            data: *mut DynRuntimeData,
            target: *mut u8,
            len: usize,
        ) -> RuntimeResult
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
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
            data: *mut DynRuntimeData,
            source: *const u8,
            len: usize,
        ) -> RuntimeResult
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            let source = std::ptr::slice_from_raw_parts(source, len);
            let source = unsafe {
                // SAFETY: The `source` pointer with `len` length must correspond to a valid
                // `&[u8]` produced by the compiled code
                &*source
            };

            data.with_runtime(|runtime| runtime.copy_outputs_from(source))
        }

        unsafe extern "C" fn initialize_global_variables<R>(
            data: *mut DynRuntimeData,
            target: *mut u8,
            len: usize,
        ) -> RuntimeResult
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            let target = std::ptr::slice_from_raw_parts_mut(target, len);
            let target = unsafe {
                // SAFETY: The `target` pointer with `len` length must correspond to a valid
                // `&mut [u8]` produced by the compiled code
                &mut *target
            };

            data.with_runtime(|runtime| runtime.initialize_global_variables(target))
        }

        unsafe extern "C" fn kill<R>(data: *mut DynRuntimeData) -> RuntimeResult
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            data.abort_payload = Some(AbortPayload::Kill);
            RuntimeResult::Abort
        }

        unsafe extern "C" fn buffer<R>(
            data: *mut DynRuntimeData,
            group: u32,
            binding: u32,
            access: u32,
            pointer_out: &mut *const u8,
            len_out: &mut usize,
        ) -> RuntimeResult
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            data.with_runtime(move |runtime| {
                let access = naga::StorageAccess::from_bits_retain(access);
                let binding = naga::ResourceBinding { group, binding };

                let (pointer, len) = if access.contains(naga::StorageAccess::STORE) {
                    let buffer = runtime.buffer_mut(binding)?;
                    (buffer.as_mut_ptr() as *const u8, buffer.len())
                }
                else if access.contains(naga::StorageAccess::LOAD) {
                    let buffer = runtime.buffer(binding)?;
                    (buffer.as_ptr(), buffer.len())
                }
                else {
                    panic!("Shader requested buffer access that is neither store or load");
                };

                *pointer_out = pointer;
                *len_out = len;

                Ok(())
            })
        }

        RuntimeVtable {
            copy_inputs_to: copy_inputs_to::<R>,
            copy_outputs_from: copy_outputs_from::<R>,
            initialize_global_variables: initialize_global_variables::<R>,
            kill: kill::<R>,
            buffer: buffer::<R>,
        }
    }
}

/// Signatures for runtime API methods.
///
/// This is used by the code generation.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeMethodSignatures {
    pub copy_inputs_to: ir::SigRef,
    pub copy_outputs_from: ir::SigRef,
    pub initialize_global_variables: ir::SigRef,
    pub kill: ir::SigRef,
    pub buffer: ir::SigRef,
}

impl RuntimeMethodSignatures {
    pub fn new(function_builder: &mut FunctionBuilder, context: &Context) -> Self {
        let pointer_param = ir::AbiParam::new(context.pointer_type());
        let context_param = pointer_param;
        let result = ir::AbiParam::new(ABORT_CODE_TYPE);

        let copy_inputs_to = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // stack slot pointer
                pointer_param,
                // stack slot size
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let copy_outputs_from = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // stack slot pointer
                pointer_param,
                // stack slot size
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let initialize_global_variables = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // stack slot pointer
                pointer_param,
                // stack slot size
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let kill = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let buffer = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // binding group
                ir::AbiParam::new(ir::types::I32),
                // binding index
                ir::AbiParam::new(ir::types::I32),
                // access flags
                ir::AbiParam::new(ir::types::I32),
                // buffer pointer, return by reference
                pointer_param,
                // buffer size, return by reference
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        Self {
            copy_inputs_to,
            copy_outputs_from,
            initialize_global_variables,
            kill,
            buffer,
        }
    }
}

/// Creates a [`RuntimeMethod`] struct which can be used to emit code to call a
/// runtime API method.
///
/// The first argument is the [`RuntimeContextValue`] containing the runtime
/// context pointer.
///
/// The second argument is the API method name (as an identifier)
macro_rules! runtime_method {
    ($runtime:expr, $func:ident) => {{
        let vtable_offset = memoffset::offset_of!(RuntimeVtable, $func);
        let vtable_offset = i32::try_from(vtable_offset).expect("runtime vtable offset overflow");
        let signature = $runtime.signatures.$func;
        RuntimeMethod {
            runtime_value: &$runtime,
            vtable_offset,
            signature,
        }
    }};
}

/// A [`ir::Value`] representing the runtime context pointer.
///
/// You can obtain a [`RuntimeMethod`] with the [`runtime_method`] macro and use
/// that to generate runtime API calls.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeContextValue {
    pub pointer: ir::Value,
    pub signatures: RuntimeMethodSignatures,
    pub memory_flags: ir::MemFlags,
}

impl RuntimeContextValue {
    pub fn new(
        context: &Context,
        function_builder: &mut FunctionBuilder,
        runtime_pointer: ir::Value,
    ) -> Self {
        // todo: these can probably be relaxed
        let memory_flags = ir::MemFlags::new();

        Self {
            pointer: runtime_pointer,
            signatures: RuntimeMethodSignatures::new(function_builder, context),
            memory_flags,
        }
    }

    /// Generates code to fetch the pointer to private memory.
    pub fn private_memory(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
    ) -> ir::Value {
        let offset: i32 = memoffset::offset_of!(RuntimeContext, private_memory)
            .try_into()
            .expect("pointer offset overflow");

        function_builder.ins().load(
            context.pointer_type(),
            self.memory_flags,
            self.pointer,
            offset,
        )
    }

    /// Stores the private memory pointer in the runtime context struct.
    pub fn stash_private_memory_pointer(
        &self,
        function_builder: &mut FunctionBuilder,
        pointer: ir::Value,
    ) {
        let offset: i32 = memoffset::offset_of!(RuntimeContext, private_memory)
            .try_into()
            .expect("pointer offset overflow");

        function_builder
            .ins()
            .store(self.memory_flags, pointer, self.pointer, offset);
    }

    /// Generates a call to the kill runtime API method which kills shader
    /// execution.
    pub fn kill(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        abort_block: ir::Block,
    ) {
        runtime_method!(self, kill).call(context, function_builder, [], abort_block);
    }

    pub fn buffer(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        resource_binding: naga::ir::ResourceBinding,
        access: naga::StorageAccess,
        abort_block: ir::Block,
    ) -> PointerRange<ir::Value> {
        // values for arguments
        let group = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.group));
        let binding = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.binding));
        let access = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(access.bits()));

        // the runtime api method will return the pointer by reference.
        let pointer_type = context.pointer_type();
        let pointer_out_stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
            kind: ir::StackSlotKind::ExplicitSlot,
            size: pointer_type.bytes(),
            align_shift: 1, // i think cranelift will figure it out
            key: None,
        });
        let pointer_out =
            function_builder
                .ins()
                .stack_addr(pointer_type, pointer_out_stack_slot, 0);

        let len_out_stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
            kind: ir::StackSlotKind::ExplicitSlot,
            size: pointer_type.bytes(),
            align_shift: 1, // i think cranelift will figure it out
            key: None,
        });
        let len_out = function_builder
            .ins()
            .stack_addr(pointer_type, len_out_stack_slot, 0);

        // call runtime
        runtime_method!(self, buffer).call(
            context,
            function_builder,
            [group, binding, access, pointer_out, len_out],
            abort_block,
        );

        // load returned pointer and len
        let pointer = function_builder
            .ins()
            .stack_load(pointer_type, pointer_out_stack_slot, 0);
        let len = function_builder
            .ins()
            .stack_load(pointer_type, len_out_stack_slot, 0);

        // we probably need to think about what memory flags to use here (depends on
        // address space i think)
        PointerRange {
            pointer: Pointer {
                value: pointer,
                memory_flags: self.memory_flags,
                offset: 0,
            },
            len,
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
        abort_block: ir::Block,
    ) {
        let data_pointer_offset: i32 = memoffset::offset_of!(RuntimeContext, runtime)
            .try_into()
            .expect("pointer offset overflow");
        let vtable_offset: i32 = memoffset::offset_of!(RuntimeContext, vtable)
            .try_into()
            .expect("pointer offset overflow");

        let data_pointer = function_builder.ins().load(
            context.pointer_type(),
            self.runtime_value.memory_flags,
            self.runtime_value.pointer,
            data_pointer_offset,
        );

        let function_pointer = function_builder.ins().load(
            context.pointer_type(),
            self.runtime_value.memory_flags,
            self.runtime_value.pointer,
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
        let abort_code = results[0];

        // check abort flag
        let continue_block = function_builder.create_block();
        function_builder.ins().brif(
            abort_code,
            abort_block,
            [&BlockArg::Value(abort_code)],
            continue_block,
            [],
        );
        function_builder.seal_block(continue_block);
        function_builder.switch_to_block(continue_block);
    }
}

/// Helper to build a shim around a shader entry point.
///
/// This shim is needed to fetch inputs, initialize global variables and write
/// back outputs.
pub struct RuntimeEntryPointBuilder<'source, 'compiler> {
    pub context: &'compiler Context<'source>,
    pub function_builder: FunctionBuilder<'compiler>,
    pub runtime: RuntimeContextValue,
}

impl<'source, 'compiler> RuntimeEntryPointBuilder<'source, 'compiler> {
    pub fn new(
        context: &'compiler Context<'source>,
        function_builder: FunctionBuilder<'compiler>,
        runtime: RuntimeContextValue,
    ) -> Self {
        Self {
            context,
            function_builder,
            runtime,
        }
    }

    pub fn load_arguments(
        &mut self,
        arguments: &[naga::FunctionArgument],
        argument_values: &mut Vec<ir::Value>,
        abort_block: ir::Block,
    ) -> Result<Vec<BindingStackLayout>, Error> {
        let mut arguments = arguments.iter();
        let mut argument_slots = vec![];

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
                return Ok(vec![]);
            };

            let argument_type = Type::from_naga(&self.context.source, first.ty)?;
            let mut type_layout = self.context.layouter[first.ty];
            visitor.visit_function_argument(first, 0);
            argument_slots.push((argument_type, 0));

            for argument in arguments {
                let argument_type = Type::from_naga(&self.context.source, argument.ty)?;
                let argument_type_layout = &self.context.layouter[argument.ty];

                let offset = type_layout.alignment.round_up(type_layout.size);
                let len = argument_type_layout.size;

                type_layout.size += len;

                visitor.visit_function_argument(argument, offset);
                argument_slots.push((argument_type, offset));
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

        runtime_method!(self.runtime, copy_inputs_to).call(
            self.context,
            &mut self.function_builder,
            [stack_slot_pointer, len],
            abort_block,
        );

        for (ty, offset) in argument_slots {
            let value = Value::load(
                self.context,
                &mut self.function_builder,
                ty,
                StackLocation::from(stack_slot)
                    .with_offset(offset.try_into().expect("stack offset overflow")),
            )?;
            argument_values.extend(value.as_ir_values());
        }

        Ok(collect_binding_stack_layouts.layouts)
    }

    pub fn allocate_result(
        &mut self,
        result: &naga::FunctionResult,
    ) -> Result<(PointerRange<u32>, Vec<BindingStackLayout>), Error> {
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

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        Ok((
            PointerRange {
                pointer: Pointer {
                    value: stack_slot_pointer,
                    memory_flags: ir::MemFlags::trusted(),
                    offset: 0,
                },
                len: type_layout.size,
            },
            collect_binding_stack_layouts.layouts,
        ))
    }

    pub fn pass_result_to_runtime(&mut self, pointer: PointerRange<u32>, abort_block: ir::Block) {
        assert_eq!(pointer.pointer.offset, 0);

        let len = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(pointer.len));

        runtime_method!(self.runtime, copy_outputs_from).call(
            self.context,
            &mut self.function_builder,
            [pointer.pointer.value, len],
            abort_block,
        );
    }

    pub fn compile_private_data_initialization(
        &mut self,
        private_memory_stack_slot_data: ir::StackSlotData,
        abort_block: ir::Block,
    ) -> Result<(), Error> {
        let size = private_memory_stack_slot_data.size;

        let stack_slot = self
            .function_builder
            .create_sized_stack_slot(private_memory_stack_slot_data);

        let stack_slot_pointer =
            self.function_builder
                .ins()
                .stack_addr(self.context.pointer_type(), stack_slot, 0);

        let size = self
            .function_builder
            .ins()
            .iconst(self.context.pointer_type(), i64::from(size));

        runtime_method!(self.runtime, initialize_global_variables).call(
            self.context,
            &mut self.function_builder,
            [stack_slot_pointer, size],
            abort_block,
        );

        self.runtime
            .stash_private_memory_pointer(&mut self.function_builder, stack_slot_pointer);

        Ok(())
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

    pub private_memory_layout: &'layout PrivateMemoryLayout,
}

impl<'layout, I, O> DefaultRuntime<'layout, I, O> {
    pub fn new(
        input: I,
        input_layout: &'layout [BindingStackLayout],
        output: O,
        output_layout: &'layout [BindingStackLayout],
        private_memory_layout: &'layout PrivateMemoryLayout,
    ) -> Self {
        Self {
            input,
            input_layout,
            output,
            output_layout,
            private_memory_layout,
        }
    }
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

    fn initialize_global_variables(&mut self, private_data: &mut [u8]) -> Result<(), Self::Error> {
        tracing::debug!("initialize global variables: {private_data:p}");

        let initialized = &self.private_memory_layout.initialized;
        private_data[..initialized.len()].copy_from_slice(initialized);
        private_data[initialized.len()..].fill(0);
        Ok(())
    }

    fn buffer(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
        todo!("buffer: {binding:?}");
    }

    fn buffer_mut(&mut self, binding: naga::ResourceBinding) -> Result<&mut [u8], Self::Error> {
        todo!("buffer_mut: {binding:?}");
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
