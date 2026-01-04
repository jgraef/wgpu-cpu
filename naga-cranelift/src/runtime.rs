use core::f32;
use std::{
    fmt::Debug,
    marker::PhantomData,
    panic::AssertUnwindSafe,
};

use cranelift_codegen::ir::{
    self,
    InstBuilder as _,
};
use cranelift_frontend::FunctionBuilder;

use crate::{
    Error,
    bindings::{
        BindingResources,
        IoBindingVisitor,
        ShaderInput,
        ShaderOutput,
        VisitIoBindings,
    },
    compiler::Context,
    function::{
        ABORT_CODE_TYPE,
        AbortCode,
        AbortPayload,
    },
    types::{
        FloatWidth,
        ScalarType,
        Type,
        VectorType,
    },
    util::alignment_log2,
    value::{
        AsIrValues,
        HandlePointer,
        Load,
        Pointer,
        PointerOffset,
        PointerRange,
        ScalarValue,
        StackLocation,
        Value,
        VectorValue,
    },
    variable::PrivateMemoryLayout,
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

    type Image;
    type Sampler;

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
    fn buffer_resource(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error>;

    /// # TODO
    ///
    /// Is this usable and safe?
    fn buffer_resource_mut(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&mut [u8], Self::Error>;

    fn image_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Image, Self::Error>;

    fn sampler_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Sampler, Self::Error>;

    fn image_sample(
        &mut self,
        image: &Self::Image,
        sampler: &Self::Sampler,
        gather: Option<naga::SwizzleComponent>,
        coordinate: [f32; 2],
        array_index: Option<u32>,
        offset: Option<u32>,
        level: naga::SampleLevel,
        depth_ref: Option<f32>,
        clamp_to_edge: bool,
    ) -> Result<[f32; 4], Self::Error>;
}

impl<R> Runtime for &mut R
where
    R: Runtime,
{
    type Error = R::Error;
    type Image = R::Image;
    type Sampler = R::Sampler;

    fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), Self::Error> {
        R::copy_inputs_to(self, target)
    }

    fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), Self::Error> {
        R::copy_outputs_from(self, source)
    }

    fn initialize_global_variables(&mut self, private_data: &mut [u8]) -> Result<(), Self::Error> {
        R::initialize_global_variables(self, private_data)
    }

    fn buffer_resource(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
        R::buffer_resource(self, binding)
    }

    fn buffer_resource_mut(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&mut [u8], Self::Error> {
        R::buffer_resource_mut(self, binding)
    }

    fn image_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Image, Self::Error> {
        R::image_resource(self, binding)
    }

    fn sampler_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Sampler, Self::Error> {
        R::sampler_resource(self, binding)
    }

    fn image_sample(
        &mut self,
        image: &Self::Image,
        sampler: &Self::Sampler,
        gather: Option<naga::SwizzleComponent>,
        coordinate: [f32; 2],
        array_index: Option<u32>,
        offset: Option<u32>,
        level: naga::SampleLevel,
        depth_ref: Option<f32>,
        clamp_to_edge: bool,
    ) -> Result<[f32; 4], Self::Error> {
        R::image_sample(
            self,
            image,
            sampler,
            gather,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            clamp_to_edge,
        )
    }
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
    pub fn with_runtime(&mut self, mut f: impl FnMut(&mut R) -> Result<(), R::Error>) -> AbortCode {
        // rust can't unwind when called from external code, so we have to handle this
        // ourselves. any panics and errors in the runtime implementation will
        // be catched here and stored in the RuntimeData. then a RuntimeResult is
        // returned to indicate to the caller (RuntimeMethod::call) to abort the shader
        // program.
        match std::panic::catch_unwind(AssertUnwindSafe(|| f(&mut self.runtime))) {
            Ok(Ok(())) => AbortCode::Ok,
            Ok(Err(runtime_error)) => {
                self.abort_payload = Some(AbortPayload::RuntimeError(runtime_error));
                AbortCode::RuntimeError
            }
            Err(panic) => {
                self.abort_payload = Some(AbortPayload::RuntimePanic(panic));
                AbortCode::RuntimePanic
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

/// Type representing the struct pointed to by an opaque image pointer.
///
/// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub struct Image {
    _data: (),
    _marker: PhantomData<(*mut u8, std::marker::PhantomPinned)>,
}

/// Type representing the struct pointed to by an opaque sampler pointer.
///
/// https://doc.rust-lang.org/nomicon/ffi.html#representing-opaque-structs
#[repr(C)]
pub struct Sampler {
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
    pub copy_inputs_to: unsafe extern "C" fn(*mut DynRuntimeData, *mut u8, usize) -> AbortCode,

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
    pub copy_outputs_from: unsafe extern "C" fn(*mut DynRuntimeData, *const u8, usize) -> AbortCode,

    /// Initialize the global variables.
    ///
    /// The runtime entry point will allocate space for its global variables on
    /// its stack and call this function to populate these with initial values.
    ///
    /// The [`Runtime`] implementation is responsible for knowing the correct
    /// layout of the global variables. (see
    /// [`GlobalVariableLayout`](super::variable::GlobalVariableLayout)).
    pub initialize_global_variables:
        unsafe extern "C" fn(*mut DynRuntimeData, *mut u8, usize) -> AbortCode,

    pub buffer_resource: unsafe extern "C" fn(
        *mut DynRuntimeData,
        group: u32,
        binding: u32,
        access: u32,
        pointer_out: &mut *const u8,
        len_out: &mut usize,
    ) -> AbortCode,

    pub image_resource: unsafe extern "C" fn(
        *mut DynRuntimeData,
        group: u32,
        binding: u32,
        pointer_out: &mut *const Image,
    ) -> AbortCode,

    pub sampler_resource: unsafe extern "C" fn(
        *mut DynRuntimeData,
        group: u32,
        binding: u32,
        pointer_out: &mut *const Sampler,
    ) -> AbortCode,

    pub image_sample: unsafe extern "C" fn(
        *mut DynRuntimeData,
        image: *const Image,
        sampler: *const Sampler,
        gather: u32,
        u: f32,
        v: f32,
        array_index: u32,
        offset: u32,
        depth_ref: f32,
        clamp_to_edge: u8,
        texel_out: &mut [f32; 4],
    ) -> AbortCode,
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
        ) -> AbortCode
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
        ) -> AbortCode
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
        ) -> AbortCode
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

        unsafe extern "C" fn buffer_resource<R>(
            data: *mut DynRuntimeData,
            group: u32,
            binding: u32,
            access: u32,
            pointer_out: &mut *const u8,
            len_out: &mut usize,
        ) -> AbortCode
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
                    let buffer = runtime.buffer_resource_mut(binding)?;
                    (buffer.as_mut_ptr() as *const u8, buffer.len())
                }
                else if access.contains(naga::StorageAccess::LOAD) {
                    let buffer = runtime.buffer_resource(binding)?;
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

        unsafe extern "C" fn image_resource<R>(
            data: *mut DynRuntimeData,
            group: u32,
            binding: u32,
            pointer_out: &mut *const Image,
        ) -> AbortCode
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            data.with_runtime(move |runtime| {
                let binding = naga::ResourceBinding { group, binding };
                let pointer = runtime.image_resource(binding)?;
                *pointer_out = pointer as *const _ as *const _;
                Ok(())
            })
        }

        unsafe extern "C" fn sampler_resource<R>(
            data: *mut DynRuntimeData,
            group: u32,
            binding: u32,
            pointer_out: &mut *const Sampler,
        ) -> AbortCode
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            data.with_runtime(move |runtime| {
                let binding = naga::ResourceBinding { group, binding };
                let pointer = runtime.sampler_resource(binding)?;
                *pointer_out = pointer as *const _ as *const _;
                Ok(())
            })
        }

        unsafe extern "C" fn image_sample<R>(
            data: *mut DynRuntimeData,
            image: *const Image,
            sampler: *const Sampler,
            gather: u32,
            u: f32,
            v: f32,
            array_index: u32,
            offset: u32,
            depth_ref: f32,
            clamp_to_edge: u8,
            texel_out: &mut [f32; 4],
        ) -> AbortCode
        where
            R: Runtime,
        {
            let data = unsafe {
                // SAFETY: The compiled code must call this function with a `*mut DynRuntime`
                // that corresponds to a valid `&mut RuntimeData<R>`
                RuntimeData::<R>::from_pointer_mut(data)
            };

            let u32_opt = |x: u32| (x != u32::MAX).then_some(x);
            let f32_opt = |x: f32| (!x.is_nan()).then_some(x);

            let image = unsafe { &*(image as *const R::Image) };
            let sampler = unsafe { &*(sampler as *const R::Sampler) };

            let gather = u32_opt(gather).map(naga::SwizzleComponent::from_index);
            let array_index = u32_opt(array_index);
            let offset = u32_opt(offset);
            let depth_ref = f32_opt(depth_ref);
            let clamp_to_edge = clamp_to_edge != 0;

            data.with_runtime(move |runtime| {
                *texel_out = runtime.image_sample(
                    image,
                    sampler,
                    gather,
                    [u, v],
                    array_index,
                    offset,
                    naga::SampleLevel::Auto,
                    depth_ref,
                    clamp_to_edge,
                )?;
                Ok(())
            })
        }

        RuntimeVtable {
            copy_inputs_to: copy_inputs_to::<R>,
            copy_outputs_from: copy_outputs_from::<R>,
            initialize_global_variables: initialize_global_variables::<R>,
            buffer_resource: buffer_resource::<R>,
            image_resource: image_resource::<R>,
            sampler_resource: sampler_resource::<R>,
            image_sample: image_sample::<R>,
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
    pub buffer_resource: ir::SigRef,
    pub image_resource: ir::SigRef,
    pub sampler_resource: ir::SigRef,
    pub image_sample: ir::SigRef,
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

        let buffer_resource = function_builder.import_signature(ir::Signature {
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

        let image_resource = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // binding group
                ir::AbiParam::new(ir::types::I32),
                // binding index
                ir::AbiParam::new(ir::types::I32),
                // buffer pointer, return by reference
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let sampler_resource = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // binding group
                ir::AbiParam::new(ir::types::I32),
                // binding index
                ir::AbiParam::new(ir::types::I32),
                // buffer pointer, return by reference
                pointer_param,
            ],
            returns: vec![result],
            call_conv: context.target_config.default_call_conv,
        });

        let image_sample = function_builder.import_signature(ir::Signature {
            params: vec![
                // context pointer
                context_param,
                // image
                pointer_param,
                // sampler
                pointer_param,
                // gather
                ir::AbiParam::new(ir::types::I32),
                // coordinate.u
                ir::AbiParam::new(ir::types::F32),
                // coordinate.v
                ir::AbiParam::new(ir::types::F32),
                // array_index
                ir::AbiParam::new(ir::types::I32),
                // offset
                ir::AbiParam::new(ir::types::I32),
                // depth_ref
                ir::AbiParam::new(ir::types::F32),
                // clamp_to_edge
                ir::AbiParam::new(ir::types::I8),
                // buffer pointer, return by reference
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
            buffer_resource,
            image_resource,
            sampler_resource,
            image_sample,
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

    pub fn buffer_resource(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        resource_binding: naga::ir::ResourceBinding,
        access: naga::StorageAccess,
        abort_block: ir::Block,
    ) -> Result<PointerRange<ir::Value>, Error> {
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
        runtime_method!(self, buffer_resource).call(
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
        Ok(PointerRange {
            pointer: Pointer {
                value: pointer,
                memory_flags: self.memory_flags,
                offset: 0,
            },
            len,
        })
    }

    pub fn image_resource(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        resource_binding: naga::ir::ResourceBinding,
        abort_block: ir::Block,
    ) -> Result<HandlePointer, Error> {
        // values for arguments
        let group = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.group));
        let binding = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.binding));

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

        // call runtime
        runtime_method!(self, image_resource).call(
            context,
            function_builder,
            [group, binding, pointer_out],
            abort_block,
        );

        // load returned pointer and len
        let pointer = function_builder
            .ins()
            .stack_load(pointer_type, pointer_out_stack_slot, 0);

        Ok(HandlePointer(pointer))
    }

    pub fn sampler_resource(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        resource_binding: naga::ir::ResourceBinding,
        abort_block: ir::Block,
    ) -> Result<HandlePointer, Error> {
        // values for arguments
        let group = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.group));
        let binding = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(resource_binding.binding));

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

        // call runtime
        runtime_method!(self, sampler_resource).call(
            context,
            function_builder,
            [group, binding, pointer_out],
            abort_block,
        );

        // load returned pointer and len
        let pointer = function_builder
            .ins()
            .stack_load(pointer_type, pointer_out_stack_slot, 0);

        Ok(HandlePointer(pointer))
    }

    pub fn image_sample(
        &self,
        context: &Context,
        function_builder: &mut FunctionBuilder,
        abort_block: ir::Block,
        image: HandlePointer,
        sampler: HandlePointer,
        gather: Option<naga::SwizzleComponent>,
        coordinate: VectorValue,
        array_index: Option<ScalarValue>,
        offset: Option<ScalarValue>,
        level: naga::SampleLevel,
        depth_ref: Option<ScalarValue>,
        clamp_to_edge: bool,
    ) -> Result<VectorValue, Error> {
        let gather = gather.map_or(u32::MAX, |swizzle_component| swizzle_component.index());
        let gather = function_builder
            .ins()
            .iconst(ir::types::I32, i64::from(gather));

        let coordinates = coordinate.components(function_builder);
        assert_eq!(coordinates.len(), 2);

        let array_index = array_index.map_or_else(
            || {
                function_builder
                    .ins()
                    .iconst(ir::types::I32, i64::from(u32::MAX))
            },
            |value| value.value,
        );

        let offset = offset.map_or_else(
            || {
                function_builder
                    .ins()
                    .iconst(ir::types::I32, i64::from(u32::MAX))
            },
            |value| value.value,
        );

        match level {
            naga::SampleLevel::Auto => {}
            _ => todo!("sample level: {level:?}"),
        }

        let depth_ref = depth_ref.map_or_else(
            || function_builder.ins().f32const(f32::NAN),
            |value| value.value,
        );

        let clamp_to_edge = function_builder
            .ins()
            .iconst(ir::types::I8, clamp_to_edge as i64);

        let texel_type = VectorType {
            size: naga::VectorSize::Quad,
            scalar: ScalarType::Float(FloatWidth::F32),
        };
        let texel_out_stack_slot = function_builder.create_sized_stack_slot(ir::StackSlotData {
            kind: ir::StackSlotKind::ExplicitSlot,
            size: 16,       // todo: don't hardcode this
            align_shift: 4, // this is log_2, so it's 16 bytes alignment
            key: None,
        });
        let texel_out =
            function_builder
                .ins()
                .stack_addr(context.pointer_type(), texel_out_stack_slot, 0);

        runtime_method!(self, image_sample).call(
            context,
            function_builder,
            [
                image.0,
                sampler.0,
                gather,
                coordinates[0],
                coordinates[1],
                array_index,
                offset,
                depth_ref,
                clamp_to_edge,
                texel_out,
            ],
            abort_block,
        );

        VectorValue::load(
            context,
            function_builder,
            texel_type,
            StackLocation::from(texel_out_stack_slot),
        )
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
            [&ir::BlockArg::Value(abort_code)],
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
                    .add_offset(offset.try_into().expect("stack offset overflow")),
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
pub struct DefaultRuntime<'layout, I, O, B> {
    pub input: I,
    pub input_layout: &'layout [BindingStackLayout],

    pub output: O,
    pub output_layout: &'layout [BindingStackLayout],

    pub binding_resources: B,

    pub private_memory_layout: &'layout PrivateMemoryLayout,
}

impl<'layout, I, O, B> DefaultRuntime<'layout, I, O, B> {
    pub fn new(
        input: I,
        input_layout: &'layout [BindingStackLayout],
        output: O,
        output_layout: &'layout [BindingStackLayout],
        binding_resources: B,
        private_memory_layout: &'layout PrivateMemoryLayout,
    ) -> Self {
        Self {
            input,
            input_layout,
            output,
            output_layout,
            binding_resources,
            private_memory_layout,
        }
    }
}

impl<'layout, I, O, B> Runtime for DefaultRuntime<'layout, I, O, B>
where
    I: ShaderInput,
    O: ShaderOutput,
    B: BindingResources,
{
    type Error = DefaultRuntimeError;
    type Image = B::Image;
    type Sampler = B::Sampler;

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
        let initialized = &self.private_memory_layout.initialized;
        private_data[..initialized.len()].copy_from_slice(initialized);
        private_data[initialized.len()..].fill(0);
        Ok(())
    }

    fn buffer_resource(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
        let buffer = self.binding_resources.buffer(binding);
        Ok(buffer)
    }

    fn buffer_resource_mut(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&mut [u8], Self::Error> {
        todo!("buffer_resource_mut: {binding:?}");
    }

    fn image_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Image, Self::Error> {
        let image = self.binding_resources.image(binding);
        Ok(image)
    }

    fn sampler_resource(
        &mut self,
        binding: naga::ResourceBinding,
    ) -> Result<&Self::Sampler, Self::Error> {
        let sampler = self.binding_resources.sampler(binding);
        Ok(sampler)
    }

    fn image_sample(
        &mut self,
        image: &Self::Image,
        sampler: &Self::Sampler,
        gather: Option<naga::SwizzleComponent>,
        coordinate: [f32; 2],
        array_index: Option<u32>,
        offset: Option<u32>,
        level: naga::SampleLevel,
        depth_ref: Option<f32>,
        clamp_to_edge: bool,
    ) -> Result<[f32; 4], Self::Error> {
        let texel = self.binding_resources.image_sample(
            image,
            sampler,
            gather,
            coordinate,
            array_index,
            offset,
            level,
            depth_ref,
            clamp_to_edge,
        );
        Ok(texel)
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
