use std::{
    marker::PhantomData,
    ops::{
        Index,
        Range,
    },
    sync::Arc,
};

use naga::{
    Binding,
    BuiltIn,
    FunctionArgument,
    FunctionResult,
    Handle,
    Module,
    ScalarKind,
    Type,
    TypeInner,
    UniqueArena,
    VectorSize,
    proc::Layouter,
};

use crate::{
    interpreter::Variable,
    memory::{
        ReadMemory,
        Slice,
        StackFrame,
        WriteMemory,
    },
    module::ShaderModule,
    util::SparseVec,
};

#[derive(Clone, Copy, Debug)]
pub struct BindingAddress {
    _todo: (),
}

pub trait ShaderInput {
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]);
}

impl<T> ShaderInput for &T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

impl<T> ShaderInput for &mut T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

pub trait ShaderOutput {
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]);
}

impl<T> ShaderOutput for &mut T
where
    T: ShaderOutput,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        T::read_from(self, binding, ty, source);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VertexInput<User> {
    pub vertex_index: u32,
    pub instance_index: u32,
    pub user_defined: User,
}

impl<User> ShaderInput for VertexInput<User>
where
    User: ReadMemory<BindingLocation>,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        let source = match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::VertexIndex => bytemuck::bytes_of(&self.vertex_index),
                    BuiltIn::InstanceIndex => bytemuck::bytes_of(&self.instance_index),
                    _ => unsupported_binding(binding),
                }
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => todo!(),
        };
        target[..source.len()].copy_from_slice(source);
    }
}

#[derive(Clone, Debug, Default)]
pub struct VertexOutput<User> {
    pub position: [f32; 4],
    pub user_defined: User,
}

impl<User> ShaderOutput for VertexOutput<User>
where
    User: WriteMemory<BindingLocation>,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        let target = match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::Position { invariant: _ } => {
                        bytemuck::bytes_of_mut(&mut self.position)
                    }
                    _ => unsupported_binding(binding),
                }
            }
            Binding::Location { location, .. } => {
                self.user_defined.write(BindingLocation::from(*location))
            }
        };
        target.copy_from_slice(&source[..target.len()]);
    }
}

pub trait Interpolate<const N: usize> {
    fn interpolate_scalar(&self, scalars: [f32; N]) -> f32;
    fn interpolate_vec2(&self, vectors: [[f32; 2]; N]) -> [f32; 2];
    fn interpolate_vec3(&self, vectors: [[f32; 3]; N]) -> [f32; 3];
    fn interpolate_vec4(&self, vectors: [[f32; 4]; N]) -> [f32; 4];
}

impl<const N: usize, I> Interpolate<N> for &I
where
    I: Interpolate<N>,
{
    fn interpolate_scalar(&self, scalars: [f32; N]) -> f32 {
        I::interpolate_scalar(self, scalars)
    }

    fn interpolate_vec2(&self, vectors: [[f32; 2]; N]) -> [f32; 2] {
        I::interpolate_vec2(self, vectors)
    }

    fn interpolate_vec3(&self, vectors: [[f32; 3]; N]) -> [f32; 3] {
        I::interpolate_vec3(self, vectors)
    }

    fn interpolate_vec4(&self, vectors: [[f32; 4]; N]) -> [f32; 4] {
        I::interpolate_vec4(self, vectors)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput<const N: usize, User, Interpolator> {
    pub position: [f32; 4],
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,
    pub barycentric: Interpolator,
    pub vertex_outputs: [User; N],
}

impl<const N: usize, User, Interpolator> ShaderInput for FragmentInput<N, User, Interpolator>
where
    User: ReadMemory<BindingLocation>,
    Interpolator: Interpolate<N>,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        match binding {
            Binding::BuiltIn(builtin) => {
                let source = match builtin {
                    BuiltIn::Position { invariant } => bytemuck::bytes_of(&self.position),
                    BuiltIn::FrontFacing => bytes_of_bool_as_u32(self.front_facing),
                    BuiltIn::PrimitiveIndex => bytemuck::bytes_of(&self.primitive_index),
                    BuiltIn::SampleIndex => bytemuck::bytes_of(&self.sample_index),
                    BuiltIn::SampleMask => bytemuck::bytes_of(&self.sample_mask),
                    _ => unsupported_binding(binding),
                };
                target[..source.len()].copy_from_slice(source);
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                let inputs = std::array::from_fn::<_, N, _>(|i| {
                    self.vertex_outputs[i].read((*location).into())
                });

                let interpolation = Interpolation::from_naga(*interpolation, *sampling);
                interpolation.interpolate_user(&self.barycentric, inputs, ty, target);
            }
        }
    }
}

// https://gpuweb.github.io/gpuweb/wgsl/#interpolation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Flat { sampling: FlatSampling },
    Perspective { sampling: InterpolationSampling },
    Linear { sampling: InterpolationSampling },
}

impl Interpolation {
    pub fn from_naga(
        interpolation: Option<naga::Interpolation>,
        sampling: Option<naga::Sampling>,
    ) -> Self {
        let invalid = || -> ! {
            panic!("Invalid sampling mode {sampling:?} for interpolation {interpolation:?}");
        };

        match (interpolation, sampling) {
            (Some(naga::Interpolation::Flat), Some(naga::Sampling::First)) => {
                return Self::Flat {
                    sampling: FlatSampling::First,
                };
            }
            (Some(naga::Interpolation::Flat), Some(naga::Sampling::Either)) => {
                return Self::Flat {
                    sampling: FlatSampling::Either,
                };
            }
            (Some(naga::Interpolation::Flat), None) => {
                return Self::Flat {
                    sampling: Default::default(),
                };
            }
            _ => {}
        }

        let sampling = match sampling {
            None | Some(naga::Sampling::Center) => InterpolationSampling::Center,
            Some(naga::Sampling::Centroid) => InterpolationSampling::Centroid,
            Some(naga::Sampling::Sample) => InterpolationSampling::Sample,
            _ => invalid(),
        };

        match interpolation {
            None | Some(naga::Interpolation::Perspective) => Self::Perspective { sampling },
            Some(naga::Interpolation::Linear) => Self::Linear { sampling },
            _ => invalid(),
        }
    }

    pub fn interpolate_user<const N: usize, Interpolator: Interpolate<N>>(
        &self,
        barycentric: Interpolator,
        inputs: [&[u8]; N],
        ty: &Type,
        output: &mut [u8],
    ) {
        let (vector_size, scalar) = ty
            .inner
            .vector_size_and_scalar()
            .unwrap_or_else(|| panic!("Invalid type for interpolation: {ty:?}"));

        match scalar.kind {
            ScalarKind::Sint | ScalarKind::Uint | ScalarKind::Bool => {
                match self {
                    Interpolation::Flat { sampling } => sampling.sample_user(inputs, output),
                    _ => panic!("Integer types must use flat interpolation"),
                }
            }
            ScalarKind::Float => {
                match self {
                    Interpolation::Flat { sampling } => sampling.sample_user(inputs, output),
                    Interpolation::Perspective { sampling } => {
                        todo!();
                    }
                    Interpolation::Linear { sampling } => {
                        macro_rules! interpolate_linear {
                            ($($pat:pat => $ty:ty as $method:ident,)*) => {
                                match vector_size {
                                    $(
                                        $pat => {
                                            let inputs =
                                                inputs.map(|input| *bytemuck::from_bytes::<$ty>(input));
                                            let output = bytemuck::from_bytes_mut::<$ty>(output);
                                            *output = barycentric.$method(inputs);
                                        }
                                    )*
                                }
                            };
                        }

                        interpolate_linear!(
                            None => f32 as interpolate_scalar,
                            Some(VectorSize::Bi) => [f32; 2] as interpolate_vec2,
                            Some(VectorSize::Tri) => [f32; 3] as interpolate_vec3,
                            Some(VectorSize::Quad) => [f32; 4] as interpolate_vec4,
                        );

                        // todo: implement the sampling behavior properly. right
                        // now we run the fragment shader once per sample anyway
                    }
                }
            }
            ScalarKind::AbstractInt | ScalarKind::AbstractFloat => {
                panic!("Unexpected abstract type: {scalar:?}")
            }
        }
    }
}

impl Default for Interpolation {
    fn default() -> Self {
        Self::Perspective {
            sampling: InterpolationSampling::Center,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FlatSampling {
    #[default]
    First,
    Either,
}

impl FlatSampling {
    fn sample_user<const N: usize>(&self, inputs: [&[u8]; N], output: &mut [u8]) {
        // either case we pick the first. the match is just here to make sure we
        // don't miss it when a variant is added.
        match self {
            FlatSampling::First | FlatSampling::Either => {
                output.copy_from_slice(inputs[0]);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterpolationSampling {
    #[default]
    Center,
    Centroid,
    Sample,
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentOutput<Color> {
    pub color_attachments: Color,
    pub raster: [u32; 2],
}

impl<Color> ShaderOutput for FragmentOutput<Color>
where
    Color: ColorAttachments,
{
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        match binding {
            Binding::BuiltIn(builtin) => {
                todo!()
            }
            Binding::Location { location, .. } => {
                let color: &[f32; 4] = bytemuck::from_bytes(source);
                self.color_attachments
                    .put_pixel(*location, self.raster, *color);
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct UserDefinedVertexInputs<'layout, 'buffer> {
    // todo: layouts and buffer guards for vertex buffers
    _phantom: PhantomData<(&'layout (), &'buffer ())>,
}

impl<'layout, 'buffer> ReadMemory<BindingLocation> for UserDefinedVertexInputs<'layout, 'buffer> {
    fn read(&self, address: BindingLocation) -> &[u8] {
        todo!();
    }
}

#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, derive_more::From, derive_more::Into,
)]
pub struct BindingLocation(pub u32);

#[derive(Clone, Copy, Debug)]
pub struct BindingLocationLayout {
    pub offset: u32,
    pub size: u32,
}

impl BindingLocationLayout {
    pub fn range(&self) -> Range<usize> {
        let start = self.offset as usize;
        let end = (self.offset + self.size) as usize;
        start..end
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageLayout {
    locations: Arc<[Option<BindingLocationLayout>]>,
    size: u32,
}

impl UserDefinedInterStageLayout {
    pub fn size(&self) -> u32 {
        self.size
    }
}

impl FromIterator<Option<BindingLocationLayout>> for UserDefinedInterStageLayout {
    fn from_iter<T: IntoIterator<Item = Option<BindingLocationLayout>>>(iter: T) -> Self {
        let mut size = None;
        let locations = iter
            .into_iter()
            .inspect(|location| {
                if let Some(location) = location {
                    let update_size: &mut u32 = size.get_or_insert_default();
                    *update_size = (*update_size).max(location.offset + location.size);
                }
            })
            .collect();

        Self {
            locations,
            size: size.unwrap_or_default(),
        }
    }
}

impl Index<BindingLocation> for UserDefinedInterStageLayout {
    type Output = Option<BindingLocationLayout>;

    fn index(&self, index: BindingLocation) -> &Self::Output {
        let index: usize = index.0.try_into().unwrap();
        &self.locations[index]
    }
}

#[derive(Clone, Debug, Default)]
pub struct UserDefinedInterStageBuffer {
    data: Vec<u8>,
    layout: UserDefinedInterStageLayout,
}

impl UserDefinedInterStageBuffer {
    pub fn new(layout: UserDefinedInterStageLayout) -> Self {
        let data = vec![0; layout.size as usize];
        Self { data, layout }
    }
}

impl ReadMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn read(&self, address: BindingLocation) -> &[u8] {
        self.layout[address]
            .as_ref()
            .map_or(&[], |layout| &self.data[layout.range()])
    }
}

impl WriteMemory<BindingLocation> for UserDefinedInterStageBuffer {
    fn write(&mut self, address: BindingLocation) -> &mut [u8] {
        self.layout[address]
            .as_ref()
            .map_or(&mut [], |layout| &mut self.data[layout.range()])
    }
}

pub trait ColorAttachments {
    fn put_pixel(&mut self, location: u32, position: [u32; 2], color: [f32; 4]);
}

#[track_caller]
fn unsupported_binding(binding: &Binding) -> ! {
    panic!("Binding not supported: {binding:?}");
}

fn bytes_of_bool_as_u32(b: bool) -> &'static [u8] {
    if b {
        bytemuck::bytes_of(&1u32)
    }
    else {
        bytemuck::bytes_of(&0u32)
    }
}

pub fn copy_shader_inputs_to_stack<'a, B, I>(
    stack_frame: &mut StackFrame<B>,
    module: &'a ShaderModule,
    inputs: I,
    argument: &FunctionArgument,
) -> Variable<'a>
where
    B: WriteMemory<BindingAddress>,
    I: ShaderInput,
{
    let variable = stack_frame.allocate_variable(argument.ty, module);

    IoBindingVisitor {
        types: &module.module.types,
        visit: CopyInputsToMemory {
            slice: variable.slice,
            memory: &mut stack_frame.memory,
            inputs,
        },
    }
    .visit_function_argument(argument, 0);

    variable
}

pub fn copy_shader_outputs_from_stack<B, O>(
    stack_frame: &StackFrame<B>,
    module: &ShaderModule,
    outputs: O,
    result: &FunctionResult,
    variable: Variable,
) where
    B: ReadMemory<BindingAddress>,
    O: ShaderOutput,
{
    IoBindingVisitor {
        types: &module.module.types,
        visit: CopyOutputsFromMemory {
            slice: variable.slice,
            memory: &stack_frame.memory,
            outputs,
        },
    }
    .visit_function_result(result, 0);
}

pub trait VisitIoBindings {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    );
}

impl<T> VisitIoBindings for &mut T
where
    T: VisitIoBindings,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        T::visit(self, binding, ty_handle, ty, offset, name);
    }
}

#[derive(Debug)]
pub struct IoBindingVisitor<'a, B> {
    pub types: &'a UniqueArena<Type>,
    pub visit: B,
}

impl<'a, B> IoBindingVisitor<'a, B>
where
    B: VisitIoBindings,
{
    pub fn visit(
        &mut self,
        binding: Option<&Binding>,
        ty_handle: Handle<Type>,
        offset: u32,
        name: Option<&str>,
    ) {
        let ty = &self.types[ty_handle];
        if let Some(binding) = binding {
            self.visit.visit(binding, ty_handle, ty, offset, name);
        }
        else {
            match &ty.inner {
                TypeInner::Struct { members, span } => {
                    for member in members {
                        self.visit(
                            member.binding.as_ref(),
                            member.ty,
                            offset + member.offset,
                            member.name.as_deref(),
                        );
                    }
                }
                _ => panic!("Invalid binding type: {:?}", ty),
            }
        }
    }

    pub fn visit_function_argument(&mut self, argument: &FunctionArgument, offset: u32) {
        self.visit(
            argument.binding.as_ref(),
            argument.ty,
            offset,
            argument.name.as_deref(),
        );
    }

    pub fn visit_function_result(&mut self, result: &FunctionResult, offset: u32) {
        self.visit(result.binding.as_ref(), result.ty, offset, None);
    }
}

#[derive(Debug)]
pub struct CopyInputsToMemory<M, I> {
    pub slice: Slice,
    pub memory: M,
    pub inputs: I,
}

impl<M, I> VisitIoBindings for CopyInputsToMemory<M, I>
where
    M: WriteMemory<Slice>,
    I: ShaderInput,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        let target = self.memory.write(self.slice.slice(offset..));
        self.inputs.write_into(binding, ty, target);
    }
}

#[derive(Debug)]
pub struct CopyOutputsFromMemory<M, O> {
    pub slice: Slice,
    pub memory: M,
    pub outputs: O,
}

impl<M, O> VisitIoBindings for CopyOutputsFromMemory<M, O>
where
    M: ReadMemory<Slice>,
    O: ShaderOutput,
{
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        let source = self.memory.read(self.slice.slice(offset..));
        self.outputs.read_from(binding, ty, source);
    }
}

#[derive(Clone, Debug)]
pub struct CollectUserDefinedInterStageLayout<'module> {
    pub layouter: &'module Layouter,
    pub buffer_offset: u32,
    pub locations: SparseVec<BindingLocationLayout>,
}

impl<'module> VisitIoBindings for CollectUserDefinedInterStageLayout<'module> {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        // this is the offset in the struct that contains this inter-stage location
        // binding. we don't care about this, since we can layout our
        // inter-stage buffer as we want. in particular the layout of the vertex
        // output and fragment input might not even match.
        let _ = offset;

        match binding {
            Binding::BuiltIn(_builtin) => {
                // nop
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
            } => {
                let type_layout = self.layouter[ty_handle];
                let offset = type_layout.alignment.round_up(self.buffer_offset);
                let size = type_layout.size;
                self.buffer_offset = offset + size;

                let index = *location as usize;
                self.locations
                    .insert(index, BindingLocationLayout { offset, size });
            }
        }
    }
}

pub fn collect_user_defined_inter_stage_layout_from_function_arguments<'a>(
    module: &Module,
    layouter: &Layouter,
    arguments: impl IntoIterator<Item = &'a FunctionArgument>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::default(),
    };

    for argument in arguments {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_argument(argument, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

pub fn collect_user_defined_inter_stage_layout_from_function_result<'a>(
    module: &Module,
    layouter: &Layouter,
    result: impl Into<Option<&'a FunctionResult>>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::new(),
    };

    if let Some(result) = result.into() {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_result(result, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

#[derive(Clone, Debug)]
pub enum UserDefinedIoLayout {
    Vertex {
        // todo: input
        output: UserDefinedInterStageLayout,
    },
    Fragment {
        input: UserDefinedInterStageLayout,
        // output: don't need it, but would be handy for verification
    },
}
