use std::ops::{
    Index,
    Range,
};

use crate::util::SparseVec;

pub trait ShaderInput {
    fn write_into(&self, binding: &naga::Binding, ty: &naga::Type, target: &mut [u8]);
}

impl<T> ShaderInput for &T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &naga::Binding, ty: &naga::Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

impl<T> ShaderInput for &mut T
where
    T: ShaderInput,
{
    fn write_into(&self, binding: &naga::Binding, ty: &naga::Type, target: &mut [u8]) {
        T::write_into(self, binding, ty, target);
    }
}

pub trait ShaderOutput {
    fn read_from(&mut self, binding: &naga::Binding, ty: &naga::Type, source: &[u8]);
}

impl<T> ShaderOutput for &mut T
where
    T: ShaderOutput,
{
    fn read_from(&mut self, binding: &naga::Binding, ty: &naga::Type, source: &[u8]) {
        T::read_from(self, binding, ty, source);
    }
}

pub trait BindingResources {
    type Image;
    type Sampler;

    fn buffer(&self, binding: naga::ResourceBinding) -> &[u8];
    fn image(&self, binding: naga::ResourceBinding) -> &Self::Image;
    fn sampler(&self, binding: naga::ResourceBinding) -> &Self::Sampler;
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
    ) -> [f32; 4];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NullBinding;

impl ShaderInput for NullBinding {
    fn write_into(&self, binding: &naga::Binding, ty: &naga::Type, target: &mut [u8]) {
        let _ = (binding, ty);
        target.fill(0);
    }
}

impl ShaderOutput for NullBinding {
    fn read_from(&mut self, binding: &naga::Binding, ty: &naga::Type, source: &[u8]) {
        let _ = (binding, ty, source);
        // nop
    }
}

impl BindingResources for NullBinding {
    type Image = ();
    type Sampler = ();

    fn buffer(&self, binding: naga::ResourceBinding) -> &[u8] {
        let _ = binding;
        &[]
    }

    fn image(&self, binding: naga::ResourceBinding) -> &Self::Image {
        let _ = binding;
        &()
    }

    fn sampler(&self, binding: naga::ResourceBinding) -> &Self::Sampler {
        let _ = binding;
        &()
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
    ) -> [f32; 4] {
        let _ = (
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
        [0.0; 4]
    }
}

pub trait VisitIoBindings {
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    );
}

impl<T> VisitIoBindings for &mut T
where
    T: VisitIoBindings,
{
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        T::visit(self, binding, ty_handle, ty, offset, name, top_level);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FnVisitIoBindings<F>(pub F);

impl<F> VisitIoBindings for FnVisitIoBindings<F>
where
    F: FnMut(&naga::Binding, naga::Handle<naga::Type>, &naga::Type, u32, Option<&str>, bool),
{
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        (self.0)(binding, ty_handle, ty, offset, name, top_level)
    }
}

#[derive(Debug)]
pub struct IoBindingVisitor<'a, B> {
    pub types: &'a naga::UniqueArena<naga::Type>,
    pub visit: B,
}

impl<'a, B> IoBindingVisitor<'a, B> {
    pub fn new(types: &'a naga::UniqueArena<naga::Type>, visit: B) -> Self {
        Self { types, visit }
    }
}

impl<'a, F> IoBindingVisitor<'a, FnVisitIoBindings<F>> {
    pub fn from_fn(types: &'a naga::UniqueArena<naga::Type>, f: F) -> Self {
        IoBindingVisitor::new(types, FnVisitIoBindings(f))
    }
}

impl<'a, B> IoBindingVisitor<'a, B>
where
    B: VisitIoBindings,
{
    pub fn visit(
        &mut self,
        binding: Option<&naga::Binding>,
        ty_handle: naga::Handle<naga::Type>,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        let ty = &self.types[ty_handle];
        if let Some(binding) = binding {
            self.visit
                .visit(binding, ty_handle, ty, offset, name, top_level);
        }
        else {
            match &ty.inner {
                naga::TypeInner::Struct { members, span: _ } => {
                    for member in members {
                        self.visit(
                            member.binding.as_ref(),
                            member.ty,
                            offset + member.offset,
                            member.name.as_deref(),
                            false,
                        );
                    }
                }
                _ => panic!("Invalid binding type: {:?}", ty),
            }
        }
    }

    pub fn visit_function_argument(&mut self, argument: &naga::FunctionArgument, offset: u32) {
        self.visit(
            argument.binding.as_ref(),
            argument.ty,
            offset,
            argument.name.as_deref(),
            true,
        );
    }

    pub fn visit_function_result(&mut self, result: &naga::FunctionResult, offset: u32) {
        self.visit(result.binding.as_ref(), result.ty, offset, None, true);
    }
}

#[derive(Clone, Debug)]
pub enum InterStageLayout {
    Vertex { output: UserDefinedInterStageLayout },
    Fragment { input: UserDefinedInterStageLayout },
}

pub fn collect_user_defined_inter_stage_layout_from_function_arguments<'a>(
    module: &naga::Module,
    layouter: &naga::proc::Layouter,
    arguments: impl IntoIterator<Item = &'a naga::FunctionArgument>,
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
    module: &naga::Module,
    layouter: &naga::proc::Layouter,
    result: impl Into<Option<&'a naga::FunctionResult>>,
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
pub struct CollectUserDefinedInterStageLayout<'module> {
    pub layouter: &'module naga::proc::Layouter,
    pub buffer_offset: u32,
    pub locations: SparseVec<BindingLocationLayout>,
}

impl<'module> VisitIoBindings for CollectUserDefinedInterStageLayout<'module> {
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        // this is the offset in the struct that contains this inter-stage location
        // binding. we don't care about this, since we can layout our
        // inter-stage buffer as we want. in particular the layout of the vertex
        // output and fragment input might not even match.
        let _ = offset;
        let _ = (ty, name, top_level);

        match binding {
            naga::Binding::BuiltIn(_builtin) => {
                // nop
            }
            naga::Binding::Location {
                location,
                interpolation: _,
                sampling: _,
                blend_src: _,
                per_primitive: _,
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
    pub(super) locations: SparseVec<BindingLocationLayout>,
    pub(super) size: u32,
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
    type Output = BindingLocationLayout;

    fn index(&self, index: BindingLocation) -> &Self::Output {
        let index: usize = index.0.try_into().unwrap();
        &self.locations[index]
    }
}
