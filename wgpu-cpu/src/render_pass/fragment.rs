use naga::{
    Binding,
    BuiltIn,
    ScalarKind,
    ShaderStage,
    Type,
    VectorSize,
};
use naga_cranelift::{
    CompiledModule,
    bindings::{
        BindingLocation,
        NullBinding,
        ShaderInput,
        ShaderOutput,
    },
    product::{
        EntryPointError,
        EntryPointIndex,
    },
};
use nalgebra::{
    Point2,
    Point3,
    SVector,
    Vector4,
};

use crate::{
    render_pass::{
        bytes_of_bool_as_u8,
        clipper::{
            ClipPosition,
            Clipped,
        },
        evaluate_compare_function,
        invalid_binding,
        primitive::{
            AsFrontFace,
            Primitive,
        },
        raster::RasterizerOutput,
        state::RenderPipelineState,
    },
    shader::{
        Error,
        ShaderModule,
        UserDefinedInterStagePoolBuffer,
        memory::ReadMemory,
    },
    texture::{
        TextureInfo,
        TextureViewAttachment,
        TextureWriteGuard,
    },
    util::interpolation::Interpolate,
};

#[derive(Debug)]
pub struct FragmentState {
    pub module: CompiledModule,
    pub entry_point: EntryPointIndex,
    pub targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl FragmentState {
    pub fn new(fragment: &wgpu::FragmentState) -> Result<Self, Error> {
        let module = fragment.module.as_custom::<ShaderModule>().unwrap().clone();
        let module = module.for_pipeline(&fragment.compilation_options)?;

        let entry_point = module
            .find_entry_point(fragment.entry_point.as_deref(), ShaderStage::Fragment)
            .unwrap();

        Ok(Self {
            module,
            entry_point,
            targets: fragment.targets.to_vec(),
        })
    }

    pub fn early_depth_test(&self) -> Option<naga::EarlyDepthTest> {
        let entry_point = self.module.entry_point(self.entry_point);
        entry_point.early_depth_test()
    }
}

type VertexOutput = crate::render_pass::vertex::VertexOutput<UserDefinedInterStagePoolBuffer>;

#[derive(Debug)]
pub struct FragmentProcessingState<'pass, 'state> {
    pub front_face: wgpu::FrontFace,
    pub cull_mode: Option<wgpu::Face>,
    pub depth_stencil_state: Option<&'state wgpu::DepthStencilState>,
    pub color_attachments: &'state mut [Option<AcquiredColorAttachment<'pass>>],
    pub depth_stencil_attachment: Option<&'state mut AcquiredDepthStencilAttachment<'pass>>,
    pub fragment_state: &'state FragmentState,
}

impl<'pass, 'state> FragmentProcessingState<'pass, 'state> {
    pub fn new(
        pipeline_state: &'state RenderPipelineState,
        color_attachments: &'state mut [Option<AcquiredColorAttachment<'pass>>],
        depth_stencil_attachment: Option<&'state mut AcquiredDepthStencilAttachment<'pass>>,
    ) -> Option<Self> {
        let pipeline_descriptor = &pipeline_state.pipeline.descriptor;
        let fragment_state = pipeline_descriptor.fragment.as_ref()?;
        let primitive_state = &pipeline_descriptor.primitive;

        Some(Self {
            front_face: primitive_state.front_face,
            cull_mode: primitive_state.cull_mode,
            depth_stencil_state: pipeline_descriptor.depth_stencil.as_ref(),
            color_attachments,
            depth_stencil_attachment,
            fragment_state,
        })
    }
}

impl<'pass, 'state> FragmentProcessingState<'pass, 'state> {
    pub fn process<const NUM_VERTICES: usize, Vertex, Face, Inter>(
        &mut self,
        primitive: &Primitive<Vertex, NUM_VERTICES, Face>,
        front_facing: bool,
        primitive_index: u32,
        rasterizer_output: RasterizerOutput<Inter>,
    ) where
        Inter: Interpolate<NUM_VERTICES>,
        Vertex: AsRef<ClipPosition> + AsRef<Clipped<VertexOutput, Inter>>,
        Face: AsFrontFace,
    {
        let sample_index = rasterizer_output
            .sample_index
            .map_or(0, |sample_index| sample_index.get().into());
        let sample_mask = !0; // todo

        let input = FragmentInput {
            position: rasterizer_output.fragment,
            front_facing,
            primitive_index,
            sample_index,
            sample_mask,
            interpolation_coefficients: rasterizer_output.interpolation,
            inter_stage_variables: primitive
                .each_vertex_ref::<Clipped<VertexOutput, Inter>>()
                .map(|vertex_output| vertex_output.unclipped.inter_stage_variables.clone()),
        };

        let mut output = FragmentOutput {
            position: rasterizer_output.framebuffer,
            frag_depth: rasterizer_output.fragment.z,
            sample_mask,
            color_attachments: &mut self.color_attachments,
            depth_stencil_attachment: self.depth_stencil_attachment.as_deref_mut(),
            depth_stencil_state: self.depth_stencil_state,
            depth_test_result: None,
        };

        // perform early depth test
        let (early_depth_test, late_depth_test) = match self.fragment_state.early_depth_test() {
            None => {
                // only do late depth test
                (false, true)
            }
            Some(naga::EarlyDepthTest::Force) => {
                // only do early depth test
                (true, false)
            }
            Some(naga::EarlyDepthTest::Allow { conservative: _ }) => {
                // do both
                (true, true)
            }
        };

        if early_depth_test {
            let depth_test_result = output.depth_test();

            if !depth_test_result {
                // early depth test reject
                return;
            }

            if !late_depth_test {
                // if we don't want to perform a late depth test we can store the result of the
                // early depth test
                output.depth_test_result = Some(depth_test_result);
            }
        }

        // run fragment shader
        let entry_point = self
            .fragment_state
            .module
            .entry_point(self.fragment_state.entry_point);

        match entry_point.run(&input, &mut output, NullBinding) {
            Ok(()) => {}
            Err(EntryPointError::Killed) => {
                // the fragment shader ran discard. it won't have written to
                // its color/depth attachments, so we don't have to do
                // anything here.
            }
            result => {
                // some other error. for now we'll panic
                result.expect("fragment shader failed");
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput<const N: usize, Inter, User> {
    pub position: Vector4<f32>,
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,
    pub interpolation_coefficients: Inter,
    pub inter_stage_variables: [User; N],
}

impl<const N: usize, Inter, User> ShaderInput for FragmentInput<N, Inter, User>
where
    Inter: Interpolate<N>,
    User: ReadMemory<BindingLocation>,
{
    fn write_into(&self, binding: &Binding, ty: &Type, target: &mut [u8]) {
        match binding {
            Binding::BuiltIn(builtin) => {
                let source = match builtin {
                    BuiltIn::Position { invariant } => bytemuck::bytes_of(&self.position),
                    BuiltIn::FrontFacing => bytes_of_bool_as_u8(self.front_facing),
                    BuiltIn::PrimitiveIndex => bytemuck::bytes_of(&self.primitive_index),
                    BuiltIn::SampleIndex => bytemuck::bytes_of(&self.sample_index),
                    BuiltIn::SampleMask => bytemuck::bytes_of(&self.sample_mask),
                    _ => invalid_binding(binding),
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
                    self.inter_stage_variables[i].read((*location).into())
                });

                let interpolation = Interpolation::from_naga(*interpolation, *sampling);
                interpolation.interpolate_user(
                    &self.interpolation_coefficients,
                    inputs,
                    ty,
                    target,
                );
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

    pub fn interpolate_user<const N: usize, Inter>(
        &self,
        interpolation_coefficients: Inter,
        inputs: [&[u8]; N],
        ty: &Type,
        output: &mut [u8],
    ) where
        Inter: Interpolate<N>,
    {
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
                            ($($pat:pat => $n:expr,)*) => {
                                match vector_size {
                                    $(
                                        $pat => {
                                            let inputs = inputs.map(|input| *bytemuck::from_bytes::<SVector<f32, $n>>(input));
                                            let output = bytemuck::from_bytes_mut::<SVector<f32, $n>>(output);
                                            *output = interpolation_coefficients.interpolate(inputs);
                                        }
                                    )*
                                }
                            };
                        }

                        interpolate_linear!(
                            None => 1,
                            Some(VectorSize::Bi) => 2,
                            Some(VectorSize::Tri) => 3,
                            Some(VectorSize::Quad) => 4,
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

#[derive(Debug)]
pub struct FragmentOutput<'state, 'pass> {
    pub position: Point2<u32>,
    pub frag_depth: f32,
    pub sample_mask: u32,
    pub color_attachments: &'state mut [Option<AcquiredColorAttachment<'pass>>],
    pub depth_stencil_attachment: Option<&'state mut AcquiredDepthStencilAttachment<'pass>>,
    pub depth_stencil_state: Option<&'state wgpu::DepthStencilState>,
    pub depth_test_result: Option<bool>,
}

impl<'state, 'pass> FragmentOutput<'state, 'pass> {
    pub fn depth_test(&mut self) -> bool {
        if let (Some(depth_stencil_attachment), Some(depth_stencil_state)) = (
            self.depth_stencil_attachment.as_mut(),
            self.depth_stencil_state.as_ref(),
        ) {
            let buffer_frag_depth = depth_stencil_attachment.get_depth(self.position);
            if evaluate_compare_function(
                depth_stencil_state.depth_compare,
                self.frag_depth,
                buffer_frag_depth,
            ) {
                // accept
                if depth_stencil_state.depth_write_enabled {
                    depth_stencil_attachment.put_depth(self.position, self.frag_depth);
                }

                true
            }
            else {
                // reject
                false
            }
        }
        else {
            true
        }
    }
}

impl<'state, 'pass> ShaderOutput for FragmentOutput<'state, 'pass> {
    fn read_from(&mut self, binding: &Binding, ty: &Type, source: &[u8]) {
        match binding {
            Binding::BuiltIn(builtin) => {
                match builtin {
                    BuiltIn::FragDepth => {
                        self.frag_depth = *bytemuck::from_bytes::<f32>(source);
                    }
                    BuiltIn::SampleMask => {
                        self.sample_mask = *bytemuck::from_bytes::<u32>(source);
                    }
                    _ => invalid_binding(binding),
                }
            }
            Binding::Location { location, .. } => {
                let depth_test_result = if let Some(depth_test_result) = self.depth_test_result {
                    depth_test_result
                }
                else {
                    let depth_test_result = self.depth_test();
                    self.depth_test_result = Some(depth_test_result);
                    depth_test_result
                };

                if depth_test_result {
                    let color = *bytemuck::from_bytes::<Vector4<f32>>(source);
                    let color_attachment =
                        self.color_attachments[*location as usize].as_mut().unwrap();
                    color_attachment.put_pixel(self.position, color);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct ColorAttachment {
    pub view: TextureViewAttachment,
    pub depth_slice: u32,
    pub resolve_target: Option<TextureViewAttachment>,
    pub ops: wgpu::Operations<wgpu::Color>,
}

impl ColorAttachment {
    pub fn new(color_attachment: &wgpu::RenderPassColorAttachment) -> Self {
        Self {
            view: TextureViewAttachment::from_wgpu(&color_attachment.view).unwrap(),
            depth_slice: color_attachment.depth_slice.unwrap_or_default(),
            resolve_target: color_attachment
                .resolve_target
                .map(|texture| TextureViewAttachment::from_wgpu(texture).unwrap()),
            ops: color_attachment.ops,
        }
    }
}

#[derive(Debug)]
pub struct AcquiredColorAttachment<'color> {
    texture_guard: TextureWriteGuard<'color>,
    texture_info: &'color TextureInfo,
    ops: wgpu::Operations<wgpu::Color>,
    depth_slice: u32,
}

impl<'color> AcquiredColorAttachment<'color> {
    pub fn new(color_attachment: &'color ColorAttachment) -> Self {
        let texture_guard = color_attachment.view.write();
        let texture_info = &color_attachment.view.info;

        if color_attachment.resolve_target.is_some() {
            todo!("color attachment resolve target");
        }

        Self {
            texture_guard,
            texture_info,
            ops: color_attachment.ops,
            depth_slice: color_attachment.depth_slice,
        }
    }

    pub fn load(&mut self) {
        match self.ops.load {
            wgpu::LoadOp::Clear(clear_color) => {
                self.texture_guard.clear_color(clear_color);
            }
            wgpu::LoadOp::Load => {
                // nop
            }
            wgpu::LoadOp::DontCare(_) => {
                // nop
            }
        }
    }

    pub fn store(&mut self) {
        // todo: what to do?
        match self.ops.store {
            wgpu::StoreOp::Store => {}
            wgpu::StoreOp::Discard => {}
        }
    }

    pub fn put_pixel(&mut self, position: Point2<u32>, color: Vector4<f32>) {
        let position = Point3::new(position.x, position.y, self.depth_slice);
        self.texture_guard.put_pixel(position, color);
    }
}

#[derive(Debug)]
pub struct DepthStencilAttachment {
    pub view: TextureViewAttachment,
    pub depth_ops: Option<wgpu::Operations<f32>>,
    pub stencil_ops: Option<wgpu::Operations<u32>>,
}

impl DepthStencilAttachment {
    pub fn new(depth_stencil_attachment: &wgpu::RenderPassDepthStencilAttachment) -> Self {
        Self {
            view: TextureViewAttachment::from_wgpu(&depth_stencil_attachment.view).unwrap(),
            depth_ops: depth_stencil_attachment.depth_ops,
            stencil_ops: depth_stencil_attachment.stencil_ops,
        }
    }
}

#[derive(Debug)]
pub struct AcquiredDepthStencilAttachment<'a> {
    texture_guard: TextureWriteGuard<'a>,
    texture_info: &'a TextureInfo,
    depth_ops: Option<wgpu::Operations<f32>>,
    stencil_ops: Option<wgpu::Operations<u32>>,
}

impl<'a> AcquiredDepthStencilAttachment<'a> {
    pub fn new(depth_stencil_attachment: &'a DepthStencilAttachment) -> Self {
        let texture_guard = depth_stencil_attachment.view.write();
        let texture_info = &depth_stencil_attachment.view.info;

        Self {
            texture_guard,
            texture_info,
            depth_ops: depth_stencil_attachment.depth_ops,
            stencil_ops: depth_stencil_attachment.stencil_ops,
        }
    }

    pub fn load(&mut self) {
        if let Some(depth_ops) = self.depth_ops {
            match depth_ops.load {
                wgpu::LoadOp::Clear(depth) => {
                    self.texture_guard.clear_depth(depth);
                }
                wgpu::LoadOp::Load => {
                    // nop
                }
                wgpu::LoadOp::DontCare(_) => {
                    // nop
                }
            }
        }

        if let Some(stencil_ops) = self.stencil_ops {
            todo!("stencil_ops");
        }
    }

    pub fn store(&mut self) {
        // todo: what to do?
    }

    pub fn get_depth(&self, position: Point2<u32>) -> f32 {
        self.texture_guard.get_depth(position)
    }

    pub fn put_depth(&mut self, position: Point2<u32>, depth: f32) {
        self.texture_guard.put_depth(position, depth);
    }
}
