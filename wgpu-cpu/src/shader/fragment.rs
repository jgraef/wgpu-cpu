use naga::{
    Binding,
    BuiltIn,
    ScalarKind,
    Type,
    VectorSize,
};
use naga_interpreter::{
    bindings::{
        BindingLocation,
        ShaderInput,
        ShaderOutput,
    },
    memory::ReadMemory,
};
use nalgebra::{
    Point2,
    SVector,
    Vector4,
};

use crate::{
    render_pass::{
        color_attachment::AcquiredColorAttachment,
        depth_stencil_attachment::AcquiredDepthStencilAttachment,
    },
    shader::{
        bytes_of_bool_as_u32,
        evaluate_compare_function,
        invalid_binding,
    },
    util::Barycentric,
};

#[derive(Clone, Copy, Debug)]
pub struct FragmentInput<const N: usize, User> {
    pub position: Vector4<f32>,
    pub front_facing: bool,
    pub primitive_index: u32,
    pub sample_index: u32,
    pub sample_mask: u32,
    pub barycentric: Barycentric<N>,
    pub vertex_outputs: [User; N],
}

impl<const N: usize, User> ShaderInput for FragmentInput<N, User>
where
    User: ReadMemory<BindingLocation>,
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
                    self.vertex_outputs[i].read((*location).into())
                });

                let interpolation = Interpolation::from_naga(*interpolation, *sampling);
                interpolation.interpolate_user(self.barycentric, inputs, ty, target);
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

    pub fn interpolate_user<const N: usize>(
        &self,
        barycentric: Barycentric<N>,
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
                            ($($pat:pat => $n:expr,)*) => {
                                match vector_size {
                                    $(
                                        $pat => {
                                            let inputs = inputs.map(|input| *bytemuck::from_bytes::<SVector<f32, $n>>(input));
                                            let output = bytemuck::from_bytes_mut::<SVector<f32, $n>>(output);
                                            *output = barycentric.interpolate(inputs);
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
                if self.depth_test() {
                    let color = *bytemuck::from_bytes::<Vector4<f32>>(source);
                    let color_attachment =
                        self.color_attachments[*location as usize].as_mut().unwrap();
                    color_attachment.put_pixel(self.position, color);
                }
            }
        }
    }
}
