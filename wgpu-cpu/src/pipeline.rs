use std::{
    num::NonZeroU32,
    sync::Arc,
};

use crate::{
    bind_group::BindGroupLayout,
    render_pass::{
        fragment::FragmentState,
        vertex::VertexState,
    },
    shader::Error,
};

#[derive(Clone, Debug)]
pub struct PipelineLayout {
    descriptor: Arc<PipelineLayoutDescriptor>,
}

impl PipelineLayout {
    pub fn new(descriptor: &wgpu::PipelineLayoutDescriptor) -> Self {
        Self {
            descriptor: Arc::new(PipelineLayoutDescriptor::new(descriptor)),
        }
    }
}

impl wgpu::custom::PipelineLayoutInterface for PipelineLayout {}

#[derive(Debug)]
pub struct PipelineLayoutDescriptor {
    pub label: Option<String>,
    pub bind_group_layouts: Vec<BindGroupLayout>,
    pub immediate_size: u32,
}

impl PipelineLayoutDescriptor {
    pub fn new(layout: &wgpu::PipelineLayoutDescriptor) -> Self {
        Self {
            label: layout.label.map(ToOwned::to_owned),
            bind_group_layouts: layout
                .bind_group_layouts
                .iter()
                .map(|layout| layout.as_custom::<BindGroupLayout>().unwrap().clone())
                .collect(),
            immediate_size: layout.immediate_size,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPipeline {
    pub descriptor: Arc<RenderPipelineDescriptor>,
}

impl RenderPipeline {
    pub fn new(descriptor: &wgpu::RenderPipelineDescriptor) -> Result<Self, Error> {
        Ok(Self {
            descriptor: Arc::new(RenderPipelineDescriptor {
                label: descriptor.label.map(ToOwned::to_owned),
                layout: descriptor
                    .layout
                    .map(|layout| layout.as_custom::<PipelineLayout>().unwrap().clone()),
                vertex: VertexState::new(&descriptor.vertex)?,
                primitive: descriptor.primitive,
                depth_stencil: descriptor.depth_stencil.clone(),
                multisample: descriptor.multisample,
                fragment: descriptor
                    .fragment
                    .as_ref()
                    .map(FragmentState::new)
                    .transpose()?,
                multiview_mask: descriptor.multiview_mask,
            }),
        })
    }
}

impl wgpu::custom::RenderPipelineInterface for RenderPipeline {
    fn get_bind_group_layout(&self, index: u32) -> wgpu::custom::DispatchBindGroupLayout {
        if let Some(layout) = &self.descriptor.layout {
            let bind_group_layout = layout.descriptor.bind_group_layouts[index as usize].clone();
            wgpu::custom::DispatchBindGroupLayout::custom(bind_group_layout)
        }
        else {
            todo!();
        }
    }
}

#[derive(Debug)]
pub struct RenderPipelineDescriptor {
    pub label: Option<String>,
    pub layout: Option<PipelineLayout>,
    pub vertex: VertexState,
    pub primitive: wgpu::PrimitiveState,
    pub depth_stencil: Option<wgpu::DepthStencilState>,
    pub multisample: wgpu::MultisampleState,
    pub fragment: Option<FragmentState>,
    pub multiview_mask: Option<NonZeroU32>,
}
