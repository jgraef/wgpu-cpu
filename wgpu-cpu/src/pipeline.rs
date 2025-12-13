use std::{
    collections::HashMap,
    num::NonZeroU32,
    sync::Arc,
};

use naga_interpreter::{
    EntryPointIndex,
    ShaderStage,
};

use crate::{
    bind_group::BindGroupLayout,
    shader::ShaderModule,
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
    pub immediates_ranges: Vec<wgpu::ImmediateRange>,
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
            immediates_ranges: layout.immediates_ranges.to_vec(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RenderPipeline {
    pub descriptor: Arc<RenderPipelineDescriptor>,
}

impl RenderPipeline {
    pub fn new(descriptor: &wgpu::RenderPipelineDescriptor) -> Self {
        Self {
            descriptor: Arc::new(RenderPipelineDescriptor {
                label: descriptor.label.map(ToOwned::to_owned),
                layout: descriptor
                    .layout
                    .map(|layout| layout.as_custom::<PipelineLayout>().unwrap().clone()),
                vertex: VertexState::new(&descriptor.vertex),
                primitive: descriptor.primitive,
                depth_stencil: descriptor.depth_stencil.clone(),
                multisample: descriptor.multisample,
                fragment: descriptor.fragment.as_ref().map(FragmentState::new),
                multiview_mask: descriptor.multiview_mask,
            }),
        }
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

#[derive(Debug)]
pub struct VertexState {
    pub module: ShaderModule,
    pub entry_point_name: Option<String>,
    pub entry_point_index: EntryPointIndex,
    pub compilation_options: PipelineCompilationOptions,
    pub buffers: Vec<VertexBufferLayout>,
}

impl VertexState {
    pub fn new(vertex: &wgpu::VertexState) -> Self {
        let module = vertex.module.as_custom::<ShaderModule>().unwrap().clone();

        let entry_point_index = module
            .as_ref()
            .find_entry_point(vertex.entry_point.as_deref(), ShaderStage::Vertex)
            .unwrap();

        Self {
            module,
            entry_point_name: vertex.entry_point.map(ToOwned::to_owned),
            entry_point_index,
            compilation_options: PipelineCompilationOptions::new(&vertex.compilation_options),
            buffers: vertex
                .buffers
                .iter()
                .map(|buffer| VertexBufferLayout::new(buffer))
                .collect(),
        }
    }
}

#[derive(Debug)]
pub struct VertexBufferLayout {
    pub array_stride: wgpu::BufferAddress,
    pub step_mode: wgpu::VertexStepMode,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn new(buffer: &wgpu::VertexBufferLayout) -> Self {
        Self {
            array_stride: buffer.array_stride,
            step_mode: buffer.step_mode,
            attributes: buffer.attributes.to_vec(),
        }
    }
}

#[derive(Debug)]
pub struct FragmentState {
    pub module: ShaderModule,
    pub entry_point_name: Option<String>,
    pub entry_point_index: EntryPointIndex,
    pub compilation_options: PipelineCompilationOptions,
    pub targets: Vec<Option<wgpu::ColorTargetState>>,
}

impl FragmentState {
    pub fn new(fragment: &wgpu::FragmentState) -> Self {
        let module = fragment.module.as_custom::<ShaderModule>().unwrap().clone();
        let entry_point_index = module
            .as_ref()
            .find_entry_point(fragment.entry_point.as_deref(), ShaderStage::Vertex)
            .unwrap();

        Self {
            module,
            entry_point_name: fragment.entry_point.map(ToOwned::to_owned),
            entry_point_index,
            compilation_options: PipelineCompilationOptions::new(&fragment.compilation_options),
            targets: fragment.targets.to_vec(),
        }
    }
}

#[derive(Debug)]
pub struct PipelineCompilationOptions {
    pub constants: HashMap<String, f64>,
    pub zero_initialize_workgroup_memory: bool,
}

impl PipelineCompilationOptions {
    pub fn new(options: &wgpu::PipelineCompilationOptions) -> Self {
        Self {
            constants: options
                .constants
                .iter()
                .map(|(name, value)| ((*name).to_owned(), *value))
                .collect(),
            zero_initialize_workgroup_memory: options.zero_initialize_workgroup_memory,
        }
    }
}
