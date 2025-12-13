#![allow(unused)]

use std::{
    path::PathBuf,
    sync::Arc,
};

use clap::Parser;
use color_eyre::{
    Section,
    eyre::Error,
};
use dotenvy::dotenv;
use wgpu::{
    Adapter,
    Color,
    CompositeAlphaMode,
    Device,
    Instance,
    LoadOp,
    Operations,
    PipelineLayout,
    PresentMode,
    Queue,
    RenderPassColorAttachment,
    RenderPassDescriptor,
    StoreOp,
    Surface,
    TextureFormat,
    TextureUsages,
    TextureView,
    wgt::SurfaceConfiguration,
};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::WindowEvent,
    event_loop::{
        ActiveEventLoop,
        EventLoop,
    },
    platform::run_on_demand::EventLoopExtRunOnDemand,
    window::{
        Window,
        WindowAttributes,
        WindowId,
    },
};

#[derive(Debug, Parser)]
struct Args {
    #[clap(short, long, default_value = "3")]
    vertices: u32,

    #[clap(short, long, default_value = "examples/triangle.wgsl")]
    shader: PathBuf,

    #[clap(short, long)]
    output: Option<PathBuf>,
}

pub fn main() -> Result<(), Error> {
    let _ = dotenv();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let shader_source = std::fs::read_to_string(&args.shader)
        .with_section(|| format!("Path: {}", args.shader.display()))?;

    tracing::info!(?args, "Hello Shader!");

    let instance = Instance::from_custom(wgpu_cpu::Instance::default());

    let (adapter, device, queue) = pollster::block_on(async {
        let adapter = instance.request_adapter(&Default::default()).await?;
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        Ok::<_, Error>((adapter, device, queue))
    })?;

    let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(&args.shader.display().to_string()),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hello_shader pipeline layout"),
        bind_group_layouts: &[],
        immediates_ranges: &[],
    });

    let target_texture_format = wgpu::TextureFormat::Rgba8UnormSrgb;

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size: wgpu::Extent3d {
            width: 512,
            height: 512,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: target_texture_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });

    let target_texture_view = target_texture.create_view(&Default::default());

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hello_shader pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader_module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_texture_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    });

    let mut command_encoder = device.create_command_encoder(&Default::default());

    {
        let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(RenderPassColorAttachment {
                view: &target_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(Color::BLACK),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });

        render_pass.set_pipeline(&pipeline);
        render_pass.draw(0..3, 0..1);
    }

    let submission_index = queue.submit([command_encoder.finish()]);
    let poll_result = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: None,
    });
    tracing::debug!(?poll_result);

    if let Some(output) = &args.output {
        tracing::info!(path = %output.display(), "Writing rendered image");
        wgpu_cpu::dump_texture(&target_texture, output)?;
    }

    tracing::info!("Example quitting");

    Ok(())
}
