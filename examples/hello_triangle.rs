#![allow(unused)]

use std::sync::Arc;

use color_eyre::eyre::Error;
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

pub fn main() -> Result<(), Error> {
    let _ = dotenv();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    tracing::info!("Hello Triangle!");

    let mut app = App::new()?;
    let mut event_loop = EventLoop::new()?;
    event_loop.run_app_on_demand(&mut app)?;

    tracing::info!("Example quitting");

    Ok(())
}

#[derive(Debug)]
struct App {
    instance: Instance,
    adapter: Adapter,
    device: Device,
    queue: Queue,
    shader_module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    window: Option<AppWindow>,
}

impl App {
    pub fn new() -> Result<Self, Error> {
        let instance = Instance::from_custom(wgpu_cpu::Instance::default());

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance.request_adapter(&Default::default()).await?;
            let (device, queue) = adapter.request_device(&Default::default()).await?;
            Ok::<_, Error>((adapter, device, queue))
        })?;

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("shader.wgsl"));

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hello_triangle pipeline layout"),
            bind_group_layouts: &[],
            immediates_ranges: &[],
        });

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            shader_module,
            pipeline_layout,
            window: None,
        })
    }

    fn render(&self, target_texture: &TextureView, pipeline: &wgpu::RenderPipeline) {
        let mut command_encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target_texture,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color::GREEN),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(pipeline);
            render_pass.draw(0..3, 0..1);
        }

        self.queue.submit([command_encoder.finish()]);
    }

    fn create_window(&self, event_loop: &ActiveEventLoop) -> Result<AppWindow, Error> {
        let window = Arc::new(
            event_loop.create_window(WindowAttributes::default().with_title("Hello Triangle"))?,
        );
        let window_size = window.inner_size();
        let surface = self.instance.create_surface(window.clone())?;
        let surface_capabilities = surface.get_capabilities(&self.adapter);
        let format = surface_capabilities.formats[0];

        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hello_triangle pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader_module,
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
                    module: &self.shader_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

        let window = AppWindow {
            window,
            surface,
            format,
            pipeline,
        };
        window.configure(&self.device, window_size);
        Ok(window)
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        match self.create_window(event_loop) {
            Ok(window) => {
                self.window = Some(window);
            }
            Err(error) => {
                tracing::error!("Failed to create window: {error}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(physical_size) => {
                if let Some(window) = &self.window {
                    window.configure(&self.device, physical_size);
                }
            }
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Destroyed => {
                self.window = None;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    match window.surface.get_current_texture() {
                        Ok(target_texture) => {
                            let view = target_texture.texture.create_view(&Default::default());
                            self.render(&view, &window.pipeline);
                            target_texture.present();
                        }
                        Err(error) => {
                            tracing::error!("Failed to get surface texture for drawing: {error}");
                            event_loop.exit();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[derive(Debug)]
struct AppWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
}

impl AppWindow {
    pub fn configure(&self, device: &Device, window_size: PhysicalSize<u32>) {
        self.surface.configure(
            device,
            &SurfaceConfiguration {
                usage: TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                width: window_size.width,
                height: window_size.height,
                present_mode: PresentMode::Immediate,
                desired_maximum_frame_latency: 0,
                alpha_mode: CompositeAlphaMode::Opaque,
                view_formats: vec![],
            },
        )
    }
}
