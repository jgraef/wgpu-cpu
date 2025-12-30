use std::{
    collections::HashSet,
    f32::consts::FRAC_PI_4,
    path::{
        Path,
        PathBuf,
    },
    sync::Arc,
};

use bytemuck::{
    Pod,
    Zeroable,
};
use clap::Parser;
use color_eyre::eyre::Error;
use dotenvy::dotenv;
use nalgebra::{
    Isometry3,
    Matrix4,
    Perspective3,
    Point3,
    Translation3,
    Vector2,
    Vector3,
    Vector4,
};
use num_traits::Bounded;
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{
        ElementState,
        KeyEvent,
        WindowEvent,
    },
    event_loop::{
        ActiveEventLoop,
        EventLoop,
    },
    keyboard::{
        KeyCode,
        PhysicalKey,
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
    #[clap(short, long)]
    output: Option<PathBuf>,

    #[clap(short, long, default_value = "wgpu-cpu/examples/teapot.obj")]
    mesh: PathBuf,

    #[clap(short = 'W', long, default_value = "600")]
    width: u32,

    #[clap(short = 'H', long, default_value = "400")]
    height: u32,
}

pub fn main() -> Result<(), Error> {
    let _ = dotenv();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let image_size = Vector2::new(args.width, args.height);

    tracing::info!(?args, "Hello Teapot!");

    let mut app = App::new(&args.mesh, image_size)?;

    if let Some(output) = &args.output {
        app.render_to_file(&output, image_size)?;
    }
    else {
        let mut event_loop = EventLoop::new()?;
        event_loop.run_app_on_demand(&mut app)?;
    }

    tracing::info!("Example quitting");

    Ok(())
}

#[derive(Debug)]
struct App {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_module: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    window: Option<AppWindow>,
    mesh: Mesh,
    camera: Camera,
    initial_window_size: Vector2<u32>,
    keys_down: HashSet<PhysicalKey>,
}

impl App {
    pub fn new(mesh: impl AsRef<Path>, initial_window_size: Vector2<u32>) -> Result<Self, Error> {
        let instance = wgpu_cpu::instance(Default::default());

        let (adapter, device, queue) = pollster::block_on(async {
            let adapter = instance.request_adapter(&Default::default()).await?;
            let (device, queue) = adapter.request_device(&Default::default()).await?;
            Ok::<_, Error>((adapter, device, queue))
        })?;

        let shader_module = device.create_shader_module(wgpu::include_wgsl!("hello_mesh.wgsl"));

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera uniform"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hello_triangle pipeline layout"),
            bind_group_layouts: &[&camera_bind_group_layout],
            immediate_size: 0,
        });

        let mesh = Mesh::load(mesh, &device)?;

        let camera = {
            let transform = Isometry3::face_towards(
                &(mesh.center + Vector3::new(0.0, 0.0, mesh.size.max() * -1.0)),
                &(mesh.center + Vector3::new(0.0, mesh.size.y * 0.25, 0.0)),
                &Vector3::y(),
            );

            Camera::new(
                &device,
                initial_window_size,
                &camera_bind_group_layout,
                transform,
            )
        };

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            shader_module,
            pipeline_layout,
            window: None,
            mesh,
            camera,
            initial_window_size,
            keys_down: HashSet::new(),
        })
    }

    fn create_pipeline(&self, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
        self.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("hello_triangle pipeline"),
                layout: Some(&self.pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x4, // position
                            1 => Float32x4, // color
                        ],
                    }],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Cw,
                    cull_mode: Some(wgpu::Face::Back),
                    //cull_mode: None,
                    unclipped_depth: false,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    conservative: false,
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    //depth_compare: wgpu::CompareFunction::Always,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
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
            })
    }

    fn create_window(&self, event_loop: &ActiveEventLoop) -> Result<AppWindow, Error> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Hello Mesh")
                    .with_inner_size(PhysicalSize::new(
                        self.initial_window_size.x,
                        self.initial_window_size.y,
                    )),
            )?,
        );
        let window_size = window.inner_size();
        let window_size = Vector2::new(window_size.width, window_size.height);
        let surface = self.instance.create_surface(window.clone())?;
        let surface_capabilities = surface.get_capabilities(&self.adapter);
        let format = surface_capabilities.formats[0];

        let (_, depth_texture_view) = create_depth_texture(&self.device, window_size);

        let pipeline = self.create_pipeline(format);

        let window = AppWindow {
            window,
            surface,
            format,
            pipeline,
            depth_texture_view,
        };
        window.configure(&self.device, window_size);
        Ok(window)
    }

    fn render_to_file(
        &self,
        output: impl AsRef<Path>,
        image_size: Vector2<u32>,
    ) -> Result<(), Error> {
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;

        let target_texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("target"),
            size: wgpu::Extent3d {
                width: image_size.x,
                height: image_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_texture_view = target_texture.create_view(&Default::default());

        let (_depth_texture, depth_texture_view) = create_depth_texture(&self.device, image_size);

        let pipeline = self.create_pipeline(format);

        let submission_index = self.render(&pipeline, &target_texture_view, &depth_texture_view);

        self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })?;

        tracing::info!(path = %output.as_ref().display(), "Writing rendered image");
        wgpu_cpu::dump_texture(&target_texture, output, None)?;

        Ok(())
    }

    fn render(
        &self,
        pipeline: &wgpu::RenderPipeline,
        target_texture_view: &wgpu::TextureView,
        depth_texture_view: &wgpu::TextureView,
    ) -> wgpu::SubmissionIndex {
        let mut command_encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target_texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            render_pass.set_pipeline(&pipeline);
            render_pass.set_bind_group(0, Some(&self.camera.bind_group), &[]);
            render_pass
                .set_index_buffer(self.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_vertex_buffer(0, self.mesh.vertex_buffer.slice(..));
            render_pass.draw_indexed(0..self.mesh.num_indices, 0, 0..1);
        }

        self.queue.submit([command_encoder.finish()])
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
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::Resized(physical_size) => {
                if let Some(window) = &mut self.window {
                    let window_size = Vector2::new(physical_size.width, physical_size.height);

                    window.configure(&self.device, window_size);

                    let (_, depth_texture_view) = create_depth_texture(&self.device, window_size);
                    window.depth_texture_view = depth_texture_view;

                    self.camera.set_window_size(window_size);
                }
            }
            WindowEvent::CloseRequested => {
                tracing::debug!("window close requested");
                event_loop.exit()
            }
            WindowEvent::Destroyed => {
                self.window = None;
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(window) = &self.window {
                    match window.surface.get_current_texture() {
                        Ok(target_texture) => {
                            self.camera.update(&self.queue);

                            let target_texture_view =
                                target_texture.texture.create_view(&Default::default());

                            self.render(
                                &window.pipeline,
                                &target_texture_view,
                                &window.depth_texture_view,
                            );
                            target_texture.present();
                        }
                        Err(error) => {
                            tracing::error!("Failed to get surface texture for drawing: {error}");
                            event_loop.exit();
                        }
                    }
                }
            }
            WindowEvent::KeyboardInput {
                device_id: _,
                event:
                    KeyEvent {
                        physical_key,
                        state,
                        ..
                    },
                is_synthetic: _,
            } => {
                match state {
                    ElementState::Pressed => {
                        if physical_key == PhysicalKey::Code(KeyCode::Escape) {
                            event_loop.exit();
                        }

                        self.keys_down.insert(physical_key);
                    }
                    ElementState::Released => {
                        self.keys_down.remove(&physical_key);
                    }
                }
            }
            _ => {}
        }

        for key in &self.keys_down {
            let move_speed = 0.1;

            match key {
                PhysicalKey::Code(KeyCode::KeyW) => {
                    self.camera.translate(Vector3::z() * move_speed);
                }
                PhysicalKey::Code(KeyCode::KeyA) => {
                    self.camera.translate(-Vector3::x() * move_speed);
                }
                PhysicalKey::Code(KeyCode::KeyS) => {
                    self.camera.translate(-Vector3::z() * move_speed);
                }
                PhysicalKey::Code(KeyCode::KeyD) => {
                    self.camera.translate(Vector3::x() * move_speed);
                }
                PhysicalKey::Code(KeyCode::Space) => {
                    self.camera.translate(Vector3::y() * move_speed);
                }
                PhysicalKey::Code(KeyCode::ShiftLeft) => {
                    self.camera.translate(-Vector3::y() * move_speed);
                }
                _ => {}
            }
        }

        if self.camera.changed
            && let Some(window) = &self.window
        {
            window.window.request_redraw();
        }
    }
}

#[derive(Debug)]
struct AppWindow {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    pipeline: wgpu::RenderPipeline,
    depth_texture_view: wgpu::TextureView,
}

impl AppWindow {
    fn configure(&self, device: &wgpu::Device, window_size: Vector2<u32>) {
        tracing::debug!(?window_size, "configure surface");

        self.surface.configure(
            device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.format,
                width: window_size.x,
                height: window_size.y,
                present_mode: wgpu::PresentMode::Immediate,
                desired_maximum_frame_latency: 0,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![],
            },
        );
    }
}

fn create_depth_texture(
    device: &wgpu::Device,
    size: Vector2<u32>,
) -> (wgpu::Texture, wgpu::TextureView) {
    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: size.x,
            height: size.y,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_texture_view = depth_texture.create_view(&Default::default());

    (depth_texture, depth_texture_view)
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct Vertex {
    position: Vector4<f32>,
    color: Vector4<f32>,
}

#[derive(Debug)]
struct Mesh {
    index_buffer: wgpu::Buffer,
    vertex_buffer: wgpu::Buffer,
    num_indices: u32,
    center: Point3<f32>,
    size: Vector3<f32>,
}

impl Mesh {
    fn load(path: impl AsRef<Path>, device: &wgpu::Device) -> Result<Self, Error> {
        // https://github.com/alecjacobson/common-3d-test-models
        let (models, _) = tobj::load_obj(path.as_ref(), &tobj::GPU_LOAD_OPTIONS)?;
        let mesh = models.into_iter().next().unwrap().mesh;

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("index buffer"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let positions = bytemuck::cast_slice::<f32, Point3<f32>>(&mesh.positions);

        let mut min = Vector3::max_value();
        let mut max = Vector3::min_value();
        for position in positions {
            min = min.zip_map(&position.coords, |a: f32, b: f32| a.min(b));
            max = max.zip_map(&position.coords, |a: f32, b: f32| a.max(b));
        }
        let size = max - min;
        let center = Point3::from(0.5 * (min + max));
        tracing::debug!(?center, ?size);

        let vertices = positions
            .iter()
            .map(|position| {
                Vertex {
                    position: position.to_homogeneous(),
                    color: Vector4::new(
                        (position.x - min.x) / size.x,
                        (position.y - min.y) / size.y,
                        (position.z - min.z) / size.z,
                        1.0,
                    ),
                }
            })
            .collect::<Vec<_>>();

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        tracing::info!(
            num_triangles = mesh.indices.len() / 3,
            num_indices = mesh.indices.len(),
            num_vertices = mesh.positions.len()
        );

        Ok(Mesh {
            index_buffer,
            vertex_buffer,
            num_indices: mesh.indices.len().try_into().unwrap(),
            center,
            size,
        })
    }
}

#[derive(Debug)]
struct Camera {
    transform: Isometry3<f32>,
    projection: Perspective3<f32>,
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
    changed: bool,
}

impl Camera {
    fn new(
        device: &wgpu::Device,
        image_size: Vector2<u32>,
        camera_bind_group_layout: &wgpu::BindGroupLayout,
        transform: Isometry3<f32>,
    ) -> Self {
        let projection = Perspective3::new(
            image_size.x as f32 / image_size.y as f32,
            FRAC_PI_4,
            0.001,
            100.0,
        );

        let data = CameraBufferData::new(&projection, &transform);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of(&data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera uniform"),
            layout: camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });

        Self {
            transform,
            projection,
            bind_group,
            buffer,
            changed: false,
        }
    }

    fn set_window_size(&mut self, window_size: Vector2<u32>) {
        self.projection
            .set_aspect(window_size.x as f32 / window_size.y as f32);
        self.changed = true;
    }

    fn translate(&mut self, translation: impl Into<Translation3<f32>>) {
        self.transform.append_translation_mut(&translation.into());
        self.changed = true;
    }

    fn update(&mut self, queue: &wgpu::Queue) {
        if self.changed {
            let data = CameraBufferData::new(&self.projection, &self.transform);
            queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&data));
            self.changed = false;
        }
    }
}

#[derive(Clone, Copy, Debug, Pod, Zeroable)]
#[repr(C)]
struct CameraBufferData {
    matrix: Matrix4<f32>,
}

impl CameraBufferData {
    fn new(projection: &Perspective3<f32>, transform: &Isometry3<f32>) -> Self {
        let mut projection = projection.to_homogeneous();
        // I think nalgebra assumes we're using a right-handed world coordinate system
        // and a left-handed NDC and thus flips the z-axis. Undo this here.
        projection[(2, 2)] *= -1.0;
        projection[(3, 2)] = 1.0;

        let matrix = projection * transform.inverse().to_homogeneous();

        Self { matrix }
    }
}
