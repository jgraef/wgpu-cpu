use std::{
    f32::consts::FRAC_PI_4,
    path::PathBuf,
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
    Vector2,
    Vector3,
    Vector4,
};
use num_traits::Bounded;
use wgpu::util::DeviceExt;

#[derive(Debug, Parser)]
struct Args {
    #[clap(short, long, default_value = "teapot.png")]
    output: PathBuf,

    #[clap(long)]
    output_depth: Option<PathBuf>,
}

pub fn main() -> Result<(), Error> {
    let _ = dotenv();
    color_eyre::install()?;
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    tracing::info!(?args, "Hello Teapot!");

    let image_size = Vector2::new(600, 400);

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let size = wgpu::Extent3d {
        width: image_size.x,
        height: image_size.y,
        depth_or_array_layers: 1,
    };

    let instance = wgpu_cpu::instance(Default::default());

    let (_adapter, device, queue) = pollster::block_on(async {
        let adapter = instance.request_adapter(&Default::default()).await?;
        let (device, queue) = adapter.request_device(&Default::default()).await?;
        Ok::<_, Error>((adapter, device, queue))
    })?;

    let (mesh, center) = {
        // https://raw.githubusercontent.com/rbarril75/Scratched-Blue-Teapot/refs/heads/master/teapot.obj
        let (models, _) = tobj::load_obj("wgpu-cpu/examples/teapot.obj", &tobj::GPU_LOAD_OPTIONS)?;
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
        let center = Point3::from(0.5 * (min + max));

        let vertices = positions
            .iter()
            .map(|position| {
                Vertex {
                    position: position.to_homogeneous(),
                    color: Vector4::new(
                        (position.x - min.x) / (max.x - min.x),
                        (position.y - min.y) / (max.y - min.y),
                        (position.z - min.z) / (max.z - min.z),
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

        let mesh = Mesh {
            index_buffer,
            vertex_buffer,
            num_indices: mesh.indices.len().try_into().unwrap(),
        };

        (mesh, center)
    };

    let camera_buffer = {
        let projection = Perspective3::new(
            image_size.x as f32 / image_size.y as f32,
            FRAC_PI_4,
            0.1,
            50.0,
        );

        let mut matrix = projection.to_homogeneous();
        // nalgebra assumes we're using a right-handed world coordinate system and a
        // left-handed NDC and thus flips the z-axis. Undo this here.
        matrix[(2, 2)] *= -1.0;
        matrix[(3, 2)] = 1.0;

        tracing::debug!(?center);
        let transform = Isometry3::face_towards(
            &(center + Vector3::new(0.0, 2.0, -8.0)),
            &center,
            &Vector3::y(),
        );

        let matrix = matrix * transform.inverse().to_homogeneous();

        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera uniform"),
            contents: bytemuck::bytes_of::<Matrix4<f32>>(&matrix),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    };

    let shader_module = device.create_shader_module(wgpu::include_wgsl!("hello_teapot.wgsl"));

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

    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("camera uniform"),
        layout: &camera_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hello_triangle pipeline layout"),
        bind_group_layouts: &[&camera_bind_group_layout],
        immediate_size: 0,
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hello_triangle pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader_module,
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
            module: &shader_module,
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

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_texture_view = target_texture.create_view(&Default::default());

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_texture_view = depth_texture.create_view(&Default::default());

    let mut command_encoder = device.create_command_encoder(&Default::default());

    {
        let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_texture_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_texture_view,
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
        render_pass.set_bind_group(0, Some(&camera_bind_group), &[]);
        render_pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
    }

    let submission_index = queue.submit([command_encoder.finish()]);

    let poll_result = device.poll(wgpu::PollType::Wait {
        submission_index: Some(submission_index),
        timeout: None,
    });
    tracing::debug!(?poll_result);

    tracing::info!(path = %args.output.display(), "Writing rendered image");
    wgpu_cpu::dump_texture(&target_texture, &args.output, None)?;

    if let Some(output) = &args.output_depth {
        tracing::info!(path = %output.display(), "Writing rendered image");
        wgpu_cpu::dump_texture(&depth_texture, output, None)?;
    }

    tracing::info!("Example quitting");

    Ok(())
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
}
