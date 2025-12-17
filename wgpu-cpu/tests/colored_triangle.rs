use std::path::Path;

use image::{
    ImageReader,
    RgbaImage,
    buffer::ConvertBuffer,
};
use wgpu_cpu::image::rgba_texture_image;

fn load_reference(path: impl AsRef<Path>) -> RgbaImage {
    let path = path.as_ref();
    ImageReader::open(Path::new("tests").join(path))
        .unwrap_or_else(|error| {
            panic!(
                "Could not open reference image: {}: {error}",
                path.display()
            )
        })
        .decode()
        .unwrap_or_else(|error| {
            panic!(
                "Could not decode reference image: {}: {error}",
                path.display()
            )
        })
        .to_rgba8()
}

fn create_device_and_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu_cpu::instance();

    let (_adapter, device, queue) = pollster::block_on(async {
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        (adapter, device, queue)
    });

    (device, queue)
}

fn render_colored_triangle() -> RgbaImage {
    let (device, queue) = create_device_and_queue();

    let shader_module = device.create_shader_module(wgpu::include_wgsl!("colored_triangle.wgsl"));

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("hello_triangle pipeline layout"),
        bind_group_layouts: &[],
        immediates_ranges: &[],
    });

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let size = wgpu::Extent3d {
        width: 512,
        height: 512,
        depth_or_array_layers: 1,
    };

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

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("hello_triangle pipeline"),
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
            front_face: wgpu::FrontFace::Cw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
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
        render_pass.draw(0..3, 0..1);
    }

    let submission_index = queue.submit([command_encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission_index),
            timeout: None,
        })
        .unwrap();

    rgba_texture_image(&target_texture).convert()
}

#[track_caller]
fn assert_eq_image(reference: &RgbaImage, rendered: &RgbaImage) {
    assert_eq!(rendered.width(), reference.width());
    assert_eq!(rendered.height(), reference.height());

    rendered
        .enumerate_pixels()
        .zip(reference.enumerate_pixels())
        .for_each(|((x, y, generated), (_, _, expected))| {
            assert_eq!(generated, expected, "Pixels differ at [{x}, {y}]");
        });
}

#[test]
fn t_colored_triangle() {
    let reference = load_reference("colored_triangle.png");
    let rendered = render_colored_triangle();
    assert_eq_image(&reference, &rendered);
}
