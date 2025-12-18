use color_eyre::eyre::{
    Error,
    bail,
};
use image::RgbaImage;

pub fn create_device_and_queue() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu_cpu::instance();

    let (_adapter, device, queue) = pollster::block_on(async {
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
        (adapter, device, queue)
    });

    (device, queue)
}

#[track_caller]
pub fn assert_eq_image(reference: &RgbaImage, rendered: &RgbaImage) {
    check_eq_image(reference, rendered).unwrap()
}

#[track_caller]
pub fn check_eq_image(reference: &RgbaImage, rendered: &RgbaImage) -> Result<(), Error> {
    if rendered.width() != reference.width() {
        bail!(
            "Image width mismatch:\n  Reference image size: [{}, {}]\n  Output image size: [{}, {}]",
            reference.width(),
            reference.height(),
            rendered.width(),
            rendered.height()
        );
    }
    if rendered.height() != reference.height() {
        bail!(
            "Image height mismatch:\n  Reference image size: [{}, {}]\n  Output image size: [{}, {}]",
            reference.width(),
            reference.height(),
            rendered.width(),
            rendered.height()
        );
    }

    for ((x, y, generated), (_, _, expected)) in rendered
        .enumerate_pixels()
        .zip(reference.enumerate_pixels())
    {
        if generated != expected {
            bail!(
                "Pixels differ at [{x}, {y}]:\n  Reference: {expected:?}\n  Output: {generated:?}"
            );
        }
    }

    Ok(())
}
