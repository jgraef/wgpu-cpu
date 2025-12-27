struct VertexInput {
    @builtin(vertex_index)
    vertex_index: u32,

    @location(0)
    vertex_position: vec4f,
    @location(1)
    vertex_color: vec4f,
}

struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    @interpolate(linear, sample)
    color: vec4f,
}

struct Camera {
    matrix: mat4x4f,
}

@group(0)
@binding(0)
var<uniform> camera: Camera;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let position = camera.matrix * input.vertex_position;

    return VertexOutput(
        position,
        input.vertex_color,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
