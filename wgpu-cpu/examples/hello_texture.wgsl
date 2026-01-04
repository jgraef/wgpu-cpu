struct VertexInput {
    @builtin(vertex_index)
    vertex_index: u32,

    @location(0)
    vertex_position: vec4f,

    @location(1)
    uv: vec2f,
}

struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    @interpolate(linear, sample)
    uv: vec2f,
}

struct Camera {
    matrix: mat4x4f,
}

@group(0)
@binding(0)
var<uniform> camera: Camera;

@group(1)
@binding(0)
var texture_albedo: texture_2d<f32>;

@group(1)
@binding(1)
var sampler_albedo: sampler;

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let position = camera.matrix * input.vertex_position;

    return VertexOutput(
        position,
        input.uv,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    var color = textureSample(texture_albedo, sampler_albedo, input.uv);
    // todo
    //return vec4f(1.0, 0.0, 1.0, 1.0);
    return color;
}
