struct VertexInput {
    @builtin(vertex_index)
    vertex_index: u32,
}

struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    @interpolate(linear, sample)
    color: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let x = 0.8 * f32(i32(input.vertex_index) - 1);
    let y = 0.8 * f32(i32(input.vertex_index & 1u) * 2 - 1);
    let position = vec4f(x, y, 0.0, 1.0);

    let r = f32(input.vertex_index == 0);
    let g = f32(input.vertex_index == 1);
    let b = f32(input.vertex_index == 2);
    let color = vec4f(r, g, b, 1.0);

    return VertexOutput(
        position,
        color,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
    //return vec4f(1.0, 0.0, 0.0, 1.0);
}
