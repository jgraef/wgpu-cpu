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

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let scale = 0.4;
    return VertexOutput(
        vec4f(
            scale * input.vertex_position.x,
            -scale * input.vertex_position.y + 0.5,
            scale * input.vertex_position.z,
            1.0
        ),
        input.vertex_color,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
