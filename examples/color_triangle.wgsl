struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec4f,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var output: VertexOutput;

    output.position.x = f32(i32(in_vertex_index) - 1);
    output.position.y = f32(i32(in_vertex_index & 1u) * 2 - 1);

    output.color[in_vertex_index % 3] = 1.0;
    output.color.w = 1.0;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(1.0, 0.0, 0.0, 1.0);
}
