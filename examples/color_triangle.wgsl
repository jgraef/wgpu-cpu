/*struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec4f,
}*/

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4f {
    /*var output: VertexOutput;

    output.position.x = f32(i32(in_vertex_index) - 1);
    output.position.y = f32(i32(in_vertex_index & 1u) * 2 - 1);

    output.color[in_vertex_index % 3] = 1.0;
    output.color.w = 1.0;

    return output;*/

    /*let x = f32(i32(in_vertex_index) - 1);
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1);

    return VertexOutput(
        vec4f(x, y, 0.0, 1.0),
        vec4f(1.0, 0.0, 0.0, 1.0),
    );*/

    let x = f32(i32(in_vertex_index) - 1);
    let y = f32(i32(in_vertex_index & 1u) * 2 - 1);
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4f) -> @location(0) vec4f {
    //return vec4f(position.x, position.y, 0.0, 1.0);
    return position;
}
