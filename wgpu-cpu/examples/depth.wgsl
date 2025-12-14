struct VertexInput {
    @builtin(vertex_index)
    vertex_index: u32,
}

struct VertexOutput {
    @builtin(position)
    position: vec4f,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let vertex_index = input.vertex_index % 3;
    let triangle = input.vertex_index / 3;

    let scale = 0.8;
    let offset = 0.1 * f32(triangle);
    let z = f32(triangle) / 10.0;

    let x = offset + scale * f32(i32(vertex_index) - 1);
    let y = offset + scale * f32(i32(vertex_index & 1u) * 2 - 1);
    let position = vec4f(x, y, z, 1.0);

    return VertexOutput(
        position,
    );
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let z = input.position.z;
    return vec4f(1.0 - z, 0.0, z, 1.0);
}
