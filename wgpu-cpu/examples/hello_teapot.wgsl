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
    let position = vec4f(
        scale * input.vertex_position.x,
        scale * input.vertex_position.y - 0.5,
        scale * input.vertex_position.z,
        1.0
    );
    return VertexOutput(
        position,
        //input.vertex_color,
        vec4f(
            position.x * 0.5 + 0.5,
            position.y * 0.5 + 0.5,
            position.z * 0.5 + 0.5,
            1.0
        )
    );
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front_face: bool) -> @location(0) vec4f {
    if front_face {
        return input.color;
    }
    else {
        return vec4f(input.color.b, 0.0, input.color.b, 1.0);
    }
}
