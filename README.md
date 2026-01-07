
# Software rendering backend for wgpu

This is a custom backend for [wgpu][1] which renders everything on the CPU. It aims to support all features that are usually guaranteed by [wgpu][1] to be supported, and some more that we can manage to implement. The project is still **under development**, so it currently only supports some basic usecases.

To facilitate running shaders on the CPU, this repository also contains `naga-cranelift`, a [cranelift][2]-powered compiler backend for [naga][3].

Examples can be found in [`wgpu-cpu/examples`](https://github.com/jgraef/wgpu-cpu/tree/main/wgpu-cpu/examples).

| ![Rendering of the Utah Teapot with interpolated vertex colors that result in a red, green and blue gradient.](https://media.githubusercontent.com/media/jgraef/wgpu-cpu/refs/heads/main/doc/teapot.png) |
|:--:|
| Utah Teapot with interpolated vertex colors ([example](https://github.com/jgraef/wgpu-cpu/blob/main/wgpu-cpu/examples/hello_mesh.rs)) |

| ![Rendering of the Stanford Bunny with a colorful test card texture.](https://media.githubusercontent.com/media/jgraef/wgpu-cpu/refs/heads/main/doc/texture.png) |
|:--:|
| Stanford Bunny with texture ([example](https://github.com/jgraef/wgpu-cpu/blob/main/wgpu-cpu/examples/hello_texture.rs)) |

[1]: https://crates.io/crates/wgpu
[2]: https://cranelift.dev/
[3]: https://docs.rs/naga/latest/naga/index.html
