use std::{
    fmt::Debug,
    ops::Range,
};

use approx::assert_abs_diff_eq;
use naga::{
    BuiltIn,
    Scalar,
    ScalarKind,
    TypeInner,
    proc::{
        Alignment,
        Layouter,
    },
};

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    entry_point::EntryPointIndex,
    interpreter::{
        InterpretedModule,
        Interpreter,
        InterpreterBackend,
    },
    make_tests,
    memory::NullMemory,
};

#[test]
#[ignore = "will ask"]
fn naga_bool_width_is_32bit() {
    // naga bools are 1 byte. this might be wrong?
    //
    // The WebGPU spec specifies [here][1] that bools are 4 bytes (size and
    // alignment). But it also mentions [here][2] that they're not host-shareable,
    // so the internal layout is not specified.
    //
    // [1]: https://gpuweb.github.io/gpuweb/wgsl/#alignment-and-size
    // [2]: https://gpuweb.github.io/gpuweb/wgsl/#internal-value-layout

    let module = naga::front::wgsl::parse_str(
        r#"
    fn main() -> bool {
        var x: bool = true;
        return x;
    }
    "#,
    )
    .unwrap();

    let mut layouter = Layouter::default();
    layouter.update(module.to_ctx()).unwrap();

    let mut found_bool = false;
    for (handle, ty) in module.types.iter() {
        match &ty.inner {
            TypeInner::Scalar(Scalar {
                kind: ScalarKind::Bool,
                width,
            }) => {
                let type_layout = layouter[handle];

                assert_eq!(*width, 4, "scalar width of bool is not 4");
                assert_eq!(type_layout.size, 4, "SizeOf(bool) is not 4");
                assert_eq!(
                    type_layout.alignment,
                    Alignment::FOUR,
                    "AlignOf(bool) is not 4"
                );

                found_bool = true;
            }
            _ => {}
        }
        println!("{ty:?}");
    }

    assert!(found_bool, "didn't find bool in module");
}

#[track_caller]
fn run_vertex_shader(source: &str, vertices: Range<u32>) -> Vec<[f32; 4]> {
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();
    let module = InterpretedModule::new(module, info).unwrap();
    let mut interpreter = Interpreter::new(module, NullMemory, EntryPointIndex::from(0));

    #[derive(Debug)]
    struct VertexInput {
        vertex_index: u32,
    }

    impl ShaderInput for VertexInput {
        fn write_into(&self, binding: &naga::Binding, _ty: &naga::Type, target: &mut [u8]) {
            match binding {
                naga::Binding::BuiltIn(BuiltIn::VertexIndex) => {
                    *bytemuck::from_bytes_mut(target) = self.vertex_index;
                }
                _ => {}
            }
        }
    }

    #[derive(Debug, Default)]
    struct VertexOutput {
        pub positions: Vec<[f32; 4]>,
    }

    impl ShaderOutput for VertexOutput {
        fn read_from(&mut self, binding: &naga::Binding, _ty: &naga::Type, source: &[u8]) {
            match binding {
                naga::Binding::BuiltIn(BuiltIn::Position { invariant: _ }) => {
                    let position = bytemuck::from_bytes::<[f32; 4]>(source);
                    self.positions.push(*position);
                }
                _ => {}
            }
        }
    }

    let mut output = VertexOutput::default();

    for i in vertices {
        interpreter.run_entry_point(VertexInput { vertex_index: i }, &mut output);
    }

    output.positions
}

#[test]
fn vertex_triangle() {
    let source = r#"
    @vertex
    fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
        let x = f32(i32(vertex_index) - 1);
        let y = f32(i32(vertex_index & 1u) * 2 - 1);
        return vec4f(x, y, 0.0, 1.0);
    }
    "#;

    let vertices = run_vertex_shader(source, 0..3);
    assert_abs_diff_eq!(
        vertices[..],
        [
            [-1.0f32, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
        ]
    );
}

make_tests!(
    InterpreterBackend => (
        init_variable,
        store_variable,
        casts,
        binops_scalars,
        comparisions,
        unops,
        if_stmt,
        early_return,
        if_early_return,
    )
);
