use approx::assert_abs_diff_eq;
use naga::BuiltIn;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    cranelift::CompiledModule,
    entry_point::EntryPointIndex,
};

#[track_caller]
fn compile(source: &str) -> CompiledModule {
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    crate::cranelift::compile(&module, &info).unwrap()
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

    let compiled_module = compile(&source);
    let entry_point = compiled_module.entry_point(EntryPointIndex::from(0));
    println!(
        "function pointer (scary): {:?}",
        entry_point.function_pointer()
    );

    #[derive(Debug)]
    struct VertexInput {
        vertex_index: u32,
    }

    impl ShaderInput for VertexInput {
        fn write_into(&self, binding: &naga::Binding, ty: &naga::Type, target: &mut [u8]) {
            println!("shader input: {binding:?} {ty:?}");
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
        fn read_from(&mut self, binding: &naga::Binding, ty: &naga::Type, source: &[u8]) {
            println!("shader output: {binding:?} {ty:?}");
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

    for i in 0..3 {
        entry_point.run(VertexInput { vertex_index: i }, &mut output);
    }

    assert_abs_diff_eq!(
        output.positions[..],
        [
            [-1.0f32, -1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, -1.0, 0.0, 1.0],
        ]
    );
}
