use approx::assert_abs_diff_eq;
use naga::BuiltIn;

use crate::{
    bindings::{
        ShaderInput,
        ShaderOutput,
    },
    compiler::{
        CompiledModule,
        CompilerBackend,
        compile_clif,
        compiler::Config,
    },
    entry_point::EntryPointIndex,
    make_tests,
    util::test::BackendTestHelper,
};

#[track_caller]
fn compile(source: &str) -> CompiledModule {
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    crate::compiler::compile_jit(&module, &info).unwrap()
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

make_tests!(
    CompilerBackend => (
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

fn helper() -> BackendTestHelper<CompilerBackend> {
    BackendTestHelper(CompilerBackend::default())
}

#[test]
fn function_call() {
    let output = helper().exec::<i32>(
        r#"
        fn my_sum(a: i32, b: i32) -> i32 {
            return a + b;
        }

        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            let output = my_sum(13, 12);
            return Output(vec4f(), output);
        }
        "#,
    );
    assert_eq!(output, 25);
}

#[test]
fn clif_output() {
    let source = r#"
    @vertex
    fn main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4f {
        let x = f32(i32(vertex_index) - 1);
        let y = f32(i32(vertex_index & 1u) * 2 - 1);
        return vec4f(x, y, 0.0, 1.0);
    }
    "#;

    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    let mut output = vec![];
    compile_clif(
        &module,
        &info,
        Config {
            collect_debug_info: true,
            ..Default::default()
        },
        None,
        &mut output,
    )
    .unwrap();
    let output = String::from_utf8(output).unwrap();

    println!("{output}");
    assert!(output.contains("function %main(i32) -> f32x4"));
}

#[test]
fn scalar_constant() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        const foo: i32 = 1234 * 2;

        @vertex
        fn main() -> Output {
            var bar: i32 = foo;
            return Output(vec4f(), bar);
        }
        "#,
    );
    assert_eq!(output, 2468);
}

#[test]
#[ignore = "wip"]
fn vector_constant() {
    let output = helper().exec::<[i32; 4]>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: vec4i,
        }

        const foo: vec4i = vec4i(1, 2, 3, 4);

        @vertex
        fn main() -> Output {
            var bar: vec4i = foo;
            return Output(vec4f(), bar);
        }
        "#,
    );
    assert_eq!(output, [1, 2, 3, 4]);
}

#[test]
fn loops() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var fac: i32 = 1;

            for (var i = 2; i <= 6; i += 1) {
                fac *= i;
            }

            return Output(vec4f(), fac);
        }
        "#,
    );
    assert_eq!(output, 720);
}
