use std::convert::Infallible;

use approx::assert_abs_diff_eq;
use naga::BuiltIn;

use crate::{
    bindings::{
        NullShaderIo,
        ShaderInput,
        ShaderOutput,
    },
    compiler::{
        CompiledModule,
        CompilerBackend,
        compile_clif,
        compile_clif_to_string,
        compile_jit,
        compiler::Config,
        product::EntryPointError,
        runtime::Runtime,
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
        entry_point
            .run(VertexInput { vertex_index: i }, &mut output)
            .unwrap();
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
    assert!(output.contains("function %main(i64, i32) -> f32x4"));
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

#[test]
fn switch_case_single() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a: i32 = 4;
            var b: i32;

            switch a {
                case 1: {
                    b = 2;
                }
                case 2, 3: {
                    b = 3;
                }
                case 4: {
                    b = 4;
                }
                default: {
                    b = -1;
                }
            }

            return Output(vec4f(), b);
        }
        "#,
    );
    assert_eq!(output, 4);
}

#[test]
fn switch_case_multiple() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a: i32 = 5;
            var b: i32;

            switch a {
                case 1: {
                    b = 2;
                }
                case 2, 3: {
                    b = 3;
                }
                case 4, 5, 6: {
                    b = 4;
                }
                default: {
                    b = 3;
                }
            }

            return Output(vec4f(), b);
        }
        "#,
    );
    assert_eq!(output, 4);
}

#[test]
fn switch_default() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a: i32 = 42;
            var b: i32;

            switch a {
                case 1: {
                    b = 2;
                }
                case 2, 3: {
                    b = 3;
                }
                case 4, 5, 6: {
                    b = 4;
                }
                default: {
                    b = 3;
                }
            }

            return Output(vec4f(), b);
        }
        "#,
    );
    assert_eq!(output, 3);
}

#[test]
fn switch_break() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a: i32 = 3;
            var b: i32;

            switch a {
                case 1: {
                    b = 2;
                }
                case 2, 3: {
                    b = 3;
                    break;
                    b = 123;
                }
                case 4, 5, 6: {
                    b = 4;
                }
                default: {
                    b = 12;
                }
            }

            return Output(vec4f(), b);
        }
        "#,
    );
    assert_eq!(output, 3);
}

#[test]
fn global_variable() {
    let output = helper().exec::<[i32; 4]>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: vec4i,
        }

        var<private> foo: vec4i = vec4i(1, 2, 3, 4);

        @vertex
        fn main() -> Output {
            foo.y = 123;
            return Output(vec4f(), foo);
        }
        "#,
    );
    assert_eq!(output, [1, 123, 3, 4]);
}

#[test]
fn access_index_vector_variable_read() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var foo = vec4i(1, 2, 3, 4);
            return Output(vec4f(), foo.y);
        }
        "#,
    );
    assert_eq!(output, 2);
}

#[test]
fn access_index_vector_variable_assign() {
    let output = helper().exec::<[i32; 4]>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: vec4i,
        }

        @vertex
        fn main() -> Output {
            var foo: vec4i = vec4i(1, 2, 3, 4);
            foo.y = 123;
            return Output(vec4f(), foo);
        }
        "#,
    );
    assert_eq!(output, [1, 123, 3, 4]);
}

#[test]
fn access_index_struct_variable_read() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        struct Foo {
            a: i32,
            b: i32,
        }

        @vertex
        fn main() -> Output {
            var foo = Foo(1, 2);
            return Output(vec4f(), foo.b);
        }
        "#,
    );
    assert_eq!(output, 2);
}

#[test]
fn access_index_struct_variable_assign() {
    let output = helper().exec::<[i32; 4]>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: vec4i,
        }

        struct Foo {
            a: i32,
            b: i32,
        }

        @vertex
        fn main() -> Output {
            var foo = Foo(1, 2);
            foo.b = 123;
            // vec2i composition triggers a bug in cranelift
            return Output(vec4f(), vec4i(foo.a, foo.b, 0, 0));
        }
        "#,
    );
    assert_eq!(output, [1, 123, 0, 0]);
}

#[test]
#[ignore = "https://github.com/bytecodealliance/wasmtime/issues/12197"]
fn insert_lane_into_i32x2_bug() {
    // https://github.com/bytecodealliance/wasmtime/issues/12197

    let source = r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: vec2i,
        }

        @vertex
        fn main() -> Output {
            var a = 1;
            var b = 2;
            return Output(vec4f(), vec2i(a, b));
        }
        "#;

    let clif = {
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
        let info = validator.validate(&module).unwrap();
        compile_clif_to_string(&module, &info, Default::default(), None).unwrap()
    };
    println!("{clif}");

    let output = helper().exec::<[i32; 2]>(source);
    assert_eq!(output, [1, 2]);
}

#[test]
#[ignore = "traps"]
fn trap_divide_by_zero() {
    let source = r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a = 123;
            var b = 0;
            var c = a / b;
            return Output(vec4f(), c);
        }
        "#;

    let _output = helper().exec::<i32>(source);
}

#[test]
#[ignore = "traps"]
fn trap_runtime_panic() {
    let source = r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            return Output(vec4f(), 0);
        }
        "#;
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    pub struct PanicInTheRuntime;
    impl Runtime for PanicInTheRuntime {
        type Error = Infallible;

        fn copy_inputs_to(&mut self, _target: &mut [u8]) -> Result<(), Self::Error> {
            panic!("copy_inputs_to")
        }

        fn copy_outputs_from(&mut self, _source: &[u8]) -> Result<(), Self::Error> {
            panic!("copy_outputs_from")
        }

        fn initialize_global_variables(
            &mut self,
            _private_data: &mut [u8],
        ) -> Result<(), Self::Error> {
            panic!("initialize_global_variables")
        }
    }

    let module = compile_jit(&module, &info).unwrap();
    let entry_point = module.entry_point(EntryPointIndex::from(0));
    entry_point.run_with_runtime(PanicInTheRuntime).unwrap();
}

#[test]
#[ignore = "traps"]
fn kill() {
    // note: wgsl discard statements translate to kill statements in naga IR. those
    // are only valid in fragment shaders though.

    let source = r#"
        @fragment
        fn main(@builtin(position) position: vec4f) -> @location(0) vec4f {
            discard;
        }
        "#;
    let module = naga::front::wgsl::parse_str(&source).unwrap();

    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    let module = compile_jit(&module, &info).unwrap();
    let entry_point = module.entry_point(EntryPointIndex::from(0));
    let result = entry_point.run(NullShaderIo, NullShaderIo);

    match result {
        Ok(()) => panic!("Expected shader invocation to be killed"),
        Err(EntryPointError::RuntimeError(runtime_error)) => {
            panic!("Unexpected runtime error: {runtime_error:?}")
        }
        Err(EntryPointError::Killed) => {
            // expected result
        }
    }
}

#[test]
fn access_global_variable_array_in_bounds_static() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        var<private> foo: array<i32, 4> = array<i32, 4>(4, 5, 6, 7);

        @vertex
        fn main() -> Output {
            var out = foo[2];
            return Output(vec4f(), out);
        }
        "#,
    );
    assert_eq!(output, 6);
}

#[test]
#[ignore = "dynamic indexing not implemented yet"]
fn access_global_variable_array_in_bounds_dynamic() {
    let output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        var<private> foo: array<i32, 4> = array<i32, 4>(4, 5, 6, 7);

        @vertex
        fn main() -> Output {
            var out = 0;
            for (var i = 0; i < 4; i += 1) {
                out += foo[i];
            }
            return Output(vec4f(), out);
        }
        "#,
    );
    assert_eq!(output, 22);
}

#[test]
#[ignore = "dynamic indexing not implemented yet"]
fn access_global_variable_array_out_of_bounds_dynamic() {
    // todo: get result or panic and check it

    let _output = helper().exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        var<private> foo: array<i32, 4> = array<i32, 4>(4, 5, 6, 7);

        @vertex
        fn main() -> Output {
            var out = 0;
            // oops, such an easy mistake to make :D
            for (var i = 0; i <= 4; i += 1) {
                out += foo[i];
            }
            return Output(vec4f(), out);
        }
        "#,
    );
}
