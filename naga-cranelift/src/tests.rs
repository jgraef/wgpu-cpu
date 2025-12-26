use std::{
    convert::Infallible,
    fmt::Debug,
};

use approx::{
    AbsDiffEq,
    assert_abs_diff_eq,
};
use arrayvec::ArrayVec;
use bytemuck::Pod;
use naga::BuiltIn;

use crate::{
    CompiledModule,
    bindings::{
        NullBinding,
        ShaderInput,
        ShaderOutput,
    },
    compile_clif,
    compile_clif_to_string,
    compile_jit,
    compiler::Config,
    product::{
        EntryPointError,
        EntryPointIndex,
    },
    runtime::{
        DefaultRuntimeError,
        Runtime,
    },
};

#[track_caller]
fn compile(source: &str) -> CompiledModule {
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();

    crate::compile_jit(&module, &info).unwrap()
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
        pub vertex_index: u32,
    }

    #[derive(Debug, Default)]
    struct VertexOutput {
        pub positions: ArrayVec<[f32; 4], 3>,
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

    impl ShaderOutput for VertexOutput {
        fn read_from(&mut self, binding: &naga::Binding, ty: &naga::Type, source: &[u8]) {
            println!("shader output: {binding:?} {ty:?}");
            match binding {
                naga::Binding::BuiltIn(BuiltIn::Position { invariant: _ }) => {
                    self.positions
                        .push(*bytemuck::from_bytes::<[f32; 4]>(source));
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

#[track_caller]
pub fn try_exec<T>(source: &str) -> Result<T, EntryPointError<DefaultRuntimeError>>
where
    T: Pod,
{
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();
    let module = compile_jit(&module, &info).unwrap();

    struct EvalOutput<T> {
        output: T,
    }

    impl<T> ShaderOutput for EvalOutput<T>
    where
        T: Pod,
    {
        fn read_from(&mut self, binding: &naga::Binding, _ty: &naga::Type, source: &[u8]) {
            match binding {
                naga::Binding::Location { location: 0, .. } => {
                    self.output = *bytemuck::from_bytes::<T>(source);
                }
                _ => {}
            }
        }
    }

    let mut output = EvalOutput {
        output: T::zeroed(),
    };
    module
        .entry_point(EntryPointIndex::from(0))
        .run(NullBinding, &mut output)?;

    Ok(output.output)
}

#[track_caller]
pub fn exec<T>(source: &str) -> T
where
    T: Pod,
{
    try_exec(source).unwrap()
}

#[track_caller]
pub fn eval<T>(expression: &str, preamble: &str, out_ty: &str) -> T
where
    T: Pod,
{
    let source = format!(
        r#"
        struct Output {{
            @builtin(position) p: vec4f,
            @location(0) output: {out_ty},
        }}

        @vertex
        fn main(@builtin(vertex_index) vertex_index: u32) -> Output {{
            {preamble}
            return Output(vec4f(), {expression});
        }}
        "#
    );

    exec(&source)
}

#[track_caller]
pub fn eval_bool(expression: &str, preamble: &str) -> bool {
    let output: u32 = eval::<u32>(&format!("u32({expression})"), preamble, "u32");
    match output {
        0 => false,
        1 => true,
        x => panic!("invalid bool: {x}"),
    }
}

#[track_caller]
pub fn assert_wgsl(assertion: &str, preamble: &str) {
    let output = eval_bool(assertion, preamble);
    assert!(output);
}

#[test]
pub fn init_variable() {
    let a = eval::<u32>("a", "var a: u32 = 123;", "u32");
    assert_eq!(a, 123);
}

#[test]
pub fn store_variable() {
    let a = eval::<u32>("a", "var a: u32; a = 123;", "u32");
    assert_eq!(a, 123);
}

#[test]
pub fn casts() {
    #[track_caller]
    pub fn test_cast(value: &str, input_ty: &str, output_ty: &str, expected: &str) {
        assert_wgsl(
            &format!("output == {expected}"),
            &format!(
                r#"
        var input: {input_ty} = {value};
        var output: {output_ty} = {output_ty}(input);
        "#
            ),
        );
    }

    test_cast("false", "bool", "u32", "0");
    test_cast("true", "bool", "u32", "1");
    test_cast("false", "bool", "i32", "0");
    test_cast("true", "bool", "i32", "1");
    test_cast("false", "bool", "f32", "0.0");
    test_cast("true", "bool", "f32", "1.0");
    test_cast("5", "u32", "f32", "5.0");
    test_cast("-3", "i32", "f32", "-3.0");
}

#[test]
pub fn binops_scalars() {
    #[track_caller]
    pub fn test_binop<T>(ty: &str, left: &str, op: &str, right: &str, expected: T)
    where
        T: Pod + AbsDiffEq + Debug,
    {
        let output = eval::<T>(
            &format!("left {op} right"),
            &format!(
                r#"
        var left: {ty} = {left};
        var right: {ty} = {right};
        "#
            ),
            ty,
        );

        assert_abs_diff_eq!(output, expected);
    }

    test_binop::<i32>("i32", "1", "+", "1", 2);
    test_binop::<i32>("i32", "2", "-", "1", 1);
    test_binop::<i32>("i32", "1", "-", "2", -1);
    test_binop::<i32>("i32", "2", "*", "3", 6);
    test_binop::<i32>("i32", "2", "*", "-3", -6);
    test_binop::<i32>("i32", "6", "/", "2", 3);
    test_binop::<i32>("i32", "3", "/", "2", 1);
    test_binop::<i32>("i32", "3", "%", "2", 1);

    test_binop::<f32>("f32", "1", "+", "1", 2.0);
    test_binop::<f32>("f32", "2", "-", "1", 1.0);
    test_binop::<f32>("f32", "1", "-", "2", -1.0);
    test_binop::<f32>("f32", "2", "*", "3", 6.0);
    test_binop::<f32>("f32", "2", "*", "-3", -6.0);
    test_binop::<f32>("f32", "6", "/", "2", 3.0);
    test_binop::<f32>("f32", "3", "/", "2", 1.5);
    test_binop::<f32>("f32", "3", "%", "2", 1.0);
}

#[test]
pub fn comparisions() {
    #[track_caller]
    pub fn test_compare(ty: &str, left: &str, cmp: &str, right: &str, expected: bool) {
        let output = eval_bool(
            &format!("left {cmp} right"),
            &format!(
                r#"
        var left: {ty} = {left};
        var right: {ty} = {right};
        "#
            ),
        );

        assert_eq!(output, expected, "{left} {cmp} {right}");
    }

    test_compare("i32", "2", "==", "2", true);
    test_compare("i32", "1", "==", "2", false);
    test_compare("i32", "1", "!=", "2", true);
    test_compare("i32", "1", "<", "2", true);
    test_compare("i32", "2", ">", "1", true);
    test_compare("i32", "1", "<=", "2", true);
    test_compare("i32", "2", "<=", "2", true);
    test_compare("i32", "2", ">=", "2", true);
    test_compare("i32", "3", ">=", "2", true);
    test_compare("i32", "-1", "<", "1", true);
}

#[test]
pub fn unops() {
    #[track_caller]
    pub fn test_unop<T>(ty: &str, op: &str, input: &str, expected: T)
    where
        T: Pod + AbsDiffEq + Debug,
    {
        let output = eval::<T>(
            &format!("{op} input"),
            &format!(
                r#"
        var input: {ty} = {input};
        "#
            ),
            ty,
        );

        assert_abs_diff_eq!(output, expected);
    }

    #[track_caller]
    pub fn test_bool_unop(op: &str, input: &str, expected: bool) {
        let output = eval_bool(
            &format!("{op} input"),
            &format!(
                r#"
        var input: bool = {input};
        "#
            ),
        );

        assert_eq!(output, expected);
    }

    test_unop::<i32>("i32", "-", "123", -123);
    test_unop::<i32>("i32", "-", "-123", 123);
    test_unop::<f32>("f32", "-", "123.0", -123.0);
    test_unop::<f32>("f32", "-", "-123.0", 123.0);
    test_bool_unop("!", "true", false);
    test_bool_unop("!", "false", true);
    test_unop("u32", "~", "123", !123);
}

#[test]
pub fn if_stmt() {
    assert_wgsl(
        "x == 1",
        "var x: u32; var c: bool = true; if c { x = 1; } else { x = 2; }",
    );
    assert_wgsl(
        "x == 2",
        "var x: u32; var c: bool = false; if c { x = 1; } else { x = 2; }",
    );
}

#[test]
pub fn early_return() {
    let out = eval::<u32>("123", "return Output(vec4f(), 456);", "u32");
    assert_eq!(out, 456);
}

#[test]
pub fn if_early_return() {
    let out = eval::<u32>(
        "123",
        "var c: bool = true; if c { return Output(vec4f(), 456); }",
        "u32",
    );
    assert_eq!(out, 456);
}

#[test]
fn function_call() {
    let output = exec::<i32>(
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
    // expected arguments: runtime pointer, result pointer, normal function
    // arguments expected return values: abort code
    assert!(output.contains("function %main(i64, i64, i32) -> i8"));
}

#[test]
fn scalar_constant() {
    let output = exec::<i32>(
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
    let output = exec::<[i32; 4]>(
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
    let output = exec::<i32>(
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
    let output = exec::<i32>(
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
    let output = exec::<i32>(
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
    let output = exec::<i32>(
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
    let output = exec::<i32>(
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
    let output = exec::<[i32; 4]>(
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
    let output = exec::<i32>(
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
    let output = exec::<[i32; 4]>(
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
    let output = exec::<i32>(
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
    let output = exec::<[i32; 4]>(
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

    let output = exec::<[i32; 2]>(source);
    assert_eq!(output, [1, 2]);
}

#[test]
fn trap_divide_by_zero() {
    let source = r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var a: i32 = 123;
            var b: i32 = 0;
            var c: i32 = a / b;
            return Output(vec4f(), c);
        }
        "#;
    let result = try_exec::<i32>(source);

    match result {
        Err(EntryPointError::DivisionByZero) => {
            // expected result
        }
        unexpected => panic!("Unexpected result: {unexpected:?}"),
    }
}

#[test]
#[should_panic(expected = "copy_outputs_from")]
fn runtime_panic() {
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
            // note: this is never called because the shader doesn't take any inputs
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

        fn buffer(&mut self, _binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
            panic!("buffer")
        }

        fn buffer_mut(
            &mut self,
            _binding: naga::ResourceBinding,
        ) -> Result<&mut [u8], Self::Error> {
            panic!("buffer_mut")
        }
    }

    let module = compile_jit(&module, &info).unwrap();
    let entry_point = module.entry_point(EntryPointIndex::from(0));
    entry_point.run_with_runtime(PanicInTheRuntime).unwrap();
}

#[test]
fn kill() {
    // note: wgsl discard statements translate to kill statements in naga IR. those
    // are only valid in fragment shaders though.

    let source = r#"
        @fragment
        fn main(@builtin(position) position: vec4f) -> @location(0) vec4f {
            discard;
        }
        "#;
    let result = try_exec::<i32>(source);

    match result {
        Err(EntryPointError::Killed) => {
            // expected result
        }
        unexpected => panic!("Unexpected result: {unexpected:?}"),
    }
}

#[test]
fn access_global_variable_array_in_bounds_static() {
    let output = exec::<i32>(
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
fn access_global_variable_array_in_bounds_dynamic() {
    let output = exec::<i32>(
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
#[ignore = "traps"]
fn access_global_variable_array_out_of_bounds_dynamic() {
    // todo: get result or panic and check it

    let _output = exec::<i32>(
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

#[test]
fn array_length() {
    let source = r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: u32,
        }

        @group(0)
        @binding(0)
        var<storage, read> what_len_is_this_array: array<i32>;

        @vertex
        fn main() -> Output {
            let out = arrayLength(&what_len_is_this_array);
            return Output(vec4f(), out);
        }
        "#;

    let module = naga::front::wgsl::parse_str(&source).unwrap();
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();
    let module = compile_jit(&module, &info).unwrap();

    struct DynamicArrayRuntime<'a> {
        buffer: Vec<i32>,
        output: &'a mut u32,
    }

    impl<'a> Runtime for DynamicArrayRuntime<'a> {
        type Error = Infallible;

        fn copy_inputs_to(&mut self, target: &mut [u8]) -> Result<(), Self::Error> {
            target.fill(0);
            Ok(())
        }

        fn copy_outputs_from(&mut self, source: &[u8]) -> Result<(), Self::Error> {
            // too lazy to actually look at the compiled stack layout. pretty sure that's
            // where the output is
            *self.output = *bytemuck::from_bytes(&source[16..20]);
            Ok(())
        }

        fn initialize_global_variables(
            &mut self,
            private_data: &mut [u8],
        ) -> Result<(), Self::Error> {
            private_data.fill(0);
            Ok(())
        }

        fn buffer(&mut self, binding: naga::ResourceBinding) -> Result<&[u8], Self::Error> {
            assert_eq!(binding.group, 0);
            assert_eq!(binding.binding, 0);

            Ok(bytemuck::cast_slice(&*self.buffer))
        }

        fn buffer_mut(
            &mut self,
            _binding: naga::ResourceBinding,
        ) -> Result<&mut [u8], Self::Error> {
            Ok(&mut [])
        }
    }

    let mut output = 0;
    module
        .entry_point(EntryPointIndex::from(0))
        .run_with_runtime(DynamicArrayRuntime {
            buffer: vec![42i32; 123],
            output: &mut output,
        })
        .unwrap();

    assert_eq!(output, 123);
}

#[test]
fn select() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var out = 0;
            var a = 123;
            var b = 456;
            var c = true;
            out = select(a, b, c);
            return Output(vec4f(), out);
        }
        "#,
    );
    assert_eq!(output, 456);

    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var out = 0;
            var a = 123;
            var b = 456;
            var c = false;
            out = select(a, b, c);
            return Output(vec4f(), out);
        }
        "#,
    );
    assert_eq!(output, 123);
}

#[test]
fn vector_access_dynamic() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var v: vec4i = vec4i(12, 23, 34, 45);
            var s: i32 = 0;
            for (var i = 0; i < 4; i += 1) {
                s += (i + 1) * v[i];
            }
            return Output(vec4f(), s);
        }
        "#,
    );
    assert_eq!(output, 340);
}

#[test]
fn vector_access_static() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        @vertex
        fn main() -> Output {
            var v: vec4i = vec4i(12, 23, 34, 45);
            var s: i32 = v.x + 2 * v.y + 3 * v.z + 4 * v.w;
            return Output(vec4f(), s);
        }
        "#,
    );
    assert_eq!(output, 340);
}

#[test]
fn vectorized_function_argument() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        fn do_stuff(input: vec4i) -> i32 {
            return input.y;
        }

        @vertex
        fn main() -> Output {
            let output = do_stuff(vec4i(1, 2, 3, 4));
            return Output(
                vec4f(),
                output,
            );
        }
        "#,
    );

    assert_eq!(output, 2);
}

#[test]
fn return_if_else_diverging() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        fn do_stuff(x: i32) -> i32 {
            if x == 1234 {
                return 45;
            }
            else {
                return 67;
            }
        }

        @vertex
        fn main() -> Output {
            let output = do_stuff(1234);
            return Output(
                vec4f(),
                output,
            );
        }
        "#,
    );

    assert_eq!(output, 45);
}

#[test]
fn return_from_loop_body_diverging() {
    let output = exec::<i32>(
        r#"
        struct Output {
            @builtin(position) p: vec4f,
            @location(0) output: i32,
        }

        fn do_stuff() -> i32 {
            loop {
                return 45;
            }
            return 123;
        }

        @vertex
        fn main() -> Output {
            let output = do_stuff();
            return Output(
                vec4f(),
                output,
            );
        }
        "#,
    );

    assert_eq!(output, 45);
}
