use std::{
    fmt::Debug,
    ops::Range,
};

use approx::{
    AbsDiffEq,
    assert_abs_diff_eq,
};
use bytemuck::Pod;
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
    },
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

#[track_caller]
fn eval<T>(expression: &str, preamble: &str, out_ty: &str) -> T
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

    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
        println!("{source}");
        panic!("{e}");
    });
    let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
    let info = validator.validate(&module).unwrap();
    let module = InterpretedModule::new(module, info).unwrap();
    let mut interpreter = Interpreter::new(module, NullMemory, EntryPointIndex::from(0));

    struct EvalInput;

    impl ShaderInput for EvalInput {
        fn write_into(&self, _binding: &naga::Binding, _ty: &naga::Type, _target: &mut [u8]) {
            // nop
        }
    }

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

    interpreter.run_entry_point(EvalInput, &mut output);

    output.output
}

#[track_caller]
fn eval_bool(expression: &str, preamble: &str) -> bool {
    let output: u32 = eval::<u32>(&format!("u32({expression})"), preamble, "u32");
    match output {
        0 => false,
        1 => true,
        x => panic!("invalid bool: {x}"),
    }
}

#[track_caller]
fn assert_wgsl(assertion: &str, preamble: &str) {
    let output = eval_bool(assertion, preamble);
    assert!(output);
}

#[test]
fn init_variable() {
    let a = eval::<u32>("a", "var a: u32 = 123;", "u32");
    assert_eq!(a, 123);
}

#[test]
fn store_variable() {
    let a = eval::<u32>("a", "var a: u32; a = 123;", "u32");
    assert_eq!(a, 123);
}

#[test]
fn casts() {
    #[track_caller]
    fn test_cast(value: &str, input_ty: &str, output_ty: &str, expected: &str) {
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
fn binops_scalars() {
    #[track_caller]
    fn test_binop<T>(ty: &str, left: &str, op: &str, right: &str, expected: T)
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
fn comparisions() {
    #[track_caller]
    fn test_compare(ty: &str, left: &str, cmp: &str, right: &str, expected: bool) {
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
fn unops() {
    #[track_caller]
    fn test_unop<T>(ty: &str, op: &str, input: &str, expected: T)
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
    fn test_bool_unop(op: &str, input: &str, expected: bool) {
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
fn if_stmt() {
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
fn early_return() {
    let out = eval::<u32>("123", "return Output(vec4f(), 456);", "u32");
    assert_eq!(out, 456);
}

#[test]
fn if_early_return() {
    let out = eval::<u32>(
        "123",
        "var c: bool = true; if c { return Output(vec4f(), 456); }",
        "u32",
    );
    assert_eq!(out, 456);
}
