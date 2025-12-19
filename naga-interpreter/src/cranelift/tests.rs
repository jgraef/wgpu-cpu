use crate::{
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
}
