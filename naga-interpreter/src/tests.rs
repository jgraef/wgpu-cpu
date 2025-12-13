use naga::{
    Scalar,
    ScalarKind,
    TypeInner,
    proc::{
        Alignment,
        Layouter,
    },
};

#[test]
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
