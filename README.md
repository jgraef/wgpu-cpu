
# TODO

- [ ] clipper:
  - [x] line
  - [ ] tri ([article](https://www.gabrielgambetta.com/computer-graphics-from-scratch/11-clipping.html))
- [ ] srgb conversion (`palette`!)
- [ ] compile naga IR with [cranelift](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/docs/index.md) ([JIT demo](https://github.com/bytecodealliance/cranelift-jit-demo/))
- [ ] remove `VariableType`:
  - disregard, will try cranelift first.
  - `Variable` can store a `&'module TypeInner` directly and an optional `ty_name: Option<&'module str>`
  - we might need a trait for things that can fetch `&'module TypeInner` given the `&'module ShaderModule`.
- [ ] do we want to change some fields from `Vec<Option<T>>` to `[Option<T>; N]` where there's a limit? (e.g. color attachments).

# Issues

Issues that can be raised with `wgpu` (marked items are patched in a local copy of wgpu):

- [x] `InstanceInterface::create_surface` returns a `wgpu::CreateSurfaceError`, but these can't be constructed other than by a conversion from `wgpu_core::CreateSurfaceError`.
- [x] custom `RequestAdapterError`
- [ ] `RenderPassDescriptor` generic over label, with `map_label`
- [ ] `RenderPassInterface::end` is never called. core implements its own Drop that calls it. But shouldn't this be called by the wrapper?
- [ ] `TextureViewDescriptor::map_label`
- [ ] `TextureViewInterface::create_view` has no knowledge of the texture's descriptor. It would need to store it, but the wrapping type does that already.
- [ ] `InstanceInterface::wgsl_language_features` is behind a feature flag. This means when I implement the interface without the flag, but another crate in the dependency graph enables it, it breaks compilation of my crate.
- [ ] `InstanceInterface::new` doesn't serve a purpose?
- [ ] naga really needs something that returns `TypeLayout` for a `TypeInner`. type resolution sometimes can't give you a handle, so `Layouter` won't always do.
- [ ] naga: can matrices and vectors really only be multiplied by floats, or by their scalar type? (http://localhost:8001/wgpu/naga/ir/enum.BinaryOperator.html#arithmetic-type-rules)
- [ ] naga: make `naga::ir::Binding` `Copy`. This might be useful, but we don't need it anymore.
- [ ] naga: bools have size and alignment 1. See `naga_interpreter::tests::naga_bool_width_is_32bit`.
- [x] naga: way to convert `Alignment` to `NonZeroU32` (or anything usable). cranelift wants to know this, just using `round_up` won't always do. Alternative: just offer a getter for the log2 of the alignment
- [ ] `map_label` for various descriptor structs (e.g. `SamplerDescriptor`)
