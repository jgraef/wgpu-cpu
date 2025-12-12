
# Issues

Issues that can be raised with `wgpu`:

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
- [x] naga: make `naga::ir::Binding` `Copy`
