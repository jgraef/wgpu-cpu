
# Issues

Issues that can be raised with `wgpu`:

- [x] `InstanceInterface::create_surface` returns a `wgpu::CreateSurfaceError`, but these can't be constructed other than by a conversion from `wgpu_core::CreateSurfaceError`.
- [x] custom `RequestAdapterError`
- [ ] `RenderPassDescriptor` generic over label, with `map_label`
- [ ] `RenderPassInterface::end` is never called. core implements its own Drop that calls it. But shouldn't this be called by the wrapper?
