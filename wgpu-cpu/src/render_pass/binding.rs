use std::collections::HashMap;

use naga_cranelift::bindings::BindingResources;

use crate::{
    bind_group::{
        BindGroup,
        BindingResource,
    },
    buffer::BufferReadGuard,
};

#[derive(Debug)]
pub struct AcquiredBindingResources<'state> {
    bindings: HashMap<naga::ResourceBinding, AcquiredBindingResource<'state>>,
}

impl<'state> AcquiredBindingResources<'state> {
    pub fn new(bind_groups: &'state [Option<BindGroup>]) -> Self {
        let mut bindings = HashMap::new();

        for (group, bind_group) in bind_groups.into_iter().enumerate() {
            if let Some(bind_group) = bind_group {
                for entry in bind_group.entries.iter() {
                    let resource = match &entry.resource {
                        BindingResource::Buffer(buffer_slice) => {
                            AcquiredBindingResource::Buffer(buffer_slice.read())
                        }
                    };

                    bindings.insert(
                        naga::ResourceBinding {
                            group: group.try_into().unwrap(),
                            binding: entry.binding,
                        },
                        resource,
                    );
                }
            }
        }

        Self { bindings }
    }
}

// todo: how to make this writable and sharable?
impl<'state> BindingResources for &AcquiredBindingResources<'state> {
    fn read(&self, binding: naga::ResourceBinding) -> &[u8] {
        let binding = self
            .bindings
            .get(&binding)
            .expect("No such binding resource: {binding:?}");
        match binding {
            AcquiredBindingResource::Buffer(buffer_read_guard) => &*buffer_read_guard,
        }
    }
}

#[derive(Debug)]
pub enum AcquiredBindingResource<'state> {
    Buffer(BufferReadGuard<'state>),
}
