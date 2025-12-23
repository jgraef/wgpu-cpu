use std::ops::{
    Index,
    IndexMut,
};

#[derive(Clone, Copy, Debug)]
pub enum GlobalVariable {
    Memory {
        address_space: naga::AddressSpace,
        offset: u32,
    },
    Resource {
        address_space: naga::AddressSpace,
        binding: naga::ResourceBinding,
    },
}

#[derive(Clone, Copy, Debug, Default)]
pub struct GlobalVariablesLayouterCursors {
    pub private: u32,
    pub work_group: u32,
}

impl Index<naga::AddressSpace> for GlobalVariablesLayouterCursors {
    type Output = u32;

    fn index(&self, index: naga::AddressSpace) -> &Self::Output {
        match index {
            naga::AddressSpace::Private => &self.private,
            naga::AddressSpace::WorkGroup => &self.work_group,
            _ => panic!("not supported in global variables layout: {index:?}"),
        }
    }
}

impl IndexMut<naga::AddressSpace> for GlobalVariablesLayouterCursors {
    fn index_mut(&mut self, index: naga::AddressSpace) -> &mut Self::Output {
        match index {
            naga::AddressSpace::Private => &mut self.private,
            naga::AddressSpace::WorkGroup => &mut self.work_group,
            _ => panic!("not supported in global variables layout: {index:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalVariablesLayouter<'layouter> {
    pub layouter: &'layouter naga::proc::Layouter,
    pub cursors: GlobalVariablesLayouterCursors,
}

impl<'layouter> GlobalVariablesLayouter<'layouter> {
    pub fn new(layouter: &'layouter naga::proc::Layouter) -> Self {
        Self {
            layouter,
            cursors: Default::default(),
        }
    }

    pub fn push(&mut self, global_variable: &naga::GlobalVariable) -> GlobalVariable {
        if let Some(binding) = global_variable.binding {
            assert!(global_variable.init.is_none());
            GlobalVariable::Resource {
                address_space: global_variable.space,
                binding,
            }
        }
        else {
            let type_layout = self.layouter[global_variable.ty];

            let cursor = &mut self.cursors[global_variable.space];
            *cursor = type_layout.alignment.round_up(*cursor);
            let offset = *cursor;
            *cursor += type_layout.size;

            GlobalVariable::Memory {
                offset,
                address_space: global_variable.space,
            }
        }
    }
}
