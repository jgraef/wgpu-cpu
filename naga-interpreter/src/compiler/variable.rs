use std::collections::HashMap;

use cranelift_codegen::ir;

use crate::compiler::types::PointerType;

#[derive(Clone, Copy, Debug)]
pub struct GlobalVariable {
    pub address_space: naga::AddressSpace,
    pub pointer_type: PointerType,
    pub inner: GlobalVariableInner,
}

#[derive(Clone, Copy, Debug)]
pub enum GlobalVariableInner {
    Memory { offset: u32, len: u32 },
    Resource { binding: naga::ResourceBinding },
}

#[derive(Clone, Copy, Debug)]
pub struct GlobalVariableLayoutEntry {
    pub offset: u32,
    pub len: u32,
    pub address_space: naga::AddressSpace,
    pub initialized: bool,
}

#[derive(Clone, Debug, Default)]
pub struct GlobalVariableLayout {
    pub entries: HashMap<naga::Handle<naga::GlobalVariable>, GlobalVariableLayoutEntry>,
    pub private_memory_layout: PrivateMemoryLayout,
}

impl GlobalVariableLayout {
    pub fn get(
        &mut self,
        handle: naga::Handle<naga::GlobalVariable>,
    ) -> Option<GlobalVariableLayoutEntry> {
        self.entries.get(&handle).copied()
    }

    pub fn iter(
        &self,
    ) -> impl Iterator<
        Item = (
            naga::Handle<naga::GlobalVariable>,
            GlobalVariableLayoutEntry,
        ),
    > {
        self.entries.iter().map(|(k, v)| (*k, *v))
    }
}

#[derive(Clone, Debug)]
pub struct GlobalVariablesLayouter<'layouter> {
    layouter: &'layouter naga::proc::Layouter,
    layout: HashMap<naga::Handle<naga::GlobalVariable>, GlobalVariableLayoutEntry>,

    private_alignment: u32,
    private_initialized: Vec<u8>,
    private_zeroed: u32,
    //work_group_initialized: Vec<u8>,
    //work_group_zeroed: u32,
}

impl<'layouter> GlobalVariablesLayouter<'layouter> {
    pub fn new(layouter: &'layouter naga::proc::Layouter) -> Self {
        Self {
            layouter,
            layout: Default::default(),
            private_alignment: 0,
            private_initialized: vec![],
            private_zeroed: 0,
        }
    }

    pub fn push(
        &mut self,
        handle: naga::Handle<naga::GlobalVariable>,
        global_variable: &naga::GlobalVariable,
        initialize: impl FnOnce(naga::Handle<naga::Expression>, &mut [u8]),
    ) {
        let type_layout = self.layouter[global_variable.ty];

        match global_variable.space {
            naga::AddressSpace::Private => {
                if self.layout.is_empty() {
                    self.private_alignment = type_layout.alignment.round_up(1);
                }

                if let Some(init) = global_variable.init {
                    let mut offset: u32 = self.private_initialized.len().try_into().unwrap();
                    offset = type_layout.alignment.round_up(offset);
                    self.private_initialized
                        .resize(usize::try_from(offset + type_layout.size).unwrap(), 0);

                    initialize(
                        init,
                        &mut self.private_initialized[usize::try_from(offset).unwrap()..],
                    );

                    self.layout.insert(
                        handle,
                        GlobalVariableLayoutEntry {
                            offset,
                            len: type_layout.size,
                            address_space: global_variable.space,
                            initialized: true,
                        },
                    );
                }
                else {
                    let offset = type_layout.alignment.round_up(self.private_zeroed);
                    self.private_zeroed = offset + type_layout.size;

                    self.layout.insert(
                        handle,
                        GlobalVariableLayoutEntry {
                            offset,
                            len: type_layout.size,
                            address_space: global_variable.space,
                            initialized: false,
                        },
                    );
                }
            }
            naga::AddressSpace::WorkGroup => todo!(),
            _ => {
                panic!(
                    "Invalid to have global variable with address space {:?} without binding",
                    global_variable.space
                )
            }
        }
    }

    pub fn finish(mut self) -> GlobalVariableLayout {
        let private_initialized_size: u32 = self.private_initialized.len().try_into().unwrap();

        for (_handle, layout) in self.layout.iter_mut() {
            if !layout.initialized {
                match layout.address_space {
                    naga::AddressSpace::Private => {
                        layout.offset += private_initialized_size;
                    }
                    naga::AddressSpace::WorkGroup => todo!(),
                    _ => unreachable!("invalid: {:?}", layout.address_space),
                }
            }
        }

        let private_memory_layout = PrivateMemoryLayout {
            alignment: self.private_alignment,
            initialized: self.private_initialized,
            zeroed: self.private_zeroed,
        };

        GlobalVariableLayout {
            entries: self.layout,
            private_memory_layout,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PrivateMemoryLayout {
    pub alignment: u32,
    pub initialized: Vec<u8>,
    pub zeroed: u32,
}

impl PrivateMemoryLayout {
    pub fn size(&self) -> u32 {
        let len_initialized: u32 = self.initialized.len().try_into().unwrap();
        len_initialized + self.zeroed
    }

    pub fn stack_slot_data(&self) -> ir::StackSlotData {
        assert!(
            self.alignment.is_power_of_two(),
            "alignment is not a power of 2: {}",
            self.alignment
        );
        let align_shift = self
            .alignment
            .ilog2()
            .try_into()
            .expect("align_shift overflow");
        ir::StackSlotData {
            kind: ir::StackSlotKind::ExplicitSlot,
            size: self.size(),
            align_shift,
            key: None,
        }
    }
}
