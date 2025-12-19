use std::{
    collections::HashMap,
    ops::Index,
};

use crate::{
    bindings::{
        BindingLocationLayout,
        IoBindingVisitor,
        UserDefinedInterStageLayout,
        VisitIoBindings,
    },
    util::SparseVec,
};

#[derive(Clone, Copy, Debug)]
pub struct EntryPointIndex {
    pub(crate) index: usize,
}

impl From<EntryPointIndex> for usize {
    fn from(value: EntryPointIndex) -> Self {
        value.index
    }
}

#[cfg(test)]
impl From<usize> for EntryPointIndex {
    fn from(value: usize) -> Self {
        Self { index: value }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EntryPointNotFound {
    #[error("Entry point '{name}' not found")]
    NameNotFound { name: String },
    #[error("No entry point for shader stage {stage:?} found")]
    NotFound { stage: naga::ShaderStage },
    #[error("There are multiple entry points for this shader stage: {stage:?}")]
    NotUnique { stage: naga::ShaderStage },
    #[error(
        "Found entry point '{name}', but it is for shader stage {module_stage:?}, and not {expected_stage:?}"
    )]
    WrongStage {
        name: String,
        module_stage: naga::ShaderStage,
        expected_stage: naga::ShaderStage,
    },
}

#[derive(Debug, Default)]
pub struct EntryPoints<T> {
    items: Vec<EntryPoint<T>>,
    by_name: HashMap<String, usize>,
    by_stage: HashMap<naga::ShaderStage, usize>,
}

impl<T> EntryPoints<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            by_name: HashMap::with_capacity(capacity),
            by_stage: HashMap::with_capacity(capacity),
        }
    }

    pub fn push<'a>(
        &'a mut self,
        entry_point: &'a naga::EntryPoint,
        data: T,
    ) -> EntryPointBuilder<'a, T> {
        let index = self.items.len();

        self.items.push(EntryPoint {
            name: entry_point.function.name.as_ref().map(ToOwned::to_owned),
            data,
            stage: entry_point.stage,
            inter_stage_layout: None,
        });

        self.by_name.insert(entry_point.name.clone(), index);

        if let Some(not_unique) = self.by_stage.get_mut(&entry_point.stage) {
            *not_unique = usize::MAX;
        }
        else {
            self.by_stage.insert(entry_point.stage, index);
        }

        EntryPointBuilder {
            item: &mut self.items[index],
            source: entry_point,
        }
    }

    pub fn find(
        &self,
        name: Option<&str>,
        stage: naga::ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        let index = if let Some(name) = name {
            let index = *self.by_name.get(name).ok_or_else(|| {
                EntryPointNotFound::NameNotFound {
                    name: name.to_owned(),
                }
            })?;

            let item = &self.items[index];
            if item.stage != stage {
                return Err(EntryPointNotFound::WrongStage {
                    name: name.to_owned(),
                    module_stage: item.stage,
                    expected_stage: stage,
                });
            }

            index
        }
        else {
            *self
                .by_stage
                .get(&stage)
                .ok_or_else(|| EntryPointNotFound::NotFound { stage })?
        };

        Ok(EntryPointIndex { index })
    }

    pub fn iter(&self) -> impl Iterator<Item = (EntryPointIndex, &EntryPoint<T>)> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| (EntryPointIndex { index }, item))
    }
}

impl<T> Index<EntryPointIndex> for EntryPoints<T> {
    type Output = EntryPoint<T>;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.items[index.index]
    }
}

#[derive(Debug)]
pub struct EntryPoint<T> {
    pub name: Option<String>,
    pub stage: naga::ShaderStage,
    pub data: T,
    pub inter_stage_layout: Option<InterStageLayout>,
}

pub struct EntryPointBuilder<'a, T> {
    item: &'a mut EntryPoint<T>,
    source: &'a naga::EntryPoint,
}

impl<'a, T> EntryPointBuilder<'a, T> {
    pub fn collect_inter_stage_layouts(
        &mut self,
        module: &naga::Module,
        layouter: &naga::proc::Layouter,
    ) {
        self.item.inter_stage_layout = match self.item.stage {
            naga::ShaderStage::Vertex => {
                Some(InterStageLayout::Vertex {
                    output: collect_user_defined_inter_stage_layout_from_function_result(
                        &module,
                        &layouter,
                        &self.source.function.result,
                    ),
                })
            }
            naga::ShaderStage::Fragment => {
                Some(InterStageLayout::Fragment {
                    input: collect_user_defined_inter_stage_layout_from_function_arguments(
                        &module,
                        &layouter,
                        &self.source.function.arguments,
                    ),
                })
            }
            _ => None,
        };
    }
}

#[derive(Clone, Debug)]
pub enum InterStageLayout {
    Vertex { output: UserDefinedInterStageLayout },
    Fragment { input: UserDefinedInterStageLayout },
}

pub fn collect_user_defined_inter_stage_layout_from_function_arguments<'a>(
    module: &naga::Module,
    layouter: &naga::proc::Layouter,
    arguments: impl IntoIterator<Item = &'a naga::FunctionArgument>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::default(),
    };

    for argument in arguments {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_argument(argument, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

pub fn collect_user_defined_inter_stage_layout_from_function_result<'a>(
    module: &naga::Module,
    layouter: &naga::proc::Layouter,
    result: impl Into<Option<&'a naga::FunctionResult>>,
) -> UserDefinedInterStageLayout {
    let mut visit = CollectUserDefinedInterStageLayout {
        layouter,
        buffer_offset: 0,
        locations: SparseVec::new(),
    };

    if let Some(result) = result.into() {
        IoBindingVisitor {
            types: &module.types,
            visit: &mut visit,
        }
        .visit_function_result(result, 0);
    }

    UserDefinedInterStageLayout {
        locations: visit.locations.into_vec().into(),
        size: visit.buffer_offset,
    }
}

#[derive(Clone, Debug)]
pub struct CollectUserDefinedInterStageLayout<'module> {
    pub layouter: &'module naga::proc::Layouter,
    pub buffer_offset: u32,
    pub locations: SparseVec<BindingLocationLayout>,
}

impl<'module> VisitIoBindings for CollectUserDefinedInterStageLayout<'module> {
    fn visit(
        &mut self,
        binding: &naga::Binding,
        ty_handle: naga::Handle<naga::Type>,
        ty: &naga::Type,
        offset: u32,
        name: Option<&str>,
        top_level: bool,
    ) {
        // this is the offset in the struct that contains this inter-stage location
        // binding. we don't care about this, since we can layout our
        // inter-stage buffer as we want. in particular the layout of the vertex
        // output and fragment input might not even match.
        let _ = offset;
        let _ = (ty, name, top_level);

        match binding {
            naga::Binding::BuiltIn(_builtin) => {
                // nop
            }
            naga::Binding::Location {
                location,
                interpolation: _,
                sampling: _,
                blend_src: _,
                per_primitive: _,
            } => {
                let type_layout = self.layouter[ty_handle];
                let offset = type_layout.alignment.round_up(self.buffer_offset);
                let size = type_layout.size;
                self.buffer_offset = offset + size;

                let index = *location as usize;
                self.locations
                    .insert(index, BindingLocationLayout { offset, size });
            }
        }
    }
}
