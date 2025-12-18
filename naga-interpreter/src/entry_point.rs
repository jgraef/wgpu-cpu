use std::collections::HashMap;

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
pub(crate) struct EntryPoints<T> {
    items: Vec<Item<T>>,
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
    pub fn push(&mut self, entry_point: &naga::EntryPoint, value: T) {
        let index = self.items.len();
        self.items.push(Item {
            value,
            stage: entry_point.stage,
        });

        self.by_name.insert(entry_point.name.clone(), index);

        if let Some(not_unique) = self.by_stage.get_mut(&entry_point.stage) {
            *not_unique = usize::MAX;
        }
        else {
            self.by_stage.insert(entry_point.stage, index);
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

    pub fn iter(&self) -> impl Iterator<Item = (EntryPointIndex, &T)> {
        self.items
            .iter()
            .enumerate()
            .map(|(index, item)| (EntryPointIndex { index }, &item.value))
    }
}

#[derive(Debug)]
struct Item<T> {
    stage: naga::ShaderStage,
    value: T,
}
