pub mod bindings;
pub mod interpreter;
pub mod memory;

use std::{
    collections::HashMap,
    ops::{
        Deref,
        Index,
    },
    sync::Arc,
};

use naga::{
    EntryPoint,
    Function,
    Handle,
    Module,
    Scalar,
    ShaderStage,
    TypeInner,
    front::Typifier,
    proc::{
        Alignment,
        Layouter,
        ResolveContext,
        TypeLayout,
    },
    valid::{
        ModuleInfo,
        Validator,
    },
};
use wgpu::custom::ShaderModuleInterface;

use crate::shader::interpreter::VariableType;

#[derive(Clone, Debug)]
pub struct ShaderModule {
    inner: Arc<ShaderModuleInner>,
}

impl ShaderModule {
    pub fn new(
        shader_source: wgpu::ShaderSource,
        shader_bound_checks: wgpu::ShaderRuntimeChecks,
    ) -> Result<Self, Error> {
        let module = match shader_source {
            wgpu::ShaderSource::Wgsl(wgsl) => naga::front::wgsl::parse_str(&wgsl)?,
            _ => return Err(Error::Unsupported),
        };

        let mut validator = Validator::new(Default::default(), Default::default());
        let module_info = validator.validate(&module).unwrap();

        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        tracing::trace!("module: {:#?}", module);
        tracing::trace!("module_info: {:#?}", module_info);
        tracing::trace!("layouter: {:#?}", layouter);

        let mut entry_points_by_name = HashMap::with_capacity(module.entry_points.len());
        let mut unique_entry_points_by_stage = HashMap::with_capacity(module.entry_points.len());
        for (i, entry_point) in module.entry_points.iter().enumerate() {
            entry_points_by_name.insert(entry_point.name.clone(), i);

            if let Some(not_unique) = unique_entry_points_by_stage.get_mut(&entry_point.stage) {
                *not_unique = usize::MAX;
            }
            else {
                unique_entry_points_by_stage.insert(entry_point.stage, i);
            }
        }
        unique_entry_points_by_stage.retain(|_, index| *index != usize::MAX);

        let expression_types = ExpressionTypes::new(&module);

        Ok(Self {
            inner: Arc::new(ShaderModuleInner {
                module,
                module_info,
                layouter,
                compilation_info: wgpu::CompilationInfo { messages: vec![] },
                entry_points_by_name,
                unique_entry_points_by_stage,
                expression_types,
            }),
        })
    }
}

impl ShaderModuleInterface for ShaderModule {
    fn get_compilation_info(
        &self,
    ) -> std::pin::Pin<Box<dyn wgpu::custom::ShaderCompilationInfoFuture>> {
        let compilation_info = self.inner.compilation_info.clone();
        Box::pin(async move { compilation_info })
    }
}

#[derive(Debug)]
pub struct ShaderModuleInner {
    pub module: Module,
    pub module_info: ModuleInfo,
    pub layouter: Layouter,
    pub compilation_info: wgpu::CompilationInfo,
    pub entry_points_by_name: HashMap<String, usize>,
    pub unique_entry_points_by_stage: HashMap<ShaderStage, usize>,
    pub expression_types: ExpressionTypes,
}

impl ShaderModuleInner {
    pub fn type_layout<'t>(&self, ty: impl Into<VariableType<'t>>) -> TypeLayout {
        // https://gpuweb.github.io/gpuweb/wgsl/#memory-layouts

        let ty = ty.into();
        match ty {
            VariableType::Handle(handle) => self.layouter[handle],
            VariableType::Inner(type_inner) => {
                match type_inner {
                    TypeInner::Scalar(Scalar { kind, width }) => {
                        TypeLayout {
                            size: *width as u32,
                            alignment: Alignment::from_width(*width),
                        }
                    }
                    _ => todo!("layout for {ty:?}"),
                }
            }
        }
    }

    pub fn entry_point(
        &self,
        name: Option<&str>,
        stage: ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        let index = if let Some(name) = name {
            self.entry_points_by_name.get(name).ok_or_else(|| {
                EntryPointNotFound::NameNotFound {
                    name: name.to_owned(),
                }
            })?
        }
        else {
            self.unique_entry_points_by_stage
                .get(&stage)
                .ok_or_else(|| EntryPointNotFound::NoUniqueForStage { stage })?
        };
        Ok(EntryPointIndex(*index))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EntryPointNotFound {
    #[error("Entry point '{name}' not found")]
    NameNotFound { name: String },
    #[error("No unique entry point for shader stage {stage:?} found")]
    NoUniqueForStage { stage: ShaderStage },
}

impl Deref for ShaderModule {
    type Target = ShaderModuleInner;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl Index<EntryPointIndex> for ShaderModuleInner {
    type Output = EntryPoint;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.module.entry_points[index.0]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Provided shader source variant is not supported")]
    Unsupported,

    #[error(transparent)]
    ParseError(#[from] naga::front::wgsl::ParseError),
}

#[derive(Debug)]
pub struct ExpressionTypes {
    entry_points: Vec<Typifier>,
    per_function: Vec<Typifier>,
}

impl ExpressionTypes {
    pub fn new(module: &Module) -> Self {
        let entry_points = module
            .entry_points
            .iter()
            .map(|entry_point| typifier_from_function(module, &entry_point.function))
            .collect();

        let mut per_function = Vec::with_capacity(module.functions.len());

        for (handle, function) in module.functions.iter() {
            let i = handle.index();
            assert_eq!(per_function.len(), i);
            per_function.push(typifier_from_function(&module, function))
        }

        Self {
            entry_points,
            per_function,
        }
    }
}

impl Index<EntryPointIndex> for ExpressionTypes {
    type Output = Typifier;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.entry_points[index.0]
    }
}

impl Index<Handle<Function>> for ExpressionTypes {
    type Output = Typifier;

    fn index(&self, index: Handle<Function>) -> &Self::Output {
        &self.per_function[index.index()]
    }
}

fn typifier_from_function(module: &Module, function: &Function) -> Typifier {
    let mut typifier = Typifier::default();
    let resolve_context =
        ResolveContext::with_locals(module, &function.local_variables, &function.arguments);

    for (handle, expression) in function.expressions.iter() {
        typifier
            .grow(handle, &function.expressions, &resolve_context)
            .unwrap();
    }

    typifier
}

#[derive(Clone, Copy, Debug)]
pub struct EntryPointIndex(pub usize);
