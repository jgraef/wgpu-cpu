use std::ops::Index;

use naga::{
    EntryPoint,
    Function,
    Handle,
    Module,
    Scalar,
    ShaderStage,
    TypeInner,
    WithSpan,
    front::Typifier,
    proc::{
        Alignment,
        LayoutError,
        Layouter,
        TypeLayout,
    },
    valid::{
        ModuleInfo,
        ValidationError,
        Validator,
    },
};

use crate::{
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        EntryPoints,
        InterStageLayout,
    },
    interpreter::variable::VariableType,
    util::{
        CoArena,
        typifier_from_function,
    },
};

#[derive(Debug)]
pub struct ShaderModule {
    pub(crate) module: Module,
    #[allow(unused)]
    module_info: ModuleInfo,
    pub(crate) layouter: Layouter,
    entry_points: EntryPoints<()>,
    pub(crate) expression_types: ExpressionTypes,
}

impl ShaderModule {
    pub fn new(module: naga::Module) -> Result<Self, Error> {
        // todo: this should only contain minimal information (anything shared and
        // usefule before pipeline constants are known) the backend
        // (interpreter/compiler) should derive all the info they need themselves (can
        // move code for this to util)
        //
        // https://docs.rs/naga/latest/naga/back/pipeline_constants/fn.process_overrides.html

        let mut validator = Validator::new(Default::default(), Default::default());
        let module_info = validator.validate(&module)?;

        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx())?;

        //tracing::trace!("module: {module:#?}");
        //tracing::trace!("module_info: {module_info:#?}");
        //tracing::trace!("layouter: {layouter:#?}");
        //tracing::debug!("types: {:#?}", module.types);
        for (handle, ty) in module.types.iter() {
            let ty_layout = &layouter[handle];
            tracing::debug!(?ty, ?ty_layout, "type layout")
        }

        let mut expression_types = ExpressionTypes {
            entry_points: Vec::with_capacity(module.entry_points.len()),
            per_function: CoArena::from_arena(&module.functions, |handle, function| {
                typifier_from_function(&module, function)
            }),
        };

        let mut entry_points = EntryPoints::default();
        for (i, entry_point) in module.entry_points.iter().enumerate() {
            entry_points
                .push(entry_point, ())
                .collect_inter_stage_layouts(&module, &layouter);

            expression_types
                .entry_points
                .push(typifier_from_function(&module, &entry_point.function));
        }

        Ok(Self {
            module,
            module_info,
            layouter,
            entry_points,
            expression_types,
        })
    }

    pub fn type_layout<'t>(&self, ty: impl Into<VariableType<'t>>) -> TypeLayout {
        // https://gpuweb.github.io/gpuweb/wgsl/#memory-layouts

        let ty = ty.into();
        match ty {
            VariableType::Handle(handle) => self.layouter[handle],
            VariableType::Inner(type_inner) => {
                // todo: type_inner has a size method
                match type_inner {
                    TypeInner::Scalar(Scalar { kind, width }) => {
                        TypeLayout {
                            size: *width as u32,
                            alignment: Alignment::from_width(*width),
                        }
                    }
                    TypeInner::Pointer { base: _, space: _ } => {
                        TypeLayout {
                            size: 4,
                            alignment: Alignment::FOUR,
                        }
                    }
                    _ => todo!("layout for {ty:?}"),
                }
            }
        }
    }

    pub fn find_entry_point(
        &self,
        name: Option<&str>,
        stage: ShaderStage,
    ) -> Result<EntryPointIndex, EntryPointNotFound> {
        self.entry_points.find(name, stage)
    }

    pub fn entry_points(&self) -> impl Iterator<Item = (EntryPointIndex, &EntryPoint)> {
        self.entry_points
            .iter()
            .map(|(index, _)| (index, &self.module.entry_points[index.index]))
    }

    pub fn offset_of<'ty>(
        &self,
        outer_ty: impl Into<VariableType<'ty>>,
        inner_ty: impl Into<VariableType<'ty>>,
        index: usize,
    ) -> u32 {
        let outer_ty = outer_ty.into().inner_with(self);
        match outer_ty {
            TypeInner::Vector { size, scalar } => {
                let inner_ty_layout = self.type_layout(inner_ty);
                let inner_stride = inner_ty_layout.to_stride();
                inner_stride * index as u32
            }
            TypeInner::Matrix {
                columns,
                rows,
                scalar,
            } => todo!(),
            TypeInner::Array { base, size, stride } => {
                let inner_ty_layout = self.type_layout(*base);
                let inner_stride = inner_ty_layout.to_stride();
                inner_stride * index as u32
            }
            TypeInner::Struct { members, span } => members[index].offset,
            _ => panic!("Can't produce offset into {outer_ty:?}"),
        }
    }

    pub fn size_of<'ty>(&self, ty: impl Into<VariableType<'ty>>) -> u32 {
        let inner = ty.into().inner_with(self);
        inner.size(self.module.to_ctx())
    }

    pub fn inter_stage_layout(&self, entry_point: EntryPointIndex) -> Option<&InterStageLayout> {
        self.entry_points[entry_point].inter_stage_layout.as_ref()
    }
}

impl AsRef<ShaderModule> for ShaderModule {
    fn as_ref(&self) -> &ShaderModule {
        self
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Validation(#[from] WithSpan<ValidationError>),
    #[error(transparent)]
    Layout(#[from] LayoutError),
}

impl Index<EntryPointIndex> for ShaderModule {
    type Output = EntryPoint;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.module.entry_points[index.index]
    }
}

#[derive(Debug)]
pub struct ExpressionTypes {
    entry_points: Vec<Typifier>,
    per_function: CoArena<Function, Typifier>,
}

impl Index<EntryPointIndex> for ExpressionTypes {
    type Output = Typifier;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.entry_points[index.index]
    }
}

impl Index<Handle<Function>> for ExpressionTypes {
    type Output = Typifier;

    fn index(&self, index: Handle<Function>) -> &Self::Output {
        &self.per_function[index]
    }
}
