use std::{
    collections::HashMap,
    ops::Index,
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

use crate::{
    bindings::{
        UserDefinedIoLayout,
        collect_user_defined_inter_stage_layout_from_function_arguments,
        collect_user_defined_inter_stage_layout_from_function_result,
    },
    interpreter::VariableType,
    util::{
        CoArena,
        SparseVec,
    },
};

#[derive(Debug)]
pub struct ShaderModule {
    pub(crate) module: Module,
    #[allow(unused)]
    module_info: ModuleInfo,
    layouter: Layouter,
    entry_points_by_name: HashMap<String, usize>,
    unique_entry_points_by_stage: HashMap<ShaderStage, usize>,
    pub(crate) expression_types: ExpressionTypes,
    user_defined_io_layouts: UserDefinedIoLayouts,
}

impl ShaderModule {
    pub fn new(module: naga::Module) -> Self {
        let mut validator = Validator::new(Default::default(), Default::default());
        let module_info = validator.validate(&module).unwrap();

        let mut layouter = Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        //tracing::trace!("module: {module:#?}");
        //tracing::trace!("module_info: {module_info:#?}");
        //tracing::trace!("layouter: {layouter:#?}");
        tracing::debug!("types: {:#?}", module.types);

        let mut expression_types = ExpressionTypes {
            entry_points: Vec::with_capacity(module.entry_points.len()),
            per_function: CoArena::from_arena(&module.functions, |handle, function| {
                typifier_from_function(&module, function)
            }),
        };

        let mut user_defined_io_layouts = SparseVec::with_capacity(module.entry_points.len());
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

            expression_types
                .entry_points
                .push(typifier_from_function(&module, &entry_point.function));

            match entry_point.stage {
                ShaderStage::Vertex => {
                    user_defined_io_layouts.insert(
                        i,
                        UserDefinedIoLayout::Vertex {
                            output: collect_user_defined_inter_stage_layout_from_function_result(
                                &module,
                                &layouter,
                                &entry_point.function.result,
                            ),
                        },
                    )
                }
                ShaderStage::Task => todo!(),
                ShaderStage::Mesh => todo!(),
                ShaderStage::Fragment => {
                    user_defined_io_layouts.insert(
                        i,
                        UserDefinedIoLayout::Fragment {
                            input: collect_user_defined_inter_stage_layout_from_function_arguments(
                                &module,
                                &layouter,
                                &entry_point.function.arguments,
                            ),
                        },
                    );
                }
                ShaderStage::Compute => {
                    // nop: these can't have user-defined io bindings
                }
            }
        }
        unique_entry_points_by_stage.retain(|_, index| *index != usize::MAX);

        Self {
            module,
            module_info,
            layouter,
            entry_points_by_name,
            unique_entry_points_by_stage,
            expression_types,
            user_defined_io_layouts: UserDefinedIoLayouts {
                inner: user_defined_io_layouts,
            },
        }
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

    pub fn user_defined_io_layout(&self, entry_point: EntryPointIndex) -> &UserDefinedIoLayout {
        &self.user_defined_io_layouts[entry_point]
    }
}

#[derive(Debug, thiserror::Error)]
pub enum EntryPointNotFound {
    #[error("Entry point '{name}' not found")]
    NameNotFound { name: String },
    #[error("No unique entry point for shader stage {stage:?} found")]
    NoUniqueForStage { stage: ShaderStage },
}

impl Index<EntryPointIndex> for ShaderModule {
    type Output = EntryPoint;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.module.entry_points[index.0]
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
        &self.entry_points[index.0]
    }
}

impl Index<Handle<Function>> for ExpressionTypes {
    type Output = Typifier;

    fn index(&self, index: Handle<Function>) -> &Self::Output {
        &self.per_function[index]
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

#[derive(Clone, Debug)]
struct UserDefinedIoLayouts {
    inner: SparseVec<UserDefinedIoLayout>,
}

impl Index<EntryPointIndex> for UserDefinedIoLayouts {
    type Output = UserDefinedIoLayout;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.inner[index.0]
    }
}
