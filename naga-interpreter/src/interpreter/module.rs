use std::ops::Index;

use naga::{
    Binding,
    EntryPoint,
    Function,
    FunctionArgument,
    FunctionResult,
    Handle,
    Module,
    Scalar,
    ShaderStage,
    Type,
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
    bindings::{
        BindingLocationLayout,
        IoBindingVisitor,
        UserDefinedInterStageLayout,
        VisitIoBindings,
    },
    entry_point::{
        EntryPointIndex,
        EntryPointNotFound,
        EntryPoints,
    },
    interpreter::variable::VariableType,
    util::{
        CoArena,
        SparseVec,
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
    user_defined_io_layouts: UserDefinedIoLayouts,
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

        let mut user_defined_io_layouts = SparseVec::with_capacity(module.entry_points.len());
        let mut entry_points = EntryPoints::default();
        for (i, entry_point) in module.entry_points.iter().enumerate() {
            entry_points.push(entry_point, ());

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

        Ok(Self {
            module,
            module_info,
            layouter,
            entry_points,
            expression_types,
            user_defined_io_layouts: UserDefinedIoLayouts {
                inner: user_defined_io_layouts,
            },
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
            .map(|(index, ())| (index, &self.module.entry_points[index.index]))
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

#[derive(Clone, Debug)]
pub struct UserDefinedIoLayouts {
    pub(super) inner: SparseVec<UserDefinedIoLayout>,
}

impl Index<EntryPointIndex> for UserDefinedIoLayouts {
    type Output = UserDefinedIoLayout;

    fn index(&self, index: EntryPointIndex) -> &Self::Output {
        &self.inner[index.index]
    }
}

#[derive(Clone, Debug)]
pub struct CollectUserDefinedInterStageLayout<'module> {
    pub layouter: &'module Layouter,
    pub buffer_offset: u32,
    pub locations: SparseVec<BindingLocationLayout>,
}

impl<'module> VisitIoBindings for CollectUserDefinedInterStageLayout<'module> {
    fn visit(
        &mut self,
        binding: &Binding,
        ty_handle: Handle<Type>,
        ty: &Type,
        offset: u32,
        name: Option<&str>,
    ) {
        // this is the offset in the struct that contains this inter-stage location
        // binding. we don't care about this, since we can layout our
        // inter-stage buffer as we want. in particular the layout of the vertex
        // output and fragment input might not even match.
        let _ = offset;

        match binding {
            Binding::BuiltIn(_builtin) => {
                // nop
            }
            Binding::Location {
                location,
                interpolation,
                sampling,
                blend_src,
                per_primitive,
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

pub fn collect_user_defined_inter_stage_layout_from_function_arguments<'a>(
    module: &Module,
    layouter: &Layouter,
    arguments: impl IntoIterator<Item = &'a FunctionArgument>,
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
    module: &Module,
    layouter: &Layouter,
    result: impl Into<Option<&'a FunctionResult>>,
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
pub enum UserDefinedIoLayout {
    Vertex {
        // todo: input
        output: UserDefinedInterStageLayout,
    },
    Fragment {
        input: UserDefinedInterStageLayout,
        // output: don't need it, but would be handy for verification
    },
}
