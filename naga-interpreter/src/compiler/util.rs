use std::{
    collections::HashMap,
    sync::Arc,
};

use cranelift_codegen::ir::immediates::Ieee16;
use half::f16;

pub fn ieee16_from_f16(x: f16) -> Ieee16 {
    Ieee16::with_bits(x.to_bits())
}

pub fn alignment_log2(alignment: naga::proc::Alignment) -> u8 {
    const ALIGNMENTS: [naga::proc::Alignment; 5] = [
        naga::proc::Alignment::ONE,
        naga::proc::Alignment::TWO,
        naga::proc::Alignment::FOUR,
        naga::proc::Alignment::EIGHT,
        naga::proc::Alignment::SIXTEEN,
    ];
    ALIGNMENTS
        .iter()
        .enumerate()
        .find(|(_i, x)| **x == alignment)
        .map(|(i, _x)| i)
        .unwrap()
        .try_into()
        .unwrap()
}

#[derive(derive_more::Debug)]
pub struct ClifOutput {
    #[debug(skip)]
    pub isa: Arc<dyn cranelift_codegen::isa::TargetIsa>,
    pub declarations: cranelift_module::ModuleDeclarations,
    pub functions: HashMap<cranelift_module::FuncId, cranelift_codegen::ir::Function>,
}

impl ClifOutput {
    pub fn new(isa: Arc<dyn cranelift_codegen::isa::TargetIsa>) -> Self {
        Self {
            isa,
            declarations: Default::default(),
            functions: Default::default(),
        }
    }

    pub fn finalize(&mut self) {
        for (func_id, function) in self.functions.iter_mut() {
            let decl = self.declarations.get_function_decl(*func_id);
            if let Some(name) = &decl.name {
                function.name = cranelift_codegen::ir::UserFuncName::testcase(name);
            }
        }
    }
}

impl cranelift_module::Module for ClifOutput {
    fn isa(&self) -> &dyn cranelift_codegen::isa::TargetIsa {
        &*self.isa
    }

    fn declarations(&self) -> &cranelift_module::ModuleDeclarations {
        &self.declarations
    }

    fn declare_function(
        &mut self,
        name: &str,
        linkage: cranelift_module::Linkage,
        signature: &cranelift_codegen::ir::Signature,
    ) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
        let (func_id, _) = self
            .declarations
            .declare_function(name, linkage, signature)?;
        Ok(func_id)
    }

    fn declare_anonymous_function(
        &mut self,
        signature: &cranelift_codegen::ir::Signature,
    ) -> cranelift_module::ModuleResult<cranelift_module::FuncId> {
        self.declarations.declare_anonymous_function(signature)
    }

    fn declare_data(
        &mut self,
        name: &str,
        linkage: cranelift_module::Linkage,
        writable: bool,
        tls: bool,
    ) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
        let (data_id, _) = self
            .declarations
            .declare_data(name, linkage, writable, tls)?;
        Ok(data_id)
    }

    fn declare_anonymous_data(
        &mut self,
        writable: bool,
        tls: bool,
    ) -> cranelift_module::ModuleResult<cranelift_module::DataId> {
        self.declarations.declare_anonymous_data(writable, tls)
    }

    fn define_function_with_control_plane(
        &mut self,
        func: cranelift_module::FuncId,
        ctx: &mut cranelift_codegen::Context,
        ctrl_plane: &mut cranelift_codegen::control::ControlPlane,
    ) -> cranelift_module::ModuleResult<()> {
        let _ = ctrl_plane;
        let function = std::mem::replace(&mut ctx.func, cranelift_codegen::ir::Function::new());
        self.functions.insert(func, function);
        Ok(())
    }

    fn define_function_bytes(
        &mut self,
        func_id: cranelift_module::FuncId,
        alignment: u64,
        bytes: &[u8],
        relocs: &[cranelift_module::ModuleReloc],
    ) -> cranelift_module::ModuleResult<()> {
        let _ = (func_id, alignment, bytes, relocs);
        Ok(())
    }

    fn define_data(
        &mut self,
        data_id: cranelift_module::DataId,
        data: &cranelift_module::DataDescription,
    ) -> cranelift_module::ModuleResult<()> {
        let _ = (data_id, data);
        Ok(())
    }
}

/*
#[derive(Debug)]
pub struct MatrixLanes {
    columns: u8,
    rows: u8,
    stride: u8,
}

impl MatrixLanes {
    pub fn new(columns: impl Into<u8>, rows: impl Into<u8>) -> Self {
        let columns = columns.into();
        let rows = rows.into();

        let stride: u8 = match columns {
            2 => 2,
            3 => 4,
            4 => 4,
            _ => unreachable!("invalid matrix size: {columns}x{rows}"),
        };

        Self {
            columns,
            rows,
            stride,
        }
    }

    pub fn lane(&self, column: u8, row: u8) -> u8 {
        assert!(column < self.columns as u8);
        assert!(row < self.rows as u8);
        row + self.stride * column
    }

    pub fn lane_flat(&self, i: u8) -> u8 {
        let column = i / self.rows;
        let row = i % self.rows;
        self.lane(column, row)
    }

    pub fn for_each(&self, mut f: impl FnMut(u8, u8, u8)) {
        self.try_for_each(|lane, row, column| {
            f(lane, row, column);
            Ok::<(), Infallible>(())
        })
        .unwrap_or_else(|e| match e {})
    }

    pub fn try_for_each<E>(&self, mut f: impl FnMut(u8, u8, u8) -> Result<(), E>) -> Result<(), E> {
        for row in 0..self.columns {
            for column in 0..self.rows {
                f(self.lane(row, column), row, column)?;
            }
        }
        Ok(())
    }

    pub fn column_offset(&self, column: u8, scalar_width: u8) -> u32 {
        u32::from(self.stride) * u32::from(column) * u32::from(scalar_width)
    }
}
 */
