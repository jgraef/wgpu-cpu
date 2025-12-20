use std::{
    fmt::Debug,
    ops::Index,
};

use cranelift_codegen::{
    ir,
    isa::TargetIsa,
};

use crate::compiler::{
    compiler::State,
    types::{
        FloatWidth,
        IntWidth,
        MatrixType,
        ScalarType,
        VectorType,
    },
};

#[derive(Clone, Copy, Debug)]
pub struct SimdContext {
    pub register_i8: SimdRegister,
    pub register_i32: SimdRegister,
    pub register_f16: SimdRegister,
    pub register_f32: SimdRegister,
}

impl SimdContext {
    pub fn new(isa: &dyn TargetIsa) -> Self {
        Self {
            register_i8: SimdRegister::from_isa(isa, ir::types::I8),
            register_i32: SimdRegister::from_isa(isa, ir::types::I32),
            register_f16: SimdRegister::from_isa(isa, ir::types::F16),
            register_f32: SimdRegister::from_isa(isa, ir::types::F32),
        }
    }

    pub fn vector(&self, vector_type: VectorType) -> SimdValues {
        let (count, ty) = match self[vector_type] {
            VectorIrType::Plain { ty } => (u8::from(vector_type.size).into(), ty),
            VectorIrType::Vector { ty } => (1, ty),
        };
        SimdValues { count, ty }
    }

    pub fn matrix(&self, matrix_type: MatrixType) -> SimdValues {
        let (count, ty) = match self[matrix_type] {
            MatrixIrType::Plain { ty } => (matrix_type.num_elements(), ty),
            MatrixIrType::ColumnVector { ty } => (matrix_type.rows.into(), ty),
            MatrixIrType::FullVector { ty } => (1, ty),
        };
        SimdValues { count, ty }
    }

    pub fn simd_immediates(&self, _state: &mut State) -> SimdImmediates {
        /*let vector_reduce_shuffle_masks = SIMD_VECTOR_REDUCE_SHUFFLE_MASKS
            .map(|mask| state.cl_context.func.dfg.immediates.push(mask.into()));
        let matrix_transpose_shuffle_masks = {
            let vector_sizes = [2, 3, 4];
            vector_sizes.map(|columns| {
                vector_sizes.map(|rows| {
                    let mask = make_transpose_shuffle_mask(columns, rows);
                    state.cl_context.func.dfg.immediates.push(mask.into())
                })
            })
        };*/

        SimdImmediates {
            //vector_reduce_shuffle_masks,
            //matrix_transpose_shuffle_masks,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SimdValues {
    pub count: u8,
    pub ty: ir::Type,
}

impl SimdValues {
    pub fn stride(&self) -> i32 {
        self.ty.bytes().try_into().expect("stack offset overflow")
    }
}

impl Index<ScalarType> for SimdContext {
    type Output = SimdRegister;

    fn index(&self, index: ScalarType) -> &Self::Output {
        match index {
            ScalarType::Bool => &self.register_i8,
            ScalarType::Int(_signedness, IntWidth::I32) => &self.register_i32,
            ScalarType::Float(FloatWidth::F16) => &self.register_f16,
            ScalarType::Float(FloatWidth::F32) => &self.register_f32,
        }
    }
}

impl Index<VectorType> for SimdContext {
    type Output = VectorIrType;

    fn index(&self, index: VectorType) -> &Self::Output {
        let i = usize::from(u8::from(index.size)) - 2;
        &self[index.scalar].vector[i]
    }
}

impl Index<MatrixType> for SimdContext {
    type Output = MatrixIrType;

    fn index(&self, index: MatrixType) -> &Self::Output {
        let i = usize::from(u8::from(index.columns)) - 2;
        let j = usize::from(u8::from(index.rows)) - 2;
        &self[index.scalar].matrix[i][j]
    }
}

#[derive(Clone, Copy)]
pub struct SimdRegister {
    pub lanes: u32,
    pub vector: [VectorIrType; 3],
    pub matrix: [[MatrixIrType; 3]; 3],
}

impl Debug for SimdRegister {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("VectorRegister");
        s.field("lanes", &self.lanes);
        for i in 0..3 {
            s.field(&format!("vec{}", i + 2), &self.vector[i]);
        }
        for i in 0..3 {
            for j in 0..3 {
                s.field(&format!("mat{}x{}", i + 2, j + 2), &self.matrix[i][j]);
            }
        }
        s.finish()
    }
}

impl SimdRegister {
    pub fn from_isa(isa: &dyn TargetIsa, base_type: ir::Type) -> Self {
        Self::new(
            base_type,
            isa.dynamic_vector_bytes(base_type) / base_type.bytes(),
        )
    }

    pub fn new(base_type: ir::Type, lanes: u32) -> Self {
        const VECTOR_SIZES: [naga::VectorSize; 3] = [
            naga::VectorSize::Bi,
            naga::VectorSize::Tri,
            naga::VectorSize::Quad,
        ];

        let vector = VECTOR_SIZES.map(|vector_size| {
            let required_lanes = u32::from(vector_size).next_power_of_two();
            if required_lanes <= lanes {
                VectorIrType::Vector {
                    ty: base_type.by(required_lanes).unwrap(),
                }
            }
            else {
                VectorIrType::Plain { ty: base_type }
            }
        });

        let matrix = VECTOR_SIZES.map(|columns| {
            VECTOR_SIZES.map(|rows| {
                let column_lanes = u32::from(columns).next_power_of_two();
                let full_lanes = (u32::from(columns) * u32::from(rows)).next_power_of_two();

                if lanes >= full_lanes {
                    MatrixIrType::FullVector {
                        ty: base_type.by(full_lanes).unwrap(),
                    }
                }
                else if lanes >= column_lanes {
                    MatrixIrType::ColumnVector {
                        ty: base_type.by(column_lanes).unwrap(),
                    }
                }
                else {
                    MatrixIrType::Plain { ty: base_type }
                }
            })
        });

        Self {
            lanes,
            vector,
            matrix,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorIrType {
    Plain { ty: ir::Type },
    Vector { ty: ir::Type },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixIrType {
    Plain { ty: ir::Type },
    ColumnVector { ty: ir::Type },
    FullVector { ty: ir::Type },
}

/*
fn make_transpose_shuffle_mask_full(columns: u8, rows: u8) -> ir::immediates::V128Imm {
    let mut mask = [0; 16];

    let lanes = MatrixLanes::new(columns, rows);

    lanes.for_each(|lane, row, column| {
        mask[usize::from(lane)] = lanes.lane(row, column);
    });

    V128Imm(mask)
}

const SIMD_VECTOR_REDUCE_SHUFFLE_MASKS: [ir::immediates::V128Imm; 2] = [
    ir::immediates::V128Imm([1, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ir::immediates::V128Imm([2, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
];
 */

#[derive(Clone, Copy, Debug)]
pub struct SimdImmediates {
    //pub vector_reduce_shuffle_masks: [ir::Immediate; 2],
    //pub matrix_transpose_shuffle_masks: [[ir::Immediate; 3]; 3],
}
