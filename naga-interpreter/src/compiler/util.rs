use cranelift_codegen::ir::immediates::Ieee16;
use half::f16;

pub fn ieee16_from_f16(x: f16) -> Ieee16 {
    Ieee16::with_bits(x.to_bits())
}

pub(super) fn alignment_log2(alignment: naga::proc::Alignment) -> u8 {
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
