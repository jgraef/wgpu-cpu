use std::{
    convert::Infallible,
    fmt::Debug,
    hash::Hash,
    marker::PhantomData,
    ops::{
        Index,
        IndexMut,
    },
    sync::Arc,
};

use arrayvec::ArrayVec;
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
    pub functions: Vec<(cranelift_module::FuncId, cranelift_codegen::ir::Function)>,
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
        self.functions.push((func, function));
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

pub(super) fn math_args_to_array_vec(
    arg0: naga::Handle<naga::Expression>,
    arg1: Option<naga::Handle<naga::Expression>>,
    arg2: Option<naga::Handle<naga::Expression>>,
    arg3: Option<naga::Handle<naga::Expression>>,
) -> ArrayVec<naga::Handle<naga::Expression>, 4> {
    let mut args = ArrayVec::new();

    args.push(arg0);
    let mut rest = [arg1, arg2, arg3].into_iter();
    while let Some(Some(arg)) = rest.next() {
        args.push(arg);
    }
    while let Some(arg) = rest.next() {
        assert!(arg.is_none());
    }

    args
}

#[derive(derive_more::Debug)]
#[debug(bound(V: Debug))]
pub struct CoArena<K, V> {
    items: Vec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V> CoArena<K, V> {
    pub(crate) fn try_from_arena_iter<'a, E>(
        arena: impl Iterator<Item = (naga::Handle<K>, &'a K)>,
        mut map: impl FnMut(naga::Handle<K>, &'a K) -> Result<V, E>,
    ) -> Result<Self, E>
    where
        K: 'a,
    {
        Ok(Self {
            items: arena
                .enumerate()
                .map(|(i, (handle, value))| {
                    assert_eq!(i, handle.index());
                    map(handle, value)
                })
                .collect::<Result<_, E>>()?,
            _phantom: PhantomData,
        })
    }

    pub fn from_arena<'a>(
        arena: &'a naga::Arena<K>,
        mut map: impl FnMut(naga::Handle<K>, &'a K) -> V,
    ) -> Self {
        Self::try_from_arena_iter::<Infallible>(arena.iter(), |h, k| Ok(map(h, k)))
            .unwrap_or_else(|e| match e {})
    }

    pub fn from_unique_arena<'a>(
        arena: &'a naga::UniqueArena<K>,
        mut map: impl FnMut(naga::Handle<K>, &'a K) -> V,
    ) -> Self
    where
        K: Eq + Hash,
    {
        Self::try_from_arena_iter::<Infallible>(arena.iter(), |h, k| Ok(map(h, k)))
            .unwrap_or_else(|e| match e {})
    }

    pub fn try_from_arena<'a, E>(
        arena: &'a naga::Arena<K>,
        map: impl FnMut(naga::Handle<K>, &'a K) -> Result<V, E>,
    ) -> Result<Self, E> {
        Self::try_from_arena_iter(arena.iter(), map)
    }

    pub fn try_from_unique_arena<'a, E>(
        arena: &'a naga::UniqueArena<K>,
        map: impl FnMut(naga::Handle<K>, &'a K) -> Result<V, E>,
    ) -> Result<Self, E>
    where
        K: Eq + Hash,
    {
        Self::try_from_arena_iter(arena.iter(), map)
    }
}

impl<K, V> Index<naga::Handle<K>> for CoArena<K, V> {
    type Output = V;

    fn index(&self, index: naga::Handle<K>) -> &Self::Output {
        &self.items[index.index()]
    }
}

impl<K, V> IndexMut<naga::Handle<K>> for CoArena<K, V> {
    fn index_mut(&mut self, index: naga::Handle<K>) -> &mut Self::Output {
        &mut self.items[index.index()]
    }
}

impl<K, V> Clone for CoArena<K, V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            _phantom: PhantomData,
        }
    }
}

#[derive(derive_more::Debug)]
#[debug(bound(V: Debug))]
pub struct SparseCoArena<K, V> {
    items: SparseVec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V> Default for SparseCoArena<K, V> {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl<K, V> SparseCoArena<K, V> {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: SparseVec::with_capacity(capacity),
            _phantom: PhantomData,
        }
    }

    pub fn from_arena(
        arena: &naga::Arena<K>,
        mut map: impl FnMut(naga::Handle<K>, &K) -> Option<V>,
    ) -> Self {
        let items = arena
            .iter()
            .enumerate()
            .map(|(i, (handle, value))| {
                assert_eq!(i, handle.index());
                map(handle, value)
            })
            .collect();

        Self {
            items,
            _phantom: PhantomData,
        }
    }

    pub fn insert(&mut self, handle: naga::Handle<K>, value: V) {
        self.items.insert(handle.index(), value);
    }

    pub fn contains(&self, handle: naga::Handle<K>) -> bool {
        self.items.contains(handle.index())
    }

    pub fn get(&self, handle: naga::Handle<K>) -> Option<&V> {
        self.items.get(handle.index())
    }

    pub fn get_mut(&mut self, handle: naga::Handle<K>) -> Option<&mut V> {
        self.items.get_mut(handle.index())
    }

    pub fn reserve_range(&mut self, range: &naga::Range<K>) {
        if let Some((_, handle)) = range.first_and_last() {
            self.items.reserve_for_index(handle.index());
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

impl<K, V> Index<naga::Handle<K>> for SparseCoArena<K, V> {
    type Output = V;

    fn index(&self, index: naga::Handle<K>) -> &Self::Output {
        &self.items[index.index()]
    }
}

impl<K, V> IndexMut<naga::Handle<K>> for SparseCoArena<K, V> {
    fn index_mut(&mut self, index: naga::Handle<K>) -> &mut Self::Output {
        &mut self.items[index.index()]
    }
}

impl<K, V> Clone for SparseCoArena<K, V>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            items: self.items.clone(),
            _phantom: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct SparseVec<T> {
    items: Vec<Option<T>>,
    count: usize,
}

impl<T> Default for SparseVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FromIterator<Option<T>> for SparseVec<T> {
    fn from_iter<I: IntoIterator<Item = Option<T>>>(iter: I) -> Self {
        let mut count = 0;
        let items = iter
            .into_iter()
            .inspect(|item| {
                if item.is_some() {
                    count += 1
                }
            })
            .collect();
        Self { items, count }
    }
}

impl<T> From<Vec<Option<T>>> for SparseVec<T> {
    fn from(value: Vec<Option<T>>) -> Self {
        let mut count = 0;
        for item in &value {
            if item.is_some() {
                count += 1;
            }
        }
        Self {
            items: value,
            count,
        }
    }
}

impl<T> SparseVec<T> {
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            items: Vec::with_capacity(capacity),
            count: 0,
        }
    }

    pub fn insert(&mut self, index: usize, value: T) {
        self.reserve_for_index(index);
        self.items[index] = Some(value);
        self.count += 1;
    }

    pub fn remove(&mut self, index: usize) -> Option<T> {
        if index < self.items.len() {
            let item_opt = std::mem::take(&mut self.items[index]);
            if item_opt.is_some() {
                self.count -= 1;
            }
            item_opt
        }
        else {
            None
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.items.get(index).is_some_and(|slot| slot.is_some())
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.items.get(index)?.as_ref()
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.items.get_mut(index)?.as_mut()
    }

    pub fn reserve_for_index(&mut self, index: usize) {
        if index >= self.items.len() {
            self.items.resize_with(index + 1, || None);
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // todo: proper ExactSizeIter and what not
        self.items.iter().filter_map(|item| item.as_ref())
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn into_vec(self) -> Vec<Option<T>> {
        self.items
    }
}

impl<T> Index<usize> for SparseVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.items[index]
            .as_ref()
            .unwrap_or_else(|| panic!("No item at index {index}"))
    }
}

impl<T> IndexMut<usize> for SparseVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.items[index]
            .as_mut()
            .unwrap_or_else(|| panic!("No item at index {index}"))
    }
}

impl<T> Debug for SparseVec<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
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

#[cfg(test)]
pub mod tests {
    pub fn assert_send<T: Send>() {}
    pub fn assert_sync<T: Sync>() {}
    pub fn assert_send_sync<T: Send + Sync>() {}
}
