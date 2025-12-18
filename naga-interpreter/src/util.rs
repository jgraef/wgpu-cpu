#![allow(dead_code)]

use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{
        Index,
        IndexMut,
    },
};

use naga::{
    Arena,
    Handle,
    Range,
};

#[derive(derive_more::Debug)]
#[debug(bound(V: Debug))]
pub struct CoArena<K, V> {
    items: Vec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V> CoArena<K, V> {
    pub fn from_arena(arena: &Arena<K>, mut map: impl FnMut(Handle<K>, &K) -> V) -> Self {
        Self {
            items: arena
                .iter()
                .enumerate()
                .map(|(i, (handle, value))| {
                    assert_eq!(i, handle.index());
                    map(handle, value)
                })
                .collect(),
            _phantom: PhantomData,
        }
    }
}

impl<K, V> Index<Handle<K>> for CoArena<K, V> {
    type Output = V;

    fn index(&self, index: Handle<K>) -> &Self::Output {
        &self.items[index.index()]
    }
}

impl<K, V> IndexMut<Handle<K>> for CoArena<K, V> {
    fn index_mut(&mut self, index: Handle<K>) -> &mut Self::Output {
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

    pub fn from_arena(arena: &Arena<K>, mut map: impl FnMut(Handle<K>, &K) -> Option<V>) -> Self {
        Self {
            items: arena
                .iter()
                .enumerate()
                .map(|(i, (handle, value))| {
                    assert_eq!(i, handle.index());
                    map(handle, value)
                })
                .collect(),
            _phantom: PhantomData,
        }
    }

    pub fn insert(&mut self, handle: Handle<K>, value: V) {
        self.items.insert(handle.index(), value);
    }

    pub fn contains(&self, handle: Handle<K>) -> bool {
        self.items.contains(handle.index())
    }

    pub fn get(&self, handle: Handle<K>) -> Option<&V> {
        self.items.get(handle.index())
    }

    pub fn get_mut(&mut self, handle: Handle<K>) -> Option<&mut V> {
        self.items.get_mut(handle.index())
    }

    pub fn reserve_range(&mut self, range: &Range<K>) {
        if let Some((_, handle)) = range.first_and_last() {
            self.items.reserve_for_index(handle.index());
        }
    }
}

impl<K, V> Index<Handle<K>> for SparseCoArena<K, V> {
    type Output = V;

    fn index(&self, index: Handle<K>) -> &Self::Output {
        &self.items[index.index()]
    }
}

impl<K, V> IndexMut<Handle<K>> for SparseCoArena<K, V> {
    fn index_mut(&mut self, index: Handle<K>) -> &mut Self::Output {
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

pub fn typifier_from_function(
    module: &naga::Module,
    function: &naga::Function,
) -> naga::front::Typifier {
    let mut typifier = naga::front::Typifier::default();
    let resolve_context = naga::proc::ResolveContext::with_locals(
        module,
        &function.local_variables,
        &function.arguments,
    );

    for (handle, expression) in function.expressions.iter() {
        typifier
            .grow(handle, &function.expressions, &resolve_context)
            .unwrap();
    }

    typifier
}
