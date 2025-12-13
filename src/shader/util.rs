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
};

#[derive(derive_more::Debug)]
#[debug(bound(V: Debug))]
pub struct CoArena<K, V> {
    items: Vec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V> CoArena<K, V> {
    pub fn new(arena: &Arena<K>, mut map: impl FnMut(Handle<K>, &K) -> V) -> Self {
        Self {
            items: arena
                .iter()
                .map(|(handle, value)| map(handle, value))
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
