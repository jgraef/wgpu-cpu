#![allow(dead_code)]

use std::{
    convert::Infallible,
    fmt::Debug,
    hash::Hash,
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
    UniqueArena,
};

#[derive(derive_more::Debug)]
#[debug(bound(V: Debug))]
pub struct CoArena<K, V> {
    items: Vec<V>,
    _phantom: PhantomData<K>,
}

impl<K, V> CoArena<K, V> {
    pub(crate) fn new(items: Vec<V>) -> Self {
        Self {
            items,
            _phantom: PhantomData,
        }
    }

    pub(crate) fn try_from_arena_iter<'a, E>(
        arena: impl Iterator<Item = (Handle<K>, &'a K)>,
        mut map: impl FnMut(Handle<K>, &'a K) -> Result<V, E>,
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

    pub fn from_arena<'a>(arena: &'a Arena<K>, mut map: impl FnMut(Handle<K>, &'a K) -> V) -> Self {
        Self::try_from_arena_iter::<Infallible>(arena.iter(), |h, k| Ok(map(h, k)))
            .unwrap_or_else(|e| match e {})
    }

    pub fn from_unique_arena<'a>(
        arena: &'a UniqueArena<K>,
        mut map: impl FnMut(Handle<K>, &'a K) -> V,
    ) -> Self
    where
        K: Eq + Hash,
    {
        Self::try_from_arena_iter::<Infallible>(arena.iter(), |h, k| Ok(map(h, k)))
            .unwrap_or_else(|e| match e {})
    }

    pub fn try_from_arena<'a, E>(
        arena: &'a Arena<K>,
        map: impl FnMut(Handle<K>, &'a K) -> Result<V, E>,
    ) -> Result<Self, E> {
        Self::try_from_arena_iter(arena.iter(), map)
    }

    pub fn try_from_unique_arena<'a, E>(
        arena: &'a UniqueArena<K>,
        map: impl FnMut(Handle<K>, &'a K) -> Result<V, E>,
    ) -> Result<Self, E>
    where
        K: Eq + Hash,
    {
        Self::try_from_arena_iter(arena.iter(), map)
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

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
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

    for (handle, _expression) in function.expressions.iter() {
        typifier
            .grow(handle, &function.expressions, &resolve_context)
            .unwrap();
    }

    typifier
}

#[cfg(test)]
pub(crate) mod test {
    use std::fmt::Debug;

    pub fn assert_send<T: Send>() {}
    pub fn assert_sync<T: Sync>() {}
    pub fn assert_send_sync<T: Send + Sync>() {}

    #[macro_export]
    macro_rules! make_tests {
        ($backend:ty => ($($name:ident,)*)) => {
            $(
                #[test]
                fn $name() {
                    $crate::util::test::BackendTestHelper(<$backend as ::std::default::Default>::default()).$name();
                }
            )*
        };
    }

    #[macro_export]
    macro_rules! make_all_tests {
        ($backend:ty) => {
            make_tests!(
                $backend => (
                    init_variable,
                    store_variable,
                    casts,
                    binops_scalars,
                    comparisions,
                    unops,
                    if_stmt,
                    early_return,
                    if_early_return,
                )
            );
        };
    }

    use approx::{
        AbsDiffEq,
        assert_abs_diff_eq,
    };
    use bytemuck::Pod;

    use crate::{
        backend::{
            Backend,
            Module,
        },
        bindings::{
            ShaderInput,
            ShaderOutput,
        },
        entry_point::EntryPointIndex,
    };

    #[derive(Clone, Copy, Debug)]
    pub struct BackendTestHelper<B>(pub B);

    impl<B> BackendTestHelper<B>
    where
        B: Backend,
    {
        #[track_caller]
        pub fn exec<T>(&self, source: &str) -> T
        where
            T: Pod,
        {
            let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| {
                println!("{source}");
                panic!("{e}");
            });
            let mut validator = naga::valid::Validator::new(Default::default(), Default::default());
            let info = validator.validate(&module).unwrap();

            let module = self.0.create_module(module, info).unwrap();

            struct EvalInput;

            impl ShaderInput for EvalInput {
                fn write_into(
                    &self,
                    _binding: &naga::Binding,
                    _ty: &naga::Type,
                    _target: &mut [u8],
                ) {
                    // nop
                }
            }

            struct EvalOutput<T> {
                output: T,
            }

            impl<T> ShaderOutput for EvalOutput<T>
            where
                T: Pod,
            {
                fn read_from(&mut self, binding: &naga::Binding, _ty: &naga::Type, source: &[u8]) {
                    match binding {
                        naga::Binding::Location { location: 0, .. } => {
                            self.output = *bytemuck::from_bytes::<T>(source);
                        }
                        _ => {}
                    }
                }
            }

            let mut output = EvalOutput {
                output: T::zeroed(),
            };

            module.run_entry_point(EntryPointIndex::from(0), EvalInput, &mut output);

            output.output
        }

        #[track_caller]
        pub fn eval<T>(&self, expression: &str, preamble: &str, out_ty: &str) -> T
        where
            T: Pod,
        {
            let source = format!(
                r#"
        struct Output {{
            @builtin(position) p: vec4f,
            @location(0) output: {out_ty},
        }}

        @vertex
        fn main(@builtin(vertex_index) vertex_index: u32) -> Output {{
            {preamble}
            return Output(vec4f(), {expression});
        }}
        "#
            );

            self.exec(&source)
        }

        #[track_caller]
        pub fn eval_bool(&self, expression: &str, preamble: &str) -> bool {
            let output: u32 = self.eval::<u32>(&format!("u32({expression})"), preamble, "u32");
            match output {
                0 => false,
                1 => true,
                x => panic!("invalid bool: {x}"),
            }
        }

        #[track_caller]
        pub fn assert_wgsl(&self, assertion: &str, preamble: &str) {
            let output = self.eval_bool(assertion, preamble);
            assert!(output);
        }

        pub fn init_variable(&self) {
            let a = self.eval::<u32>("a", "var a: u32 = 123;", "u32");
            assert_eq!(a, 123);
        }

        pub fn store_variable(&self) {
            let a = self.eval::<u32>("a", "var a: u32; a = 123;", "u32");
            assert_eq!(a, 123);
        }

        #[track_caller]
        pub fn test_cast(&self, value: &str, input_ty: &str, output_ty: &str, expected: &str) {
            self.assert_wgsl(
                &format!("output == {expected}"),
                &format!(
                    r#"
        var input: {input_ty} = {value};
        var output: {output_ty} = {output_ty}(input);
        "#
                ),
            );
        }

        pub fn casts(&self) {
            self.test_cast("false", "bool", "u32", "0");
            self.test_cast("true", "bool", "u32", "1");
            self.test_cast("false", "bool", "i32", "0");
            self.test_cast("true", "bool", "i32", "1");
            self.test_cast("false", "bool", "f32", "0.0");
            self.test_cast("true", "bool", "f32", "1.0");
            self.test_cast("5", "u32", "f32", "5.0");
            self.test_cast("-3", "i32", "f32", "-3.0");
        }

        #[track_caller]
        pub fn test_binop<T>(&self, ty: &str, left: &str, op: &str, right: &str, expected: T)
        where
            T: Pod + AbsDiffEq + Debug,
        {
            let output = self.eval::<T>(
                &format!("left {op} right"),
                &format!(
                    r#"
        var left: {ty} = {left};
        var right: {ty} = {right};
        "#
                ),
                ty,
            );

            assert_abs_diff_eq!(output, expected);
        }

        pub fn binops_scalars(&self) {
            self.test_binop::<i32>("i32", "1", "+", "1", 2);
            self.test_binop::<i32>("i32", "2", "-", "1", 1);
            self.test_binop::<i32>("i32", "1", "-", "2", -1);
            self.test_binop::<i32>("i32", "2", "*", "3", 6);
            self.test_binop::<i32>("i32", "2", "*", "-3", -6);
            self.test_binop::<i32>("i32", "6", "/", "2", 3);
            self.test_binop::<i32>("i32", "3", "/", "2", 1);
            self.test_binop::<i32>("i32", "3", "%", "2", 1);

            self.test_binop::<f32>("f32", "1", "+", "1", 2.0);
            self.test_binop::<f32>("f32", "2", "-", "1", 1.0);
            self.test_binop::<f32>("f32", "1", "-", "2", -1.0);
            self.test_binop::<f32>("f32", "2", "*", "3", 6.0);
            self.test_binop::<f32>("f32", "2", "*", "-3", -6.0);
            self.test_binop::<f32>("f32", "6", "/", "2", 3.0);
            self.test_binop::<f32>("f32", "3", "/", "2", 1.5);
            self.test_binop::<f32>("f32", "3", "%", "2", 1.0);
        }

        #[track_caller]
        pub fn test_compare(&self, ty: &str, left: &str, cmp: &str, right: &str, expected: bool) {
            let output = self.eval_bool(
                &format!("left {cmp} right"),
                &format!(
                    r#"
        var left: {ty} = {left};
        var right: {ty} = {right};
        "#
                ),
            );

            assert_eq!(output, expected, "{left} {cmp} {right}");
        }

        pub fn comparisions(&self) {
            self.test_compare("i32", "2", "==", "2", true);
            self.test_compare("i32", "1", "==", "2", false);
            self.test_compare("i32", "1", "!=", "2", true);
            self.test_compare("i32", "1", "<", "2", true);
            self.test_compare("i32", "2", ">", "1", true);
            self.test_compare("i32", "1", "<=", "2", true);
            self.test_compare("i32", "2", "<=", "2", true);
            self.test_compare("i32", "2", ">=", "2", true);
            self.test_compare("i32", "3", ">=", "2", true);
            self.test_compare("i32", "-1", "<", "1", true);
        }

        #[track_caller]
        pub fn test_unop<T>(&self, ty: &str, op: &str, input: &str, expected: T)
        where
            T: Pod + AbsDiffEq + Debug,
        {
            let output = self.eval::<T>(
                &format!("{op} input"),
                &format!(
                    r#"
        var input: {ty} = {input};
        "#
                ),
                ty,
            );

            assert_abs_diff_eq!(output, expected);
        }

        #[track_caller]
        pub fn test_bool_unop(&self, op: &str, input: &str, expected: bool) {
            let output = self.eval_bool(
                &format!("{op} input"),
                &format!(
                    r#"
        var input: bool = {input};
        "#
                ),
            );

            assert_eq!(output, expected);
        }

        pub fn unops(&self) {
            self.test_unop::<i32>("i32", "-", "123", -123);
            self.test_unop::<i32>("i32", "-", "-123", 123);
            self.test_unop::<f32>("f32", "-", "123.0", -123.0);
            self.test_unop::<f32>("f32", "-", "-123.0", 123.0);
            self.test_bool_unop("!", "true", false);
            self.test_bool_unop("!", "false", true);
            self.test_unop("u32", "~", "123", !123);
        }

        pub fn if_stmt(&self) {
            self.assert_wgsl(
                "x == 1",
                "var x: u32; var c: bool = true; if c { x = 1; } else { x = 2; }",
            );
            self.assert_wgsl(
                "x == 2",
                "var x: u32; var c: bool = false; if c { x = 1; } else { x = 2; }",
            );
        }

        pub fn early_return(&self) {
            let out = self.eval::<u32>("123", "return Output(vec4f(), 456);", "u32");
            assert_eq!(out, 456);
        }

        pub fn if_early_return(&self) {
            let out = self.eval::<u32>(
                "123",
                "var c: bool = true; if c { return Output(vec4f(), 456); }",
                "u32",
            );
            assert_eq!(out, 456);
        }
    }
}
