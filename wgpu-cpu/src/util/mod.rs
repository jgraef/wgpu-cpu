pub mod bresenham;
pub mod interpolation;
pub mod scanline;
pub mod sort;
pub mod sync;

use arrayvec::ArrayVec;

pub fn make_label_owned(label: &Option<&str>) -> Option<String> {
    label.map(ToOwned::to_owned)
}

pub trait IteratorExt: Iterator {
    fn array_chunks_<const N: usize>(self) -> ArrayChunksIter<N, Self>
    where
        Self: Sized,
    {
        ArrayChunksIter { inner: self }
    }
}

impl<T> IteratorExt for T where T: Iterator {}

pub trait ArrayExt<T, const N: usize> {
    fn try_map_<U>(self, f: impl FnMut(T) -> Option<U>) -> Option<[U; N]>;
}

impl<T, const N: usize> ArrayExt<T, N> for [T; N] {
    fn try_map_<U>(self, mut f: impl FnMut(T) -> Option<U>) -> Option<[U; N]> {
        let mut out: ArrayVec<U, N> = ArrayVec::new();

        for item in self {
            out.push(f(item)?);
        }

        Some(
            out.into_inner()
                .unwrap_or_else(|_| unreachable!("output array must be full")),
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ArrayChunksIter<const N: usize, I> {
    inner: I,
}

impl<const N: usize, I> Iterator for ArrayChunksIter<N, I>
where
    I: Iterator,
{
    type Item = [I::Item; N];

    fn next(&mut self) -> Option<Self::Item> {
        let mut buf = ArrayVec::new();

        for i in 0..N {
            if let Some(item) = self.inner.next() {
                buf.push(item);
            }
            else {
                return None;
            }
        }

        Some(buf.into_inner().ok().unwrap())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.inner.size_hint();
        (min / N, max.map(|max| max / N))
    }
}

impl<const N: usize, I> ExactSizeIterator for ArrayChunksIter<N, I> where I: ExactSizeIterator {}
