use std::cmp::Ordering;

pub fn bubblesort3<T>(mut values: &mut [T; 3], mut compare: impl FnMut(&T, &T) -> Ordering) {
    compare_and_swap(&mut values, &mut compare, 0, 1);
    compare_and_swap(&mut values, &mut compare, 1, 2);
    compare_and_swap(&mut values, &mut compare, 1, 2);
}

pub fn compare_and_swap<T>(
    values: &mut [T; 3],
    mut compare: impl FnMut(&T, &T) -> Ordering,
    left: usize,
    right: usize,
) -> bool {
    if matches!(compare(&values[left], &values[right]), Ordering::Greater) {
        values.swap(left, right);
        true
    }
    else {
        false
    }
}

pub fn try_bubblesort3<T>(
    mut values: &mut [T; 3],
    mut compare: impl FnMut(&T, &T) -> Option<Ordering>,
) -> Result<(), SortError> {
    try_compare_and_swap(&mut values, &mut compare, 0, 1)?;
    try_compare_and_swap(&mut values, &mut compare, 1, 2)?;
    try_compare_and_swap(&mut values, &mut compare, 2, 0)?;
    Ok(())
}

pub fn try_compare_and_swap<T>(
    values: &mut [T; 3],
    mut compare: impl FnMut(&T, &T) -> Option<Ordering>,
    left: usize,
    right: usize,
) -> Result<bool, SortError> {
    match compare(&values[left], &values[right]) {
        None => Err(SortError { left, right }),
        Some(Ordering::Greater) => {
            values.swap(left, right);
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
#[error("Can't compare index {left} with {right}")]
pub struct SortError {
    pub left: usize,
    pub right: usize,
}
