use std::cmp::Ordering;

pub fn bubblesort3<T>(mut values: &mut [T; 3], mut compare: impl FnMut(&T, &T) -> Ordering) {
    compare_and_swap(&mut values, &mut compare, 0, 1);
    compare_and_swap(&mut values, &mut compare, 0, 2);
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
    try_compare_and_swap(&mut values, &mut compare, 0, 2)?;
    try_compare_and_swap(&mut values, &mut compare, 1, 2)?;
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

#[cfg(test)]
mod tests {
    use crate::util::sort::bubblesort3;

    #[test]
    fn bubblesort3_bug() {
        // these values triggered an assertion because bubblesort was wrong.
        // i just completely copied the comparision indices incorrectly -.-
        let mut values = [230, 84, 75];

        bubblesort3(&mut values, |a, b| a.cmp(b));
        println!("sorted: {values:?}");
        assert_eq!(values[0], 75);
        assert_eq!(values[1], 84);
        assert_eq!(values[2], 230);
    }

    #[test]
    fn random_bubblesorts() {
        let pairs = [
            [[847, 895, 730], [730, 847, 895]],
            [[407, 622, 50], [50, 407, 622]],
            [[498, 698, 242], [242, 498, 698]],
            [[729, 619, 845], [619, 729, 845]],
            [[78, 484, 855], [78, 484, 855]],
            [[856, 128, 44], [44, 128, 856]],
            [[64, 618, 512], [64, 512, 618]],
            [[132, 858, 663], [132, 663, 858]],
            [[122, 182, 486], [122, 182, 486]],
            [[521, 930, 464], [464, 521, 930]],
            [[536, 956, 417], [417, 536, 956]],
            [[440, 364, 860], [364, 440, 860]],
            [[46, 855, 920], [46, 855, 920]],
            [[226, 163, 389], [163, 226, 389]],
            [[369, 992, 285], [285, 369, 992]],
            [[823, 464, 856], [464, 823, 856]],
            [[847, 1000, 179], [179, 847, 1000]],
            [[144, 7, 213], [7, 144, 213]],
            [[839, 82, 108], [82, 108, 839]],
            [[36, 649, 856], [36, 649, 856]],
            [[508, 363, 603], [363, 508, 603]],
            [[252, 34, 29], [29, 34, 252]],
            [[274, 629, 663], [274, 629, 663]],
            [[900, 439, 406], [406, 439, 900]],
            [[231, 941, 884], [231, 884, 941]],
            [[98, 524, 25], [25, 98, 524]],
            [[714, 63, 201], [63, 201, 714]],
            [[587, 748, 250], [250, 587, 748]],
            [[569, 602, 123], [123, 569, 602]],
            [[896, 104, 31], [31, 104, 896]],
            [[525, 463, 510], [463, 510, 525]],
            [[290, 324, 18], [18, 290, 324]],
            [[822, 770, 850], [770, 822, 850]],
            [[330, 592, 965], [330, 592, 965]],
            [[489, 70, 491], [70, 489, 491]],
            [[166, 434, 779], [166, 434, 779]],
            [[430, 435, 367], [367, 430, 435]],
            [[580, 220, 810], [220, 580, 810]],
            [[807, 391, 254], [254, 391, 807]],
            [[214, 297, 524], [214, 297, 524]],
            [[494, 777, 964], [494, 777, 964]],
            [[452, 409, 62], [62, 409, 452]],
            [[676, 631, 494], [494, 631, 676]],
            [[297, 97, 938], [97, 297, 938]],
            [[381, 788, 572], [381, 572, 788]],
            [[330, 192, 138], [138, 192, 330]],
            [[354, 229, 575], [229, 354, 575]],
            [[290, 110, 807], [110, 290, 807]],
            [[258, 313, 183], [183, 258, 313]],
            [[253, 103, 815], [103, 253, 815]],
        ];

        for [x, y] in pairs {
            let mut y2 = y;
            bubblesort3(&mut y2, |a, b| a.cmp(b));
            assert_eq!(y, y2);
        }
    }
}
