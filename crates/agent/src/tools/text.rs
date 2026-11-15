//! Text cuts that never split a UTF-8 character.

/// The largest index `<= max` that is a char boundary of `s`. Returns
/// `s.len()` when `max >= s.len()`.
pub(crate) fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    (0..=max)
        .rev()
        .find(|&i| s.is_char_boundary(i))
        .unwrap_or(0)
}

/// Truncate `s` to at most `max` bytes, cutting on a char boundary.
pub(crate) fn truncate_at_char_boundary(s: &mut String, max: usize) {
    let i = floor_char_boundary(s, max);
    s.truncate(i);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_boundary_inside_a_char() {
        // "é" is two bytes: index 1 is inside it.
        assert_eq!(floor_char_boundary("é", 1), 0);
        assert_eq!(floor_char_boundary("aé", 2), 1);
        assert_eq!(floor_char_boundary("abc", 10), 3);
        assert_eq!(floor_char_boundary("abc", 2), 2);
    }

    #[test]
    fn truncate_keeps_valid_utf8() {
        let mut s = "é".repeat(60);
        truncate_at_char_boundary(&mut s, 101);
        assert_eq!(s.len(), 100);
        assert!(s.chars().all(|c| c == 'é'));
    }

    #[test]
    fn web_fetch_cut_is_valid_string() {
        // The same cut that WebFetchTool makes on its body.
        let body = "é".repeat(10);
        let cut = &body[..floor_char_boundary(&body, 5)];
        assert_eq!(cut, "éé");
    }
}
