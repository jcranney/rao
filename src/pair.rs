use thiserror::Error;

/// The `Pair` type is a tuple of borrowed variables of the same type.
#[derive(Debug, Clone)]
pub struct Pair<'a, T>(pub &'a T, pub &'a T);

#[derive(Debug, Error)]
pub enum PairCheckError {
    #[error("multiple matches, not a safe set of pairs")]
    AmbiguousPairs,
}

/// `Pair` implements a strict PartialEq based on the raw pointers of the
/// underlying data. This is probably unnecessary, but I want to be completely
/// sure that two pairs are only considered identical if they are independently
/// initialised. In practice, this turns out to be less strict than I'd like for
/// many data types, like float, which appear to resolve (e.g.) 5.0 and 5.0 to
/// the same raw pointer. I don't understand why, but I supposed I don't need
/// to.
impl<'a, T> PartialEq for Pair<'a, T> {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.0, other.0) && std::ptr::eq(self.1, other.1)
    }
}

/// Now we implement the core functionality for telescoping pairs.
impl<'a, T: std::clone::Clone> Pair<'a, T> {
    /// this is the high level API for this functionality. This function takes
    /// a vector of pairs of generic values, and returns a vector of pairs with
    /// the consecutive pairs merged. E.g., [(a, b), (b, c)] -> [(a, c)].
    pub fn reduce_pairs(mut pairs: Vec<Self>) -> Vec<Self> {
        let mut result: Vec<Pair<T>> = vec![];
        while let Some(pair_a) = pairs.pop() {
            result.insert(0, Self::_recurse_pairs(pair_a, &mut pairs));
        }
        result
    }

    /// this function will return an error if the input pairs have some duplication
    /// or ambiguity. E.g.,  [(a, b), (b, c), (c, d)] is valid, but
    /// [(a, b), (b, c), (b, d)] is not valid, because it's not clear if (a, b)
    /// should join with (b, c) or (b, d). This ambiguity is undefined behaivour
    /// so a user should call `check_pairs` first to guarantee that their data
    /// is going to produce valid results when `reduce_pairs` is executed. This
    /// function is relatively slow, so if the user can guarantee that the input
    /// data is unambiguous, they should skip this step.
    pub fn check_pairs(pairs: &[Self]) -> Result<(), PairCheckError> {
        for (index_a, pair_a) in pairs.iter().enumerate() {
            let mut inner_matches: usize = 0;
            let mut outer_matches: usize = 0;
            for pair_b in pairs[index_a + 1..].iter() {
                if std::ptr::eq(pair_a.0, pair_b.1) {
                    inner_matches += 1;
                }
                if std::ptr::eq(pair_a.1, pair_b.0) {
                    outer_matches += 1;
                }
            }
            if inner_matches > 1 || outer_matches > 1 {
                return Err(PairCheckError::AmbiguousPairs);
            }
        }
        Ok(())
    }

    /// recursive function used to find a matching chain of pairs, returning the
    /// reduced pair.
    fn _recurse_pairs(start_pair: Self, input_pairs: &mut Vec<Self>) -> Self {
        for (index, pair) in input_pairs.clone().iter().enumerate() {
            if std::ptr::eq(start_pair.1, pair.0) {
                input_pairs.remove(index);
                return Self::_recurse_pairs(Pair(start_pair.0, pair.1), input_pairs);
            }
            if std::ptr::eq(start_pair.0, pair.1) {
                input_pairs.remove(index);
                return Self::_recurse_pairs(Pair(pair.0, start_pair.1), input_pairs);
            }
        }
        start_pair
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_telescoping_pairs() {
        // define some points with x and y coordinates
        let p1 = (1, 0);
        let p2 = (2, 0);
        let p3 = (3, 0);
        let p4 = (4, 0);
        let p5 = (5, 0);
        let p6 = (6, 0);
        // pair up some of those points
        let pairs = vec![
            Pair(&p2, &p3),
            Pair(&p3, &p4),
            Pair(&p1, &p2),
            Pair(&p5, &p6),
        ];
        // reduce the set of pairs by any that telescope
        let reduced_pairs = Pair::reduce_pairs(pairs);
        // make sure that the reduced pairs are correct
        assert_eq!(reduced_pairs.len(), 2);
        assert_eq!(reduced_pairs[0], Pair(&p1, &p4));
        assert_eq!(reduced_pairs[1], Pair(&p5, &p6));
    }

    #[test]
    fn valid_pairs() {
        // define some points with x and y coordinates
        let p1 = (1, 0);
        let p2 = (2, 0);
        let p3 = (3, 0);
        let p4 = (4, 0);
        let p5 = (5, 0);
        let p6 = (6, 0);
        // pair up some of those points
        let pairs = vec![
            Pair(&p1, &p2),
            Pair(&p2, &p3),
            Pair(&p3, &p4),
            Pair(&p5, &p6),
        ];
        // check that the pairs are valid, would panic if not valid
        assert!(Pair::check_pairs(&pairs).is_ok());
    }

    #[test]
    fn invalid_pairs() {
        // define some points with x and y coordinates
        let p1 = (1, 0);
        let p2 = (2, 0);
        let p3 = (3, 0);
        let p4 = (4, 0);
        let p5 = (5, 0);
        let p6 = (6, 0);
        // pair up some of those points
        let pairs = vec![
            Pair(&p1, &p2),
            Pair(&p2, &p3),
            Pair(&p3, &p4),
            Pair(&p3, &p5), // this is an invalid pair, since it has a common
            // root to the pair above.
            Pair(&p4, &p6),
        ];
        // check that the pairs are valid, would panic if not valid
        assert!(Pair::check_pairs(&pairs).is_err());
    }
}
