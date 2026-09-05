//! How long a search may run.

/// The budget a time-bounded drawing algorithm is given.
///
/// Encoding "no limit" as a sentinel value of the limit itself — the zero that
/// [`SearchBudget::Milliseconds`] would otherwise carry — puts an unbounded run
/// one slipped `0` away from any caller: an unset field, a config value that
/// parsed as empty, a derived `Default`. Naming it as its own case means a
/// search runs without a limit only where someone wrote
/// [`SearchBudget::Unbounded`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchBudget {
    /// Stop exploring new branches once this many milliseconds have passed.
    ///
    /// Running out is not a failure: the best solution found so far is
    /// returned, so a truncated search still yields a usable drawing. Zero is
    /// therefore meaningful — it takes whatever the initial heuristic produced,
    /// without refining it.
    Milliseconds(u64),
    /// Run until the search is exhausted and the result is a proven optimum.
    ///
    /// The cost of that proof grows steeply with the size of the lattice, so
    /// this belongs in benchmarks, tests and deliberate offline runs — never in
    /// a default.
    Unbounded,
}

/// The budget the drawing algorithms take when the caller does not name one.
pub const DEFAULT_SEARCH_BUDGET_MS: u64 = 1000;

impl Default for SearchBudget {
    fn default() -> Self {
        SearchBudget::Milliseconds(DEFAULT_SEARCH_BUDGET_MS)
    }
}

impl SearchBudget {
    /// The limit in milliseconds, or `None` if there is none.
    pub fn limit_ms(self) -> Option<u64> {
        match self {
            SearchBudget::Milliseconds(ms) => Some(ms),
            SearchBudget::Unbounded => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_the_default_budget_is_bounded() {
        assert_eq!(
            SearchBudget::default().limit_ms(),
            Some(DEFAULT_SEARCH_BUDGET_MS)
        );
    }

    /// Zero milliseconds is a real budget, not a way of spelling "unbounded".
    #[test]
    fn test_zero_milliseconds_is_a_limit() {
        assert_eq!(SearchBudget::Milliseconds(0).limit_ms(), Some(0));
        assert_eq!(SearchBudget::Unbounded.limit_ms(), None);
    }
}
