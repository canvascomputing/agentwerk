//! Shared backoff policy for transient provider failures — one implementation, every provider waits the same way.

/// Cap on the per-attempt backoff so exponential growth doesn't run away past
/// a few attempts. Matches claude-code's `maxDelayMs` default.
pub(crate) const MAX_DELAY_MS: u64 = 32_000;

/// Compute the delay before the next retry attempt.
///
/// Exponential backoff `backoff_ms * 2^attempt`, clamped at [`MAX_DELAY_MS`],
/// then extended by additive jitter in `[0, 25%]` of the clamped value. If
/// the server provides a `retry_after_ms` hint, that value takes precedence
/// and bypasses the cap (the server is explicit about what it wants).
pub(crate) fn compute_delay(backoff_ms: u64, attempt: u32, retry_after_ms: Option<u64>) -> u64 {
    if let Some(server_delay) = retry_after_ms {
        return server_delay;
    }

    let exponential = backoff_ms
        .saturating_mul(1u64 << attempt.min(31))
        .min(MAX_DELAY_MS);
    let jitter_range = exponential / 4;

    if jitter_range == 0 {
        return exponential;
    }

    let entropy = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos() as u64;
    let jitter_offset = entropy % jitter_range;

    exponential.saturating_add(jitter_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_backoff() {
        for attempt in 0..3 {
            let delay = compute_delay(1000, attempt, None);
            let expected_base = 1000u64 * (1u64 << attempt);
            let jitter_range = expected_base / 4;
            assert!(delay >= expected_base);
            assert!(delay <= expected_base + jitter_range);
        }
    }

    #[test]
    fn respects_retry_after() {
        let delay = compute_delay(1000, 0, Some(5000));
        assert_eq!(delay, 5000);
    }

    #[test]
    fn caps_at_max_delay_ms() {
        let delay = compute_delay(1000, 20, None);
        let jitter_range = MAX_DELAY_MS / 4;
        assert!(delay >= MAX_DELAY_MS);
        assert!(delay <= MAX_DELAY_MS + jitter_range);
    }

    #[test]
    fn saturates_instead_of_overflow() {
        let _delay = compute_delay(u64::MAX, 10, None);
    }
}
