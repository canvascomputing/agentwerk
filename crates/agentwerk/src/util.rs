//! Small internal helpers shared across the crate: cancellation-aware sleep, name and date formatting, and the `Retry` strategy used by the request loop.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Resolves when the cancel flag flips to true, polling every 100 ms. Pair
/// with `tokio::select!` to abort any mid-flight work: the loser branch is
/// dropped, which cascades to dropped HTTP futures (reqwest aborts the
/// connection) and dropped child processes (if `kill_on_drop(true)` is set).
pub(crate) async fn wait_for_cancel(cancel: &Arc<AtomicBool>) {
    while !cancel.load(Ordering::Relaxed) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Sleep for `duration`, but bail as soon as `cancel` trips. Returns `true` if
/// the full duration elapsed, `false` if cancel fired. Used by retry backoff
/// to stay responsive to Ctrl-C.
pub(crate) async fn cancellable_sleep(duration: Duration, cancel: &Arc<AtomicBool>) -> bool {
    tokio::select! {
        biased;
        _ = wait_for_cancel(cancel) => false,
        _ = tokio::time::sleep(duration) => true,
    }
}

pub(crate) fn generate_agent_name(name: &str) -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{name}_{nanos}")
}

pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Today's date as `YYYY-MM-DD`, via the civil-from-days algorithm.
/// http://howardhinnant.github.io/date_algorithms.html
pub(crate) fn format_current_date() -> String {
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let days = epoch_secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}")
}

/// Retry strategy: total attempt budget and delay between attempts. Concrete
/// impls live in this module; call sites hold the strategy by value and
/// invoke it through the trait.
pub(crate) trait Retry {
    fn max_attempts(&self) -> u32;

    /// Delay between attempt `attempt` and `attempt + 1` (0-indexed).
    /// `server_hint` (e.g. HTTP `Retry-After`) takes precedence when honoured;
    /// impls that ignore the hint document that choice.
    fn delay(&self, attempt: u32, server_hint: Option<Duration>) -> Duration;
}

/// Error that can classify itself as transient or terminal. Lets retry loops
/// take `<E: Retryable>` without depending on the top-level `Error`.
pub(crate) trait Retryable {
    fn is_retryable(&self) -> bool;
    fn retry_delay(&self) -> Option<Duration> {
        None
    }
}

/// Cap on per-attempt backoff so exponential growth doesn't run away past a
/// few attempts. Matches claude-code's `maxDelayMs` default.
const MAX_RETRY_DELAY: Duration = Duration::from_millis(32_000);

/// Exponential backoff `base_delay * 2^attempt`, clamped at 32 s, extended by
/// additive jitter in `[0, 25%]` of the clamped value. A `server_hint`
/// bypasses the cap and jitter: the server is explicit about what it wants.
pub(crate) struct ExponentialRetry {
    pub base_delay: Duration,
    pub max_attempts: u32,
}

impl Retry for ExponentialRetry {
    fn max_attempts(&self) -> u32 {
        self.max_attempts
    }

    fn delay(&self, attempt: u32, server_hint: Option<Duration>) -> Duration {
        if let Some(hint) = server_hint {
            return hint;
        }

        let base_ms = self.base_delay.as_millis() as u64;
        let exponential_ms = base_ms
            .saturating_mul(1u64 << attempt.min(31))
            .min(MAX_RETRY_DELAY.as_millis() as u64);
        let jitter_range = exponential_ms / 4;

        if jitter_range == 0 {
            return Duration::from_millis(exponential_ms);
        }

        let entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        let jitter_offset = entropy % jitter_range;

        Duration::from_millis(exponential_ms.saturating_add(jitter_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(base_ms: u64) -> ExponentialRetry {
        ExponentialRetry {
            base_delay: Duration::from_millis(base_ms),
            max_attempts: 10,
        }
    }

    #[test]
    fn exponential_backoff() {
        let policy = policy(1000);
        for attempt in 0..3 {
            let delay = policy.delay(attempt, None);
            let expected_base_ms = 1000u64 * (1u64 << attempt);
            let jitter_range_ms = expected_base_ms / 4;
            let delay_ms = delay.as_millis() as u64;
            assert!(delay_ms >= expected_base_ms);
            assert!(delay_ms <= expected_base_ms + jitter_range_ms);
        }
    }

    #[test]
    fn respects_retry_delay() {
        let delay = policy(1000).delay(0, Some(Duration::from_millis(5000)));
        assert_eq!(delay, Duration::from_millis(5000));
    }

    #[test]
    fn caps_at_max_delay() {
        let delay = policy(1000).delay(20, None);
        let max_ms = MAX_RETRY_DELAY.as_millis() as u64;
        let jitter_range_ms = max_ms / 4;
        let delay_ms = delay.as_millis() as u64;
        assert!(delay_ms >= max_ms);
        assert!(delay_ms <= max_ms + jitter_range_ms);
    }

    #[test]
    fn saturates_instead_of_overflow() {
        let _delay = policy(u64::MAX).delay(10, None);
    }
}
