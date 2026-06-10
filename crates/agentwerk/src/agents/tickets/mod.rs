//! Ticket queue and run orchestration. Exposes the value types
//! ([`Ticket`], [`Status`], [`Reply`], [`ReplyContent`], [`TicketError`])
//! and the [`TicketSystem`] orchestrator that owns the shared queue,
//! registered agents, policies, interrupt signal, and run stats.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::event::PolicyKind;

use super::policy::Policies;
use super::stats::Stats;

mod error;
mod reply;
mod store;
mod ticket;
mod ticket_system;

#[cfg(test)]
pub(super) mod test_util;

pub use error::TicketError;
pub use reply::{Reply, ReplyContent};
pub use ticket::{Status, Ticket};
pub use ticket_system::TicketSystem;

pub(crate) use ticket::Replies;

/// Whether the run-wide policies have been exceeded by the current
/// stats reading. Used by the `finish` watcher and by the per-agent
/// loop's pre-claim check.
pub(crate) fn policy_violated(policies: &Policies, stats: &Stats) -> bool {
    policy_violated_kind(policies, stats).is_some()
}

/// Same as [`policy_violated`] but returns which policy tripped and its
/// configured limit, for the `PolicyViolated` event.
pub(crate) fn policy_violated_kind(
    policies: &Policies,
    stats: &Stats,
) -> Option<(PolicyKind, u64)> {
    if let Some(limit) = policies.max_turns {
        if stats.turns() >= u64::from(limit) {
            return Some((PolicyKind::Turns, u64::from(limit)));
        }
    }
    if let Some(limit) = policies.max_input_tokens {
        if stats.input_tokens() >= limit {
            return Some((PolicyKind::InputTokens, limit));
        }
    }
    if let Some(limit) = policies.max_output_tokens {
        if stats.output_tokens() >= limit {
            return Some((PolicyKind::OutputTokens, limit));
        }
    }
    None
}

pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Trailing numeric part of a `TICKET-N` key. Falls back to `u32::MAX`
/// so malformed keys sort last and tie-break stably.
pub(crate) fn numeric_id(key: &str) -> u32 {
    key.rsplit('-')
        .next()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}
