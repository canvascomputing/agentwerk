//! Errors raised by ticket-store mutations.

use std::fmt;

use super::ticket::Status;

/// Errors raised by ticket-store mutations.
#[derive(Debug)]
pub enum TicketError {
    /// No ticket exists at `key`.
    TicketMissing { key: String },
    /// Status transition `from → to` is not allowed.
    TransitionRejected { from: Status, to: Status },
}

impl fmt::Display for TicketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TicketMissing { key } => write!(f, "Ticket {key} not found"),
            Self::TransitionRejected { from, to } => {
                write!(f, "Illegal transition {from:?} -> {to:?}")
            }
        }
    }
}

impl std::error::Error for TicketError {}
