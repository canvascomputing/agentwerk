//! Errors raised by the internal on-disk stores (todo lists, session transcripts).

use std::fmt;

/// Failures from the todo list and session store.
#[derive(Debug)]
pub enum PersistenceError {
    /// An item id that was expected to exist could not be found.
    TodoItemNotFound(String),
    /// An item write was attempted on an item already marked completed.
    TodoItemAlreadyCompleted(String),
    /// An item is blocked by another item that has not yet completed.
    TodoItemBlocked { item_id: String, blocker_id: String },
    /// Acquiring the on-disk lock failed after the configured retry budget.
    LockFailed { attempts: u32 },
    /// Underlying filesystem I/O failed.
    IoFailed(std::io::Error),
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PersistenceError::TodoItemNotFound(id) => write!(f, "Item {id} not found"),
            PersistenceError::TodoItemAlreadyCompleted(id) => {
                write!(f, "Item {id} already completed")
            }
            PersistenceError::TodoItemBlocked {
                item_id,
                blocker_id,
            } => write!(f, "Item {item_id} blocked by unfinished item {blocker_id}"),
            PersistenceError::LockFailed { attempts } => {
                write!(f, "Failed to acquire lock after {attempts} attempts")
            }
            PersistenceError::IoFailed(err) => write!(f, "Persistence I/O failed: {err}"),
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            PersistenceError::IoFailed(err) => Some(err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(err: std::io::Error) -> Self {
        PersistenceError::IoFailed(err)
    }
}

impl From<serde_json::Error> for PersistenceError {
    fn from(err: serde_json::Error) -> Self {
        PersistenceError::IoFailed(std::io::Error::new(std::io::ErrorKind::InvalidData, err))
    }
}

/// Result alias used inside `persistence/`. Converts to the crate-level
/// `Result` via `?` at the call boundary.
pub(crate) type PersistenceResult<T> = std::result::Result<T, PersistenceError>;
