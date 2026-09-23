//! Defines agents and the execution loop they use to claim tasks.

pub mod agent;
pub(crate) mod compaction;
mod condition;
pub mod knowledge;
pub(crate) mod r#loop;
pub mod policy;
mod query;
pub(crate) mod retry;
pub(crate) mod stats;
pub mod tasks;

pub use agent::Agent;
pub use condition::Condition;
pub use knowledge::Knowledge;
pub use policy::{Policy, PolicyViolation};
pub use query::{Matcher, Query, QueryError};
pub use tasks::{Reply, Status, Task, TaskError, Werk};

/// Create an agent with no provider, model, or tools.
#[allow(non_snake_case)]
pub fn Agent() -> Agent {
    Agent::new()
}

/// Create a task carrying `task`.
#[allow(non_snake_case)]
pub fn Task<T: serde::Serialize>(task: T) -> Task {
    Task::new(task)
}

/// Create a condition released by `query`.
#[allow(non_snake_case)]
pub fn Condition(query: impl Into<Query>) -> Condition {
    Condition::new(query)
}

/// Compile an AQL string.
#[allow(non_snake_case)]
pub fn Query(query: &str) -> Result<Query, QueryError> {
    Query::new(query)
}
