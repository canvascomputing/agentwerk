use crate::event::Event;

use super::query::Origin;
use super::{Agent, Query, QueryError, Task};

/// Release agents and tasks when an AQL query matches a task or event.
///
/// A condition belongs to the [`Werk`](crate::Werk) it is added to. It fires
/// at most once per run and is neither persisted nor restored with a session.
///
/// ```no_run
/// use agentwerk::{Agent, Condition, Task, Werk};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let werk = Werk::new();
/// let id = werk.add_condition(
///     Condition::new("task.label = draft AND task.status = finished")?
///         .add_agent(Agent::from_env().label("edit"))
///         .add_task(Task::labeled("edit", "Edit the completed draft.")),
/// );
/// assert_eq!(id, "condition-1");
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Condition {
    pub(crate) query: Query,
    pub(crate) agents: Vec<Agent>,
    pub(crate) tasks: Vec<Task>,
    pub(crate) fired: bool,
}

impl Condition {
    /// Compile `aql` as the query that releases this condition's work.
    pub fn new(aql: &str) -> Result<Self, QueryError> {
        Ok(Self {
            query: Query::new(aql)?,
            agents: Vec::new(),
            tasks: Vec::new(),
            fired: false,
        })
    }

    /// Add an agent to activate when the condition matches.
    pub fn add_agent(mut self, agent: Agent) -> Self {
        self.agents.push(agent);
        self
    }

    /// Add a task to create when the condition matches.
    pub fn add_task(mut self, task: impl Into<Task>) -> Self {
        self.tasks.push(task.into());
        self
    }

    pub(crate) fn matches(&self, task: Option<&Task>, event: &Event) -> bool {
        match self.query.origin() {
            Origin::Task => task.is_some_and(|task| self.query.matches_task(task)),
            Origin::Event => self.query.matches_event(event),
            Origin::Joined => task.is_some_and(|task| self.query.matches_joined(task, event)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_valid_aql() {
        assert!(Condition::new("task.label = draft").is_ok());
    }

    #[test]
    fn rejects_invalid_aql() {
        assert!(Condition::new("label = draft").is_err());
    }
}
