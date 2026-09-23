use crate::event::Event;

use super::query::Origin;
use super::{Agent, Query, Task};

/// Release agents and tasks when an AQL query matches a task or event.
///
/// A condition belongs to the [`Werk`](struct@crate::Werk) it is added to. It fires
/// at most once per run by default and is neither persisted nor restored with
/// a session.
///
/// ```no_run
/// use agentwerk::{Agent, Condition, Task, Werk};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let werk = Werk(".agentwerk")?;
/// let id = werk.add_condition(
///     Condition("task.label = draft AND task.status = finished")
///         .agent(Agent::from_env().label("edit"))
///         .task(Task("Edit the completed draft.").label("edit")),
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
    pub(crate) max_triggers: Option<usize>,
    pub(crate) remaining_triggers: Option<usize>,
}

impl Condition {
    /// Create a condition released by `query`.
    ///
    /// A string literal that does not compile as AQL panics. Build a
    /// [`struct@Query`] first when the string is only known at run time.
    pub fn new(query: impl Into<Query>) -> Self {
        Self {
            query: query.into(),
            agents: Vec::new(),
            tasks: Vec::new(),
            max_triggers: Some(1),
            remaining_triggers: Some(1),
        }
    }

    /// Set how many times this condition may activate per run.
    ///
    /// The default is one. `None` or zero allows every matching event. An
    /// unlimited condition whose actions emit another matching event can
    /// recurse without bound.
    pub fn times(mut self, times: impl Into<Option<usize>>) -> Self {
        let times = times.into().filter(|&times| times != 0);
        self.max_triggers = times;
        self.remaining_triggers = times;
        self
    }

    /// Add an agent to activate when the condition matches.
    pub fn agent(mut self, agent: Agent) -> Self {
        self.agents.push(agent);
        self
    }

    /// Add agents to activate when the condition matches.
    pub fn agents(mut self, agents: impl IntoIterator<Item = Agent>) -> Self {
        self.agents.extend(agents);
        self
    }

    /// Add a task to create when the condition matches.
    pub fn task(mut self, task: impl Into<Task>) -> Self {
        self.tasks.push(task.into());
        self
    }

    /// Add tasks to create when the condition matches.
    pub fn tasks<I, T>(mut self, tasks: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<Task>,
    {
        self.tasks.extend(tasks.into_iter().map(Into::into));
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
    fn accepts_borrowed_owned_and_compiled_queries() {
        let event = Event::new("ready");

        assert!(Condition::new("event.name = ready").matches(None, &event));
        assert!(Condition::new(String::from("event.name = ready")).matches(None, &event));
        assert!(Condition::new(Query::new("event.name = ready").unwrap()).matches(None, &event));
    }

    #[test]
    #[should_panic(expected = "invalid query")]
    fn an_invalid_query_literal_panics() {
        Condition::new("label = draft");
    }

    #[test]
    fn a_dynamic_invalid_query_returns_an_error_before_condition_construction() {
        fn condition(input: &str) -> Result<Condition, super::super::QueryError> {
            Ok(Condition::new(Query::new(input)?))
        }

        assert!(condition("label = draft").is_err());
    }

    #[test]
    fn singular_and_plural_builders_collect_actions() {
        let condition = Condition::new("event.name = ready")
            .agent(crate::Agent())
            .agents([crate::Agent(), crate::Agent()])
            .task("one")
            .tasks(["two", "three"]);

        assert_eq!(condition.agents.len(), 3);
        assert_eq!(condition.tasks.len(), 3);
    }

    #[test]
    fn times_sets_a_finite_or_unlimited_trigger_count() {
        let finite = Condition::new("event.name = ready").times(3);
        assert_eq!(finite.max_triggers, Some(3));
        assert_eq!(finite.remaining_triggers, Some(3));

        for unlimited in [
            Condition::new("event.name = ready").times(None),
            Condition::new("event.name = ready").times(0),
        ] {
            assert_eq!(unlimited.max_triggers, None);
            assert_eq!(unlimited.remaining_triggers, None);
        }
    }
}
