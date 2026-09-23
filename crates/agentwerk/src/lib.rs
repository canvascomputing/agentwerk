#![warn(missing_docs)]

//! Run agentic workflows where many agents work in parallel on a shared
//! [`struct@Werk`]. An [`struct@Agent`] picks up tasks from a Werk,
//! calls the LLM provider, runs the tools it requests, and writes results
//! back. Tasks are assigned to agents by label; the Werk
//! handles concurrency, automatic context compaction, schema validation,
//! retries, and limits.
//!
//! # Quick start
//!
//! ```no_run
//! use agentwerk::Agent;
//! use agentwerk::tools::{GrepTool, ReadFileTool};
//!
//! # async fn run() {
//! let agent = Agent::from_env()
//!     .role("You are a Rust developer who explores source files to answer questions.")
//!     .tool(ReadFileTool)
//!     .tool(GrepTool);
//!
//! let task = agent.add_task("Find every `pub trait` defined under src/ and explain each in one sentence.");
//! let result = agent.finish_task(task).await.unwrap();
//!
//! println!("{}", result.as_str().unwrap_or_default());
//! # }
//! ```
//!
//! # Many agents working together
//!
//! ```no_run
//! use agentwerk::{Agent, Task, Werk};
//! use agentwerk::tools::FetchTool;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let werk = Werk(".agentwerk")?;
//!
//! for _ in 0..4 {
//!     werk.add_agent(
//!         Agent::from_env()
//!             .label("research")
//!             .tool(FetchTool),
//!     );
//! }
//!
//! for url in [
//!     "https://canvascomputing.org",
//!     "https://canvascomputing.org/about",
//!     "https://canvascomputing.org/products",
//!     "https://canvascomputing.org/blog",
//! ] {
//!     werk.add_task(Task(format!("Summarize {url}")).label("research"));
//! }
//!
//! werk.finish().await;
//!
//! for task in werk.get_tasks() {
//!     if let Some(result) = task.get_result() {
//!         println!("{}: {}", task.get_id(), result);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # Main types
//!
//! - [`struct@Agent`]: picks up tasks and produces results.
//! - [`struct@Condition`]: releases agents and tasks when AQL matches a task or event.
//! - [`struct@Werk`]: stores tasks and runs agents.
//! - [`struct@Task`]: defines work with an optional label and schema.
//! - [`struct@Query`]: a reusable AQL selection over tasks, events, or joined task-event pairs.
//! - [`struct@Knowledge`]: durable memory the agent shares across tasks and other agents.
//! - [`struct@Event`]: records requests, tool usage, failures, and other activity.
//! - [`tools`]: the built-in tools agents call, for files, search, commands, web, knowledge, and tasks.
//!
//! # Primary constructors
//!
//! A type and its callable constructor share one import.
//!
//! ```no_run
//! use std::sync::Arc;
//! use agentwerk::{Agent, Condition, Event, Knowledge, Query, Schema, Task, Werk};
//! use agentwerk::providers::{Anthropic, LiteLlm, Mistral, Model, OpenAi, Provider};
//! use agentwerk::tools::Tool;
//! use serde_json::json;
//!
//! let _: Agent = Agent();
//! let _: Arc<Werk> = Werk("./session")?;
//! let _: Arc<Knowledge> = Knowledge("./notes")?;
//! let _: Tool = Tool("search");
//! let _: Task = Task("inspect");
//! let _: Condition = Condition("event.name = ready");
//! let _: Query = Query("research")?;
//! let _: Model = Model("mock");
//! let _: Schema = Schema(json!({ "type": "object" }))?;
//! let _: Event = Event("ready");
//! let _: Anthropic = Anthropic("key");
//! let _: OpenAi = OpenAi("key");
//! let _: Mistral = Mistral("key");
//! let _: LiteLlm = LiteLlm("");
//! let _: Provider = Provider(Anthropic("key"));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod agents;
pub mod codegrep;
pub mod event;
pub(crate) mod persistence;
pub(crate) mod prompts;
pub mod providers;
pub mod schemas;
pub mod tools;

#[cfg(test)]
pub(crate) mod test_util;

pub use agents::Agent;
pub use agents::Condition;
pub use agents::Query;
pub use agents::Reply;
pub use agents::Status;
pub use agents::Task;
pub use agents::Werk;

pub use agents::Knowledge;
pub use agents::Policy;
pub use agents::PolicyViolation;

pub use schemas::Schema;

pub use agents::tasks::FinishReason;
pub use event::Event;

#[cfg(test)]
mod callable_constructor_tests {
    use super::{Agent, Condition, Event, Knowledge, Query, Schema, Task, Werk};
    use crate::providers::Model;
    use crate::tools::Tool;
    use serde_json::json;

    #[test]
    fn callable_and_associated_constructors_match_public_state() {
        let callable_agent = Agent();
        assert!(callable_agent.label.is_none());
        assert!(!callable_agent.interactive);

        let dir = crate::test_util::TempDir::new().unwrap();
        let werk = Werk(dir.path()).unwrap();
        assert_eq!(werk.get_dir(), dir.path());
        assert!(werk.get_tasks().is_empty());

        let knowledge = Knowledge(dir.path().join("knowledge")).unwrap();
        assert!(knowledge.get_index().is_empty());

        let tool = Tool("search").description("Search indexed documents");
        assert_eq!(tool.get_name(), "search");
        assert_eq!(tool.get_description(), "Search indexed documents");

        let callable_task = Task(json!({ "work": "inspect" }));
        let associated_task = Task::new(json!({ "work": "inspect" }));
        assert_eq!(callable_task.get_task(), associated_task.get_task());
        assert_eq!(callable_task.get_status(), associated_task.get_status());

        let callable_model = Model("gpt-5");
        let associated_model = Model::new("gpt-5");
        assert_eq!(callable_model.get_name(), associated_model.get_name());
        assert_eq!(
            callable_model.get_context_window(),
            associated_model.get_context_window()
        );

        let document = json!({ "type": "object" });
        let callable_schema = Schema(document.clone()).unwrap();
        let associated_schema = Schema::new(document).unwrap();
        assert_eq!(
            callable_schema.get_raw_schema(),
            associated_schema.get_raw_schema()
        );

        let callable_event = Event("ready");
        let associated_event = Event::new("ready");
        assert_eq!(callable_event.get_name(), associated_event.get_name());
        assert_eq!(callable_event.get_data(), associated_event.get_data());

        let callable_condition = Condition("event.name = ready");
        let associated_condition = Condition::new("event.name = ready");
        assert_eq!(
            callable_condition.max_triggers,
            associated_condition.max_triggers
        );
        assert_eq!(
            callable_condition.agents.len(),
            associated_condition.agents.len()
        );
        assert_eq!(
            callable_condition.tasks.len(),
            associated_condition.tasks.len()
        );

        let callable_werk = Werk(dir.path().join("callable")).unwrap();
        let associated_werk = Werk(dir.path().join("associated")).unwrap();
        assert_eq!(
            callable_werk.add_condition(callable_condition),
            associated_werk.add_condition(associated_condition)
        );
    }

    #[test]
    fn fallible_callable_constructors_preserve_errors() {
        assert_eq!(Query(" ").unwrap_err(), Query::new(" ").unwrap_err());

        let document = json!({ "type": "array" });
        assert_eq!(
            Schema(document.clone()).unwrap_err().message,
            Schema::new(document).unwrap_err().message
        );
    }
}
