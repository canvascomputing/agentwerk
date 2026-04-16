//! Execute multiple agents with controlled parallelism.
//!
//! Each agent is a fully configured `AgentBuilder` with its own provider, prompts,
//! and tools. The pipeline controls how many run concurrently via a semaphore.
//! Results are returned in push order. Individual failures do not abort the pipeline.

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::error::Result;
use super::builder::AgentBuilder;
use super::output::AgentOutput;

const DEFAULT_BATCH_SIZE: usize = 10;

/// Execute multiple agents with controlled parallelism.
pub struct Pipeline {
    batch_size: usize,
    agents: Vec<AgentBuilder>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            batch_size: DEFAULT_BATCH_SIZE,
            agents: Vec::new(),
        }
    }

    /// Maximum number of agents running concurrently.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Add a configured agent to the pipeline.
    pub fn push(mut self, agent: AgentBuilder) -> Self {
        self.agents.push(agent);
        self
    }

    /// Execute all queued agents and return results in push order.
    pub async fn run(self) -> Vec<Result<AgentOutput>> {
        let agent_count = self.agents.len();
        if agent_count == 0 {
            return Vec::new();
        }

        let semaphore = Arc::new(Semaphore::new(self.batch_size));
        let mut set = JoinSet::new();

        for (index, builder) in self.agents.into_iter().enumerate() {
            let permit = semaphore.clone().acquire_owned().await.unwrap();

            set.spawn(async move {
                let result = builder.run().await;
                drop(permit);
                (index, result)
            });
        }

        let mut results: Vec<Option<Result<AgentOutput>>> = (0..agent_count).map(|_| None).collect();
        while let Some(join_result) = set.join_next().await {
            let (index, result) = join_result.unwrap();
            results[index] = Some(result);
        }

        results.into_iter().map(|r| r.unwrap()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use crate::testutil::{MockProvider, tool_response, text_response};
    use crate::tools::{ToolBuilder, ToolResult};

    fn agent_with_response(text: &str) -> AgentBuilder {
        let provider = Arc::new(MockProvider::text(text));
        AgentBuilder::new()
            .name("test")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(provider)
    }

    #[tokio::test]
    async fn pipeline_executes_in_order() {
        let pipeline = Pipeline::new()
            .batch_size(2)
            .push(agent_with_response("first"))
            .push(agent_with_response("second"))
            .push(agent_with_response("third"));

        let results = pipeline.run().await;

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().response_raw, "first");
        assert_eq!(results[1].as_ref().unwrap().response_raw, "second");
        assert_eq!(results[2].as_ref().unwrap().response_raw, "third");
    }

    #[tokio::test]
    async fn pipeline_individual_failures() {
        let pipeline = Pipeline::new()
            .batch_size(2)
            .push(agent_with_response("ok"))
            .push({
                let provider = Arc::new(MockProvider::new(vec![]));
                AgentBuilder::new()
                    .name("fail")
                    .model("mock")
                    .identity_prompt("")
                    .instruction_prompt("go")
                    .provider(provider)
            })
            .push(agent_with_response("also ok"));

        let results = pipeline.run().await;

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err());
        assert!(results[2].is_ok());
    }

    #[tokio::test]
    async fn pipeline_empty() {
        let pipeline = Pipeline::new();
        let results = pipeline.run().await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn pipeline_runs_concurrently() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let mut pipeline = Pipeline::new().batch_size(3);

        for _ in 0..6 {
            let running = running.clone();
            let max_concurrent = max_concurrent.clone();

            let slow_tool = ToolBuilder::new("slow", "Simulates slow work")
                .schema(serde_json::json!({"type": "object", "properties": {}}))
                .handler(move |_, _| {
                    let running = running.clone();
                    let max_concurrent = max_concurrent.clone();
                    Box::pin(async move {
                        let current = running.fetch_add(1, Ordering::SeqCst) + 1;
                        max_concurrent.fetch_max(current, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        running.fetch_sub(1, Ordering::SeqCst);
                        Ok(ToolResult::success("done"))
                    })
                })
                .build();

            let provider = Arc::new(MockProvider::new(vec![
                tool_response("slow", "c1", serde_json::json!({})),
                text_response("finished"),
            ]));

            pipeline = pipeline.push(
                AgentBuilder::new()
                    .name("worker")
                    .model("mock")
                    .identity_prompt("")
                    .instruction_prompt("go")
                    .tool(slow_tool)
                    .provider(provider)
            );
        }

        let results = pipeline.run().await;

        assert_eq!(results.len(), 6);
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(
            max_concurrent.load(Ordering::SeqCst) >= 3,
            "Expected at least 3 concurrent agents, got {}",
            max_concurrent.load(Ordering::SeqCst)
        );
    }
}
