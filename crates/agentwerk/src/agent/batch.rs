//! Run many agents in parallel with a fixed concurrency cap. The one thing a caller gets beyond raw `Agent::run` is backpressure on the provider.

use futures_util::stream::{self, Stream, StreamExt};

use crate::error::Result;

use super::agent::Agent;
use super::output::AgentOutput;

/// Run every `agent` concurrently, capped at `concurrency` simultaneous runs.
/// Yields results in completion order. Correlate each result to its input via
/// [`AgentOutput::name`].
pub fn batch<I>(agents: I, concurrency: usize) -> impl Stream<Item = Result<AgentOutput>>
where
    I: IntoIterator<Item = Agent>,
{
    stream::iter(agents.into_iter())
        .map(|agent| async move { agent.run().await })
        .buffer_unordered(concurrency.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{text_response, tool_response, MockProvider};
    use crate::tools::{Tool, ToolResult};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    fn agent_with_response(name: &str, text: &str) -> Agent {
        Agent::new()
            .name(name)
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(Arc::new(MockProvider::text(text)))
    }

    fn agent_with_delay(name: &str, delay_ms: u64, text: &str) -> Agent {
        let slow_tool = Tool::new("slow", "simulates work")
            .schema(serde_json::json!({"type": "object", "properties": {}}))
            .handler(move |_, _| {
                Box::pin(async move {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    Ok(ToolResult::success("done"))
                })
            });

        let provider = Arc::new(MockProvider::new(vec![
            tool_response("slow", "c1", serde_json::json!({})),
            text_response(text),
        ]));

        Agent::new()
            .name(name)
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .tool(slow_tool)
            .provider(provider)
    }

    #[tokio::test]
    async fn empty_iterator_yields_empty_stream() {
        let results: Vec<_> = batch(Vec::<Agent>::new(), 4).collect().await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn output_carries_agent_name() {
        let results: Vec<_> = batch(
            ["a", "b", "c"].iter().map(|n| agent_with_response(n, "ok")),
            4,
        )
        .collect()
        .await;
        assert_eq!(results.len(), 3);
        let mut names: Vec<String> = results
            .iter()
            .map(|r| r.as_ref().unwrap().name.clone())
            .collect();
        names.sort();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn failures_do_not_block_other_jobs() {
        let failing = Agent::new()
            .name("fail")
            .model("mock")
            .identity_prompt("")
            .instruction_prompt("go")
            .provider(Arc::new(MockProvider::new(vec![])));

        let agents = vec![
            agent_with_response("ok1", "first"),
            failing,
            agent_with_response("ok2", "second"),
        ];
        let results: Vec<_> = batch(agents, 2).collect().await;
        assert_eq!(results.len(), 3);
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 2);
        assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    }

    #[tokio::test]
    async fn concurrency_cap_bounds_parallelism() {
        let running = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let make = |i: usize| {
            let r = running.clone();
            let m = max_concurrent.clone();
            let slow_tool = Tool::new("slow", "work")
                .schema(serde_json::json!({"type": "object", "properties": {}}))
                .handler(move |_, _| {
                    let r = r.clone();
                    let m = m.clone();
                    Box::pin(async move {
                        let cur = r.fetch_add(1, Ordering::SeqCst) + 1;
                        m.fetch_max(cur, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        r.fetch_sub(1, Ordering::SeqCst);
                        Ok(ToolResult::success("done"))
                    })
                });
            Agent::new()
                .name(&format!("w{i}"))
                .model("mock")
                .identity_prompt("")
                .instruction_prompt("go")
                .tool(slow_tool)
                .provider(Arc::new(MockProvider::new(vec![
                    tool_response("slow", "c1", serde_json::json!({})),
                    text_response("finished"),
                ])))
        };

        let results: Vec<_> = batch((0..10).map(make), 3).collect().await;
        assert_eq!(results.len(), 10);
        assert!(results.iter().all(|r| r.is_ok()));
        let peak = max_concurrent.load(Ordering::SeqCst);
        assert!(peak <= 3, "peak concurrency {peak} exceeded cap of 3");
        assert!(
            peak >= 2,
            "peak concurrency {peak} never reached meaningful overlap"
        );
    }

    #[tokio::test]
    async fn concurrency_scales_throughput() {
        let start = tokio::time::Instant::now();
        let seq: Vec<_> = batch(
            (0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))),
            1,
        )
        .collect()
        .await;
        let seq_elapsed = start.elapsed();

        let start = tokio::time::Instant::now();
        let par: Vec<_> = batch(
            (0..10).map(|i| agent_with_delay("w", 30, &format!("r{i}"))),
            10,
        )
        .collect()
        .await;
        let par_elapsed = start.elapsed();

        assert_eq!(seq.len(), 10);
        assert_eq!(par.len(), 10);
        assert!(
            seq_elapsed > par_elapsed * 3,
            "sequential ({seq_elapsed:?}) should dwarf parallel ({par_elapsed:?})",
        );
    }

    #[tokio::test]
    async fn high_throughput_smoke() {
        let results: Vec<_> = batch(
            (0..500).map(|i| agent_with_response("w", &format!("r{i}"))),
            50,
        )
        .collect()
        .await;
        assert_eq!(results.len(), 500);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
