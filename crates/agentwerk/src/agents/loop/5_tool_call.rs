//! Tool execution: dispatches tool calls, offloads large outputs, and enforces schema-retry budget.

use std::collections::HashMap;
use std::path::PathBuf;

use crate::event::{EventKind, ToolFailureKind};
use crate::providers::ContentBlock;
use crate::tools::{ToolContext, ToolError, TICKET_FINISHER_TOOLS};
use crate::prompts::{retry_directive, schema_retry_detail};

use super::turn::TicketScope;
use super::Reply;
use super::Action;

pub(super) async fn run(scope: &mut TicketScope<'_>, reply: Reply) -> Action<()> {
    let max_schema_retries = scope.policies.max_schema_retries.unwrap_or(u32::MAX);

    match reply {
        Reply::TextOnly => {
            let has_schema = scope
                .ticket_system
                .get_ticket(&scope.key)
                .map(|t| t.schema.is_some())
                .unwrap_or(false);
            if !has_schema {
                return Action::Proceed(());
            }
            let registered: Vec<&str> = TICKET_FINISHER_TOOLS
                .iter()
                .copied()
                .filter(|&n| scope.agent.tool_registry().get(n).is_some())
                .collect();
            if registered.is_empty() {
                return Action::Proceed(());
            }
            scope.consecutive_schema_failures =
                scope.consecutive_schema_failures.saturating_add(1);
            let detail = format!(
                "Your reply was text-only. Call `{}` to finish the ticket \
                 — your work is not recorded until you do.",
                registered.join("` or `")
            );
            scope.emit(EventKind::SchemaRetried {
                attempt: scope.consecutive_schema_failures,
                max_attempts: max_schema_retries,
                message: detail.clone(),
            });
            scope
                .ticket_system
                .add_comment(&scope.key, crate::agents::tickets::Comment::user_text(retry_directive(&detail)));
            if scope.consecutive_schema_failures >= max_schema_retries {
                scope.fail_ticket_schema_exhausted();
                return Action::Replay;
            }
            Action::Proceed(())
        }
        Reply::Calls(calls) => {
            for call in &calls {
                scope.emit(EventKind::ToolCallStarted {
                    tool_name: call.name.clone(),
                    call_id: call.id.clone(),
                    input: call.input.clone(),
                });
            }
            let tool_context = ToolContext::new(scope.agent.dir_or_default())
                .interrupt_signal(std::sync::Arc::clone(&scope.interrupt_signal))
                .registry(std::sync::Arc::new(scope.agent.tool_registry().clone()))
                .ticket_system(std::sync::Arc::clone(scope.ticket_system))
                .agent_name(scope.agent.get_name().to_string())
                .ticket_key(scope.key.clone())
                .knowledge(scope.agent.knowledge_or_default());
            let outcomes = scope
                .agent
                .tool_registry()
                .execute(&calls, &tool_context)
                .await;

            let mut schema_failure_message: Option<String> = None;
            for (block, tool_result, _path) in &outcomes {
                let ContentBlock::ToolResult { tool_use_id, .. } = block else {
                    continue;
                };
                let call = calls.iter().find(|c| &c.id == tool_use_id);
                let tool_name = call.map(|c| c.name.clone()).unwrap_or_default();
                match tool_result {
                    Ok(output) => {
                        if call
                            .is_some_and(|c| TICKET_FINISHER_TOOLS.contains(&c.name.as_str()))
                        {
                            scope.consecutive_schema_failures = 0;
                        }
                        scope.emit(EventKind::ToolCallFinished {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            output: output.clone(),
                        });
                    }
                    Err(err) => {
                        if matches!(err, ToolError::SchemaValidationFailed { .. }) {
                            scope.consecutive_schema_failures =
                                scope.consecutive_schema_failures.saturating_add(1);
                            if schema_failure_message.is_none() {
                                schema_failure_message = Some(err.message());
                            }
                        }
                        let failure_kind = match err {
                            ToolError::ToolNotFound { .. } => ToolFailureKind::ToolNotFound,
                            ToolError::ExecutionFailed { .. } => ToolFailureKind::ExecutionFailed,
                            ToolError::SchemaValidationFailed { .. } => ToolFailureKind::SchemaValidationFailed,
                        };
                        scope.emit(EventKind::ToolCallFailed {
                            tool_name,
                            call_id: tool_use_id.clone(),
                            message: err.message(),
                            kind: failure_kind,
                        });
                    }
                }
            }

            let mut paths: HashMap<String, PathBuf> = HashMap::new();
            let mut blocks: Vec<ContentBlock> = Vec::with_capacity(outcomes.len());
            for (block, _, path) in outcomes {
                if let (ContentBlock::ToolResult { tool_use_id, .. }, Some(p)) = (&block, path) {
                    paths.insert(tool_use_id.clone(), p);
                }
                blocks.push(block);
            }
            if let Some(validator_message) = &schema_failure_message {
                let schema_detail = schema_retry_detail(validator_message);
                scope.emit(EventKind::SchemaRetried {
                    attempt: scope.consecutive_schema_failures,
                    max_attempts: max_schema_retries,
                    message: schema_detail.clone(),
                });
                blocks.push(ContentBlock::Text {
                    text: retry_directive(&schema_detail),
                });
            }
            scope
                .ticket_system
                .add_comment(&scope.key, crate::agents::tickets::Comment::user(&blocks, &paths));

            scope.record_tool_calls(calls.len());

            if scope.consecutive_schema_failures >= max_schema_retries {
                scope.fail_ticket_schema_exhausted();
                return Action::Replay;
            }
            Action::Proceed(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::agents::r#loop::test_util::*;
    use crate::agents::tickets::Status;
    use crate::event::{EventKind, PolicyKind};
    use crate::schemas::Schema;

    // Text-only replies

    #[tokio::test]
    async fn text_reply_no_schema_exits_cleanly() {
        let provider = MockProvider::with_results(vec![Ok(text_response("Hello!"))]);
        let (events, provider, ticket) = run_one(provider, 3, 10, None).await;

        assert_eq!(provider.requests(), 1);
        assert_eq!(schema_retries_in(&events).len(), 0);
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(failed, 0);
        assert_eq!(ticket.status, Status::InProgress);
    }

    #[tokio::test]
    async fn text_reply_with_schema_exhausts_retries_and_fails() {
        let provider = MockProvider::with_results(vec![
            Ok(text_response("a")),
            Ok(text_response("b")),
            Ok(text_response("c")),
        ]);
        let (events, _, ticket) = run_one(provider, 3, 2, Some(schema_for_partial_sum())).await;

        let retries = schema_retries_in(&events);
        assert_eq!(retries.len(), 2);
        let policy_violated = events.iter().any(|e| {
            matches!(
                &e.kind,
                EventKind::PolicyViolated {
                    kind: PolicyKind::MaxSchemaRetries,
                    limit: 2,
                },
            )
        });
        assert!(policy_violated, "expected MaxSchemaRetries PolicyViolated");
        assert_eq!(ticket.status, Status::Failed);
    }

    #[tokio::test]
    async fn text_reply_with_schema_retries_then_recovers() {
        let provider = MockProvider::with_results(vec![
            Ok(text_response("Hello!")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 1}))),
        ]);
        let (events, provider, ticket) =
            run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        assert_eq!(provider.requests(), 2);
        let retries = schema_retries_in(&events);
        assert_eq!(retries.len(), 1);
        assert!(retries[0].2.contains("finish_ticket"));
        let done = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFinished { .. }))
            .count();
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(done, 1);
        assert_eq!(failed, 0);
        assert_eq!(ticket.status, Status::Finished);
    }

    #[tokio::test]
    async fn write_result_settles_ticket_done_with_valid_json() {
        let provider = MockProvider::with_results(vec![Ok(write_result_value(
            serde_json::json!({"partial_sum": 42}),
        ))]);
        let (events, provider, ticket) =
            run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        assert_eq!(provider.requests(), 1);
        let done = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFinished { .. }))
            .count();
        let failed = events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::TicketFailed { .. }))
            .count();
        assert_eq!(done, 1);
        assert_eq!(failed, 0);
        assert_eq!(ticket.status, Status::Finished);
        assert_eq!(ticket.result.as_ref().unwrap()["partial_sum"], 42);
    }

    // Schema retries

    #[tokio::test]
    async fn schema_violation_emits_schema_retried_with_attempt_numbers() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("not json")),
            Ok(write_result_response("not json again")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 42}))),
        ]);
        let (events, _, ticket) = run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;

        let schema_retries = schema_retries_in(&events);
        let attempts: Vec<u32> = schema_retries.iter().map(|(a, ..)| *a).collect();
        assert_eq!(attempts, vec![1, 2]);
        for (_, max_attempts, _) in &schema_retries {
            assert_eq!(*max_attempts, 10);
        }
        assert_eq!(ticket.status, Status::Finished);
    }

    #[tokio::test]
    async fn schema_retry_appends_directive_to_user_message() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("not json")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 1}))),
        ]);
        let (events, _, _) = run_one(provider, 3, 10, Some(schema_for_partial_sum())).await;
        let schema_retries = schema_retries_in(&events);
        assert_eq!(schema_retries.len(), 1);
        assert!(
            !schema_retries[0].2.is_empty(),
            "schema-retry message must carry validator detail"
        );
    }

    #[tokio::test]
    async fn schema_retry_exhausted_emits_policy_violated_and_force_fails_ticket() {
        let provider = MockProvider::with_results(vec![
            Ok(write_result_response("nope")),
            Ok(write_result_response("still nope")),
            Ok(write_result_response("never")),
        ]);
        let (events, _, ticket) = run_one(provider, 3, 2, Some(schema_for_partial_sum())).await;

        let policy_violated = events.iter().any(|e| {
            matches!(
                &e.kind,
                EventKind::PolicyViolated {
                    kind: PolicyKind::MaxSchemaRetries,
                    limit: 2,
                },
            )
        });
        assert!(policy_violated, "expected MaxSchemaRetries PolicyViolated");
        assert_eq!(ticket.status, Status::Failed);
    }

    fn schema_for_partial_sum() -> Schema {
        Schema::parse(serde_json::json!({
            "type": "object",
            "properties": {
                "partial_sum": { "type": "integer" }
            },
            "required": ["partial_sum"]
        }))
        .expect("valid schema")
    }
}
