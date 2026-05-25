//! Multi-agent loop driver. One tokio task per registered agent,
//! reading the shared `TicketSystem` through the upgraded
//! `Weak<TicketSystem>` stamped at `bind_agent`.

use std::time::Duration;

use crate::tools::ToolCall;

#[path = "1_main.rs"]
mod main;
#[path = "2_turn.rs"]
mod turn;
#[path = "3_compaction.rs"]
mod compaction;
#[path = "4_reply.rs"]
mod reply;
#[path = "5_tool_call.rs"]
mod tool_call;

pub(super) use self::main::run_main_loop;
use self::main::wait_for_signal;

const POLL_INTERVAL: Duration = Duration::from_millis(50);

enum Action<T> {
    Proceed(T),
    Replay,
    Stop,
}

enum Reply {
    Calls(Vec<ToolCall>),
    TextOnly,
}

#[cfg(test)]
pub(crate) mod test_util;

#[cfg(test)]
mod tests {
    use crate::agents::r#loop::test_util::*;
    use crate::agents::tickets::CommentContent;
    use crate::schemas::Schema;

    // Comment transcript

    #[tokio::test]
    async fn comments_capture_full_transcript() {
        let provider = MockProvider::with_results(vec![Ok(write_result_response("ok"))]);
        let (_, _, ticket) = run_one(provider, 3, 10, None).await;

        let comments = &ticket.comments;
        assert_eq!(comments.len(), 5, "got {comments:?}");

        assert_eq!(comments[0].author, "system");
        assert!(matches!(
            &comments[0].content[..],
            [CommentContent::Text(_)]
        ));

        assert_eq!(comments[1].author, "user");
        assert!(
            matches!(&comments[1].content[..], [CommentContent::Text(t)] if t.starts_with("## Context")),
            "second comment must be the auto-injected context prelude",
        );

        assert_eq!(comments[2].author, "user");
        assert!(
            matches!(&comments[2].content[..], [CommentContent::Text(t)] if t == "go"),
            "third comment must carry the task body",
        );

        assert_eq!(comments[3].author, "assistant");
        assert!(
            matches!(&comments[3].content[..], [CommentContent::ToolUse { name, .. }] if name == "finish_ticket"),
            "assistant comment must mirror the model's ToolUse block",
        );

        assert_eq!(comments[4].author, "user");
        assert!(
            matches!(
                &comments[4].content[..],
                [CommentContent::ToolResult { .. }]
            ),
            "tool-result comment must carry a ToolResult block",
        );

        for w in comments.windows(2) {
            assert!(
                w[0].created_at <= w[1].created_at,
                "comment timestamps must be monotonic",
            );
        }
    }

    #[tokio::test]
    async fn text_reply_with_schema_injects_directive_into_transcript() {
        let provider = MockProvider::with_results(vec![
            Ok(text_response("Hello!")),
            Ok(write_result_value(serde_json::json!({"partial_sum": 1}))),
        ]);
        let schema = Schema::parse(serde_json::json!({
            "type": "object",
            "properties": { "partial_sum": { "type": "integer" } },
            "required": ["partial_sum"]
        }))
        .expect("valid schema");
        let (_, _, ticket) = run_one(provider, 3, 10, Some(schema)).await;

        let comments = &ticket.comments;

        let first_assistant = comments
            .iter()
            .position(|c| {
                c.author == "assistant"
                    && matches!(&c.content[..], [CommentContent::Text(t)] if t == "Hello!")
            })
            .expect("expected the text-only assistant reply in the transcript");

        let directive = &comments[first_assistant + 1];
        assert_eq!(directive.author, "user");
        let directive_text = match &directive.content[..] {
            [CommentContent::Text(t)] => t,
            other => panic!("expected a single text block for the directive, got {other:?}"),
        };
        assert!(
            directive_text.contains("finish_ticket"),
            "directive must name the missing finisher: {directive_text}",
        );

        let second_assistant = comments
            .iter()
            .skip(first_assistant + 2)
            .find(|c| {
                c.author == "assistant"
                    && matches!(&c.content[..], [CommentContent::ToolUse { name, .. }] if name == "finish_ticket")
            });
        assert!(
            second_assistant.is_some(),
            "expected a recovering ToolUse assistant comment after the directive",
        );
    }

    #[tokio::test]
    async fn comments_after_compaction_keep_only_system_and_summary() {
        let provider = MockProvider::with_results(vec![
            Ok(text_response("turn 1")),
            Err(crate::providers::ProviderError::ContextWindowExceeded {
                message: "exceeded".into(),
            }),
            Ok(text_response_with_usage(
                "SUMMARY",
                crate::providers::types::TokenUsage::default(),
            )),
            Ok(write_result_response("ok")),
        ]);
        let (_, _, ticket) = run_one(provider, 0, 10, Some(string_schema())).await;

        let comments = &ticket.comments;

        assert_eq!(comments[0].author, "system");

        let summary_idx = comments
            .iter()
            .position(|c| {
                c.author == "user"
                    && matches!(&c.content[..], [CommentContent::Text(t)] if t == "SUMMARY")
            })
            .expect("expected a `user` comment carrying the summariser text");
        assert!(summary_idx >= 1, "summary must follow the system prompt");

        assert!(
            !comments.iter().any(|c| {
                matches!(&c.content[..], [CommentContent::Text(t)] if t == "turn 1" || t == "go")
            }),
            "compaction must drop pre-compaction non-system comments",
        );
    }
}
