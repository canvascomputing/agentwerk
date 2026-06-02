//! Multi-agent loop driver. One tokio task per registered agent,
//! reading the shared `TicketSystem` through the upgraded
//! `Weak<TicketSystem>` stamped at `bind_agent`.

use std::time::Duration;

use crate::tools::ToolCall;

#[path = "3_compaction.rs"]
mod compaction;
#[path = "1_main.rs"]
mod main;
#[path = "4_reply.rs"]
mod reply;
#[path = "5_tool_call.rs"]
mod tool_call;
#[path = "2_turn.rs"]
mod turn;

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
    use crate::agents::tickets::ReplyContent;
    use crate::schemas::Schema;

    // Reply transcript

    #[tokio::test]
    async fn replies_capture_full_transcript() {
        let provider = MockProvider::with_results(vec![Ok(write_result_response("ok"))]);
        let (_, _, ticket) = run_one(provider, 3, 10, None).await;

        let replies = &ticket.replies;
        assert_eq!(replies.len(), 5, "got {replies:?}");

        assert_eq!(replies[0].author, "system");
        assert!(matches!(&replies[0].content[..], [ReplyContent::Text(_)]));

        assert_eq!(replies[1].author, "user");
        assert!(
            matches!(&replies[1].content[..], [ReplyContent::Text(t)] if t.starts_with("## Context")),
            "second reply must be the auto-injected context prelude",
        );

        assert_eq!(replies[2].author, "user");
        assert!(
            matches!(&replies[2].content[..], [ReplyContent::Text(t)] if t == "go"),
            "third reply must carry the task body",
        );

        assert_eq!(replies[3].author, "assistant");
        assert!(
            matches!(&replies[3].content[..], [ReplyContent::ToolUse { name, .. }] if name == "finish_ticket"),
            "assistant reply must mirror the model's ToolUse block",
        );

        assert_eq!(replies[4].author, "user");
        assert!(
            matches!(&replies[4].content[..], [ReplyContent::ToolResult { .. }]),
            "tool-result reply must carry a ToolResult block",
        );

        for w in replies.windows(2) {
            assert!(
                w[0].created_at <= w[1].created_at,
                "reply timestamps must be monotonic",
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

        let replies = &ticket.replies;

        let first_assistant = replies
            .iter()
            .position(|r| {
                r.author == "assistant"
                    && matches!(&r.content[..], [ReplyContent::Text(t)] if t == "Hello!")
            })
            .expect("expected the text-only assistant reply in the transcript");

        let directive = &replies[first_assistant + 1];
        assert_eq!(directive.author, "user");
        let directive_text = match &directive.content[..] {
            [ReplyContent::Text(t)] => t,
            other => panic!("expected a single text block for the directive, got {other:?}"),
        };
        assert!(
            directive_text.contains("finish_ticket"),
            "directive must name the missing finisher: {directive_text}",
        );

        let second_assistant = replies
            .iter()
            .skip(first_assistant + 2)
            .find(|r| {
                r.author == "assistant"
                    && matches!(&r.content[..], [ReplyContent::ToolUse { name, .. }] if name == "finish_ticket")
            });
        assert!(
            second_assistant.is_some(),
            "expected a recovering ToolUse assistant reply after the directive",
        );
    }

    #[tokio::test]
    async fn replies_after_compaction_keep_only_system_and_summary() {
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

        let replies = &ticket.replies;

        assert_eq!(replies[0].author, "system");

        let summary_idx = replies
            .iter()
            .position(|r| {
                r.author == "user"
                    && matches!(&r.content[..], [ReplyContent::Text(t)] if t == "SUMMARY")
            })
            .expect("expected a `user` reply carrying the summariser text");
        assert!(summary_idx >= 1, "summary must follow the system prompt");

        assert!(
            !replies.iter().any(|r| {
                matches!(&r.content[..], [ReplyContent::Text(t)] if t == "turn 1" || t == "go")
            }),
            "compaction must drop pre-compaction non-system replies",
        );
    }
}
