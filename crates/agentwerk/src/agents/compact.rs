//! Context-window compaction. The threshold seams (`proactive_event`,
//! `reactive_event`) produce the typed warning event; `summarize_and_replace`
//! collapses the tail of an agent's message vector into a single user-role
//! summary by calling the provider with no tools.

use std::sync::Arc;

use crate::event::{CompactReason, EventKind};
use crate::prompts::compaction_directive;
use crate::providers::types::StreamEvent;
use crate::providers::{
    ContentBlock, Message, ModelRequest, Provider, ProviderError, ProviderResult, TokenUsage,
};

/// Tokens reserved for the model's response. The context window holds
/// input + output combined, so the input must leave at least this much
/// room for the next reply.
const RESERVED_RESPONSE_TOKENS: u64 = 20_000;

/// Headroom below the hard window limit so the warning fires with room
/// to spare. Also absorbs drift in the `bytes / 4` token estimate, which
/// tends to under-count code and JSON.
const COMPACTION_HEADROOM_TOKENS: u64 = 13_000;

/// Token count at which the proactive seam fires for a model with
/// context window `window`. `None` when the window is unknown.
pub(crate) fn threshold(window: Option<u64>) -> Option<u64> {
    window.map(|size| {
        size.saturating_sub(RESERVED_RESPONSE_TOKENS)
            .saturating_sub(COMPACTION_HEADROOM_TOKENS)
    })
}

/// Estimate of the next request's input-token count: the last response's
/// reported input tokens plus a `bytes / 4` estimate over any messages
/// appended since.
pub(crate) fn estimate_next_request_tokens(last_usage: &TokenUsage, messages: &[Message]) -> u64 {
    let new_bytes: usize = messages.iter().map(message_bytes).sum();
    last_usage.input_tokens + (new_bytes / 4) as u64
}

fn message_bytes(message: &Message) -> usize {
    match message {
        Message::System { content } => content.len(),
        Message::User { content } | Message::Assistant { content } => {
            content.iter().map(block_bytes).sum()
        }
    }
}

fn block_bytes(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text } => text.len(),
        ContentBlock::ToolUse { name, input, .. } => name.len() + input.to_string().len(),
        ContentBlock::ToolResult { content, .. } => content.len(),
    }
}

/// Proactive seam: the warning event when the estimated next-request
/// input crosses the threshold. `None` when the window is unknown or
/// the estimate is still under it.
pub(crate) fn proactive_event(
    window: Option<u64>,
    last_usage: &TokenUsage,
    messages: &[Message],
) -> Option<EventKind> {
    let threshold = threshold(window)?;
    let tokens = estimate_next_request_tokens(last_usage, messages);
    if tokens < threshold {
        return None;
    }
    Some(EventKind::ContextCompacted {
        tokens,
        threshold,
        reason: CompactReason::Proactive,
    })
}

/// Reactive seam: the warning event when the provider itself reports
/// context-window overflow. Carries sentinel `tokens = 0` and
/// `threshold = 0` since the authoritative numbers come from the
/// provider, not our estimator.
pub(crate) fn reactive_event() -> EventKind {
    EventKind::ContextCompacted {
        tokens: 0,
        threshold: 0,
        reason: CompactReason::Reactive,
    }
}

/// Replace `messages[head_len..]` with one user-role message containing a
/// model-generated summary. The first `head_len` entries (the context
/// message, if any, and the task message) stay verbatim; everything
/// after is passed to the provider with no tools, and the assistant's
/// text reply becomes the new tail.
///
/// No-ops when there are fewer than two messages to summarize.
pub(crate) async fn summarize_and_replace(
    provider: &Arc<dyn Provider>,
    model: &str,
    messages: &mut Vec<Message>,
    head_len: usize,
) -> ProviderResult<()> {
    if messages.len() <= head_len + 1 {
        return Ok(());
    }
    let request = ModelRequest {
        model: model.to_string(),
        system_prompt: compaction_directive().to_string(),
        messages: messages[head_len..].to_vec(),
        tools: Vec::new(),
        max_request_tokens: None,
        tool_choice: None,
    };
    let on_stream: Arc<dyn Fn(StreamEvent) + Send + Sync> = Arc::new(|_| {});
    let response = provider.respond(request, on_stream).await?;
    let summary = response
        .content
        .iter()
        .find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .ok_or_else(|| ProviderError::ResponseMalformed {
            message: "compaction reply contained no text".into(),
        })?;
    messages.truncate(head_len);
    messages.push(Message::user(summary));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_saturates_on_tiny_or_zero_window() {
        assert_eq!(threshold(Some(100)), Some(0));
        assert_eq!(threshold(Some(0)), Some(0));
    }

    #[test]
    fn threshold_is_none_for_unknown_window() {
        assert_eq!(threshold(None), None);
    }

    #[test]
    fn estimate_sums_last_input_tokens_and_byte_quarters() {
        // 400 bytes / 4 = 100; plus last response's 5_000 input tokens = 5_100.
        let usage = TokenUsage {
            input_tokens: 5_000,
            output_tokens: 200,
        };
        let messages = [Message::user("x".repeat(400))];
        assert_eq!(estimate_next_request_tokens(&usage, &messages), 5_100);
    }

    #[test]
    fn proactive_event_none_when_window_unknown() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(proactive_event(None, &usage, &messages).is_none());
    }

    #[test]
    fn proactive_event_none_when_under_threshold() {
        let usage = TokenUsage {
            input_tokens: 1_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        assert!(proactive_event(Some(200_000), &usage, &messages).is_none());
    }

    #[test]
    fn proactive_event_some_when_over_threshold() {
        let usage = TokenUsage {
            input_tokens: 170_000,
            output_tokens: 0,
        };
        let messages = [Message::user("hi")];
        let event = proactive_event(Some(200_000), &usage, &messages)
            .expect("threshold should have tripped");
        match event {
            EventKind::ContextCompacted {
                tokens,
                threshold,
                reason,
            } => {
                assert_eq!(threshold, 167_000);
                assert!(tokens >= 170_000);
                assert_eq!(reason, CompactReason::Proactive);
            }
            other => panic!("expected ContextCompacted, got {other:?}"),
        }
    }

    #[test]
    fn reactive_event_carries_sentinel_zeros() {
        match reactive_event() {
            EventKind::ContextCompacted {
                tokens,
                threshold,
                reason,
            } => {
                assert_eq!(tokens, 0);
                assert_eq!(threshold, 0);
                assert_eq!(reason, CompactReason::Reactive);
            }
            other => panic!("expected ContextCompacted, got {other:?}"),
        }
    }

    // ---- summarize_and_replace ----

    use crate::providers::types::{ModelResponse, ResponseStatus};
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Mutex as StdMutex;

    /// Scripted provider: serves one canned result per `respond` call
    /// in FIFO order, and records the request it received so tests
    /// can assert on it.
    struct ScriptedProvider {
        results: StdMutex<Vec<ProviderResult<ModelResponse>>>,
        received: StdMutex<Vec<ModelRequest>>,
    }

    impl ScriptedProvider {
        fn new(results: Vec<ProviderResult<ModelResponse>>) -> Arc<Self> {
            Arc::new(Self {
                results: StdMutex::new(results),
                received: StdMutex::new(Vec::new()),
            })
        }

        fn last_request(&self) -> Option<ModelRequest> {
            self.received.lock().unwrap().last().cloned()
        }

        fn call_count(&self) -> usize {
            self.received.lock().unwrap().len()
        }
    }

    impl Provider for ScriptedProvider {
        fn respond(
            &self,
            request: ModelRequest,
            _on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
        ) -> Pin<Box<dyn Future<Output = ProviderResult<ModelResponse>> + Send + '_>> {
            self.received.lock().unwrap().push(request);
            let mut results = self.results.lock().unwrap();
            if results.is_empty() {
                panic!("ScriptedProvider out of scripted results");
            }
            let next = results.remove(0);
            Box::pin(async move { next })
        }
    }

    fn summary_response(text: &str) -> ModelResponse {
        ModelResponse {
            content: vec![ContentBlock::Text { text: text.into() }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        }
    }

    /// Conversation with `head_len` head messages (`ctx`, `task`) and
    /// `tail` alternating assistant/user messages, so tests can spell
    /// the message shape they need without scattering vec![...] noise.
    fn conversation(head_len: usize, tail: usize) -> Vec<Message> {
        assert!((1..=2).contains(&head_len), "head_len must be 1 or 2");
        let mut msgs = Vec::with_capacity(head_len + tail);
        if head_len == 2 {
            msgs.push(Message::user("ctx"));
        }
        msgs.push(Message::user("task"));
        for i in 0..tail {
            msgs.push(if i % 2 == 0 {
                Message::assistant(format!("turn {i}"))
            } else {
                Message::user(format!("turn {i} result"))
            });
        }
        msgs
    }

    /// Last message's first text block, for tests that want to
    /// confirm the summary landed where they expect.
    fn last_text(messages: &[Message]) -> &str {
        match messages.last().expect("messages must be non-empty") {
            Message::User { content } | Message::Assistant { content } => match &content[0] {
                ContentBlock::Text { text } => text,
                other => panic!("expected text block, got {other:?}"),
            },
            Message::System { content } => content,
        }
    }

    #[tokio::test]
    async fn summarize_and_replace_keeps_head_and_replaces_tail() {
        let provider: Arc<dyn Provider> =
            ScriptedProvider::new(vec![Ok(summary_response("SUMMARY"))]);
        let mut messages = conversation(2, 3);

        summarize_and_replace(&provider, "mock", &mut messages, 2)
            .await
            .expect("summarize should succeed");

        assert_eq!(messages.len(), 3);
        assert_eq!(last_text(&messages), "SUMMARY");
    }

    #[tokio::test]
    async fn summarize_and_replace_keeps_a_single_head_message() {
        // No context message: head_len = 1, so only the task survives.
        let provider: Arc<dyn Provider> =
            ScriptedProvider::new(vec![Ok(summary_response("SUMMARY"))]);
        let mut messages = conversation(1, 3);

        summarize_and_replace(&provider, "mock", &mut messages, 1)
            .await
            .unwrap();

        assert_eq!(messages.len(), 2);
        assert!(matches!(&messages[0], Message::User { content }
            if matches!(&content[0], ContentBlock::Text { text } if text == "task")));
        assert_eq!(last_text(&messages), "SUMMARY");
    }

    #[tokio::test]
    async fn summarize_and_replace_is_a_noop_when_tail_too_short_to_summarize() {
        // Boundary: nothing gained from summarizing zero or one message.
        for tail in [0, 1] {
            let provider = ScriptedProvider::new(Vec::new());
            let provider_handle: Arc<dyn Provider> = provider.clone();
            let mut messages = conversation(2, tail);
            let original = messages.clone();

            summarize_and_replace(&provider_handle, "mock", &mut messages, 2)
                .await
                .expect("no-op should succeed");

            assert_eq!(messages.len(), original.len(), "tail={tail}");
            assert_eq!(
                provider.call_count(),
                0,
                "tail={tail}: provider must not be called"
            );
        }
    }

    #[tokio::test]
    async fn summarize_and_replace_propagates_provider_error() {
        let provider: Arc<dyn Provider> =
            ScriptedProvider::new(vec![Err(ProviderError::ConnectionFailed {
                message: "dns".into(),
            })]);
        let mut messages = conversation(2, 2);
        let original_len = messages.len();

        let err = summarize_and_replace(&provider, "mock", &mut messages, 2)
            .await
            .expect_err("should propagate the connection failure");

        assert!(matches!(err, ProviderError::ConnectionFailed { .. }));
        assert_eq!(
            messages.len(),
            original_len,
            "messages must not be mutated on error"
        );
    }

    #[tokio::test]
    async fn summarize_and_replace_rejects_text_less_reply() {
        let no_text = ModelResponse {
            content: vec![ContentBlock::ToolUse {
                id: "x".into(),
                name: "irrelevant".into(),
                input: serde_json::json!({}),
            }],
            status: ResponseStatus::EndTurn,
            usage: TokenUsage::default(),
            model: "mock".into(),
        };
        let provider: Arc<dyn Provider> = ScriptedProvider::new(vec![Ok(no_text)]);
        let mut messages = conversation(2, 2);

        let err = summarize_and_replace(&provider, "mock", &mut messages, 2)
            .await
            .expect_err("text-less reply must fail");

        assert!(matches!(err, ProviderError::ResponseMalformed { .. }));
    }

    #[tokio::test]
    async fn builds_a_tool_less_summarization_request() {
        let provider = ScriptedProvider::new(vec![Ok(summary_response("SUMMARY"))]);
        let provider_handle: Arc<dyn Provider> = provider.clone();
        let mut messages = conversation(2, 2);

        summarize_and_replace(&provider_handle, "mock", &mut messages, 2)
            .await
            .unwrap();

        let req = provider.last_request().expect("provider was called");
        assert!(req.tools.is_empty(), "tools must be disabled");
        assert!(req.tool_choice.is_none(), "tool_choice must be unset");
        assert_eq!(req.messages.len(), 2, "only the tail is summarized");
        assert_eq!(req.system_prompt, compaction_directive());
    }
}
