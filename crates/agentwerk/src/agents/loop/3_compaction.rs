//! Stateful compaction orchestration: wraps the pure algorithms in `agents/compaction` with ticket mutation, event emission, and persistence.

use crate::event::{CompactReason, EventKind};
use crate::providers::types::TokenUsage;
use crate::providers::{Message, ProviderError};
use crate::agents::compaction as algo;
use crate::agents::tickets::to_messages;

use super::turn::{fail_ticket, model_name, LoopContext};
use super::Action;

pub(super) async fn compact(scope: &mut LoopContext<'_>, reason: CompactReason) -> Action<()> {
    scope.agent.emit(EventKind::CompactionStarted { reason });

    let Some(ticket) = scope.ticket_system.get_ticket(&scope.ticket_key) else {
        return Action::Stop;
    };

    let model_name = model_name(scope);
    let messages = to_messages(&ticket.comments);
    let result = algo::compact(&scope.agent.provider_handle(), &model_name, &messages).await;

    if let Err(e) = result {
        scope.agent.emit(EventKind::CompactionFailed { reason, message: e.to_string() });
        fail_ticket(scope, &e);
        return Action::Stop;
    }

    let summary = result.unwrap();

    if summary.is_none() && matches!(reason, CompactReason::Reactive) {
        fail_ticket(scope, &ProviderError::ContextWindowExceeded {
            message: "context still exceeds window after compaction".into(),
        });
        return Action::Stop;
    }

    if let Some(summary) = summary {
        let dir = scope.ticket_system.dir_value();
        if let Some(t) = scope.ticket_system.tickets.lock().unwrap().get_mut(&scope.ticket_key) {
            t.summarize(summary);
        }
        if let Some(t) = scope.ticket_system.get_ticket(&scope.ticket_key) {
            use crate::persistence::Persist;
            let _ = t.save(&dir);
        }

        if matches!(reason, CompactReason::Reactive) {
            let window = scope.agent.model.as_ref().and_then(|m| m.context_window);
            if let Some(threshold) = algo::blocking_threshold(window) {
                let updated = scope
                    .ticket_system
                    .get_ticket(&scope.ticket_key)
                    .map(|t| to_messages(&t.comments))
                    .unwrap_or_default();
                let tools = scope.agent.tool_definitions();
                let default_usage = TokenUsage::default();
                let usage = scope.last_usage.as_ref().unwrap_or(&default_usage);
                let estimate = algo::estimate_next_request_tokens(
                    usage,
                    &updated,
                    &scope.system_prompt,
                    &tools,
                );
                if estimate >= threshold {
                    fail_ticket(scope, &ProviderError::ContextWindowExceeded {
                        message: "context still exceeds window after compaction".into(),
                    });
                    return Action::Stop;
                }
            }
        }
    }

    scope.agent.emit(EventKind::CompactionFinished { reason });
    match reason {
        CompactReason::Proactive => Action::Proceed(()),
        CompactReason::Reactive => Action::Replay,
    }
}

pub(super) async fn proactive_compact(
    scope: &mut LoopContext<'_>,
    mut messages: Vec<Message>,
) -> Action<Vec<Message>> {
    let tools = scope.agent.tool_definitions();
    let window = scope.agent.model.as_ref().and_then(|m| m.context_window);

    let exceeds_proactive_threshold = scope
        .last_usage
        .as_ref()
        .is_some_and(|usage| algo::should_compact_proactively(window, usage, &messages, &scope.system_prompt, &tools));

    if exceeds_proactive_threshold {
        match compact(scope, CompactReason::Proactive).await {
            Action::Stop => return Action::Stop,
            Action::Replay => return Action::Replay,
            Action::Proceed(()) => {}
        }
        messages = scope
            .ticket_system
            .get_ticket(&scope.ticket_key)
            .map(|t| to_messages(&t.comments))
            .unwrap_or_default();
    }

    if let Some(threshold) = algo::blocking_threshold(window) {
        let default_usage = TokenUsage::default();
        let usage = scope.last_usage.as_ref().unwrap_or(&default_usage);
        let estimate = algo::estimate_next_request_tokens(
            usage,
            &messages,
            &scope.system_prompt,
            &tools,
        );
        if estimate >= threshold {
            scope.agent.emit(EventKind::BlockingLimitExceeded {
                estimated_tokens: estimate,
                threshold_tokens: threshold,
            });
            match compact(scope, CompactReason::Reactive).await {
                Action::Stop => return Action::Stop,
                Action::Replay => return Action::Replay,
                Action::Proceed(()) => {}
            }
        }
    }

    Action::Proceed(messages)
}
