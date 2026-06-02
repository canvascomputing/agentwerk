//! Stateful compaction orchestration: wraps the pure algorithms in `agents/compaction` with ticket mutation, event emission, and persistence.

use crate::event::{CompactReason, EventKind};
use crate::providers::types::TokenUsage;
use crate::providers::Message;
use crate::agents::compaction as algo;
use crate::agents::tickets::to_messages;

use super::turn::LoopContext;
use super::Action;

pub(super) async fn compact(context: &mut LoopContext<'_>, reason: CompactReason) -> Action<()> {
    context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::CompactionStarted { reason });

    let Some(ticket) = context.ticket_system.get_ticket(&context.ticket_key) else {
        return Action::Stop;
    };

    let model_name = context.model.name.clone();
    let messages = to_messages(&ticket.comments);
    let result = algo::compact(&context.agent.provider_handle(), &model_name, &messages).await;

    if let Err(e) = result {
        context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::CompactionFailed { reason, message: e.to_string() });
        context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::RequestFailed { kind: e.kind(), message: e.to_string() });
        context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::TicketFailed { key: context.ticket_key.clone() });
        return Action::Stop;
    }

    let summary = result.unwrap();

    if summary.is_none() && matches!(reason, CompactReason::Reactive) {
        context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::RequestFailed {
            kind: crate::providers::RequestErrorKind::ContextWindowExceeded,
            message: "context still exceeds window after compaction".into(),
        });
        context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::TicketFailed { key: context.ticket_key.clone() });
        return Action::Stop;
    }

    if let Some(summary) = summary {
        let dir = context.ticket_system.dir_value();
        if let Some(t) = context.ticket_system.tickets.lock().unwrap().get_mut(&context.ticket_key) {
            t.summarize(summary);
        }
        if let Some(t) = context.ticket_system.get_ticket(&context.ticket_key) {
            use crate::persistence::Persist;
            let _ = t.save(&dir);
        }

        if matches!(reason, CompactReason::Reactive) {
            let window = context.model.context_window;
            if let Some(threshold) = algo::blocking_threshold(window) {
                let updated = context
                    .ticket_system
                    .get_ticket(&context.ticket_key)
                    .map(|t| to_messages(&t.comments))
                    .unwrap_or_default();
                let tools = context.agent.tool_definitions();
                let default_usage = TokenUsage::default();
                let usage = context.last_usage.as_ref().unwrap_or(&default_usage);
                let estimate = algo::estimate_next_request_tokens(
                    usage,
                    &updated,
                    &context.system_prompt,
                    &tools,
                );
                if estimate >= threshold {
                    context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::RequestFailed {
                        kind: crate::providers::RequestErrorKind::ContextWindowExceeded,
                        message: "context still exceeds window after compaction".into(),
                    });
                    context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::TicketFailed { key: context.ticket_key.clone() });
                    return Action::Stop;
                }
            }
        }
    }

    context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::CompactionFinished { reason });
    match reason {
        CompactReason::Proactive => Action::Proceed(()),
        CompactReason::Reactive => Action::Replay,
    }
}

pub(super) async fn proactive_compact(
    context: &mut LoopContext<'_>,
    mut messages: Vec<Message>,
) -> Action<Vec<Message>> {
    let tools = context.agent.tool_definitions();
    let window = context.model.context_window;

    let exceeds_proactive_threshold = context
        .last_usage
        .as_ref()
        .is_some_and(|usage| algo::should_compact_proactively(window, usage, &messages, &context.system_prompt, &tools));

    if exceeds_proactive_threshold {
        match compact(context, CompactReason::Proactive).await {
            Action::Stop => return Action::Stop,
            Action::Replay => return Action::Replay,
            Action::Proceed(()) => {}
        }
        messages = context
            .ticket_system
            .get_ticket(&context.ticket_key)
            .map(|t| to_messages(&t.comments))
            .unwrap_or_default();
    }

    if let Some(threshold) = algo::blocking_threshold(window) {
        let default_usage = TokenUsage::default();
        let usage = context.last_usage.as_ref().unwrap_or(&default_usage);
        let estimate = algo::estimate_next_request_tokens(
            usage,
            &messages,
            &context.system_prompt,
            &tools,
        );
        if estimate >= threshold {
            context.ticket_system.emit(&context.ticket_key, context.agent.get_name(), EventKind::BlockingLimitExceeded {
                estimated_tokens: estimate,
                threshold_tokens: threshold,
            });
            match compact(context, CompactReason::Reactive).await {
                Action::Stop => return Action::Stop,
                Action::Replay => return Action::Replay,
                Action::Proceed(()) => {}
            }
        }
    }

    Action::Proceed(messages)
}
