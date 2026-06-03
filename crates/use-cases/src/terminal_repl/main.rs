//! Interactive terminal chat. One `TicketSystem` + `Agent` + `Knowledge`
//! lives for the whole session, and one chat ticket spans every turn:
//! the first input creates the ticket via `tickets.task(...)`, every
//! subsequent input lands as a user reply via `tickets.reply(&key, ...)`.
//! The agent loop's wait-for-input branch picks each comment up and
//! drives the next model turn on the same growing transcript. Tickets
//! and knowledge both persist to `./.agentwerk/`, so an existing chat
//! resumes across process restarts.
//! The model's response streams to stdout via
//! `EventKind::TextChunkReceived`. Slash commands:
//! `/new` starts a fresh chat ticket, `/list` lists every ticket,
//! `/stats` prints run counters, `/clear` resets knowledge.
//! Ctrl-C at the prompt exits with code 130; Ctrl-D exits with
//! code 0; Ctrl-C during a turn cancels that turn (a second Ctrl-C
//! while the cancel is still draining force-quits with exit code 130).
//!
//! Every exit path goes through `std::process::exit` rather than a
//! plain `return`: the stdin reader runs on a tokio blocking thread
//! parked in `read(2)`, which the runtime can't cancel on shutdown.
//! Exiting the process directly bypasses the runtime drop and avoids
//! a hang on outstanding blocking tasks.

use std::io::{self, IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use agentwerk::agents::tickets::ReplyContent;
use agentwerk::event::{Event, EventKind};
use agentwerk::tools::{
    GlobTool, GrepTool, ListDirectoryTool, ManageTicketsTool, ReadFileTool, ReadTicketsTool,
    WriteFileTool,
};
use agentwerk::{Agent, Knowledge, Ticket, TicketSystem};

const ROLE: &str = include_str!("prompts/repl.role.md");

#[tokio::main]
async fn main() {
    let style = Style::detect();
    eprintln!(
        "{}agentwerk REPL: /new /list /stats /clear, Ctrl-C to cancel.{}",
        style.dim, style.reset,
    );

    // Optional first positional arg: synthetic context-window size in
    // **tokens**, used only to drive a REPL-side usage line and warning
    // when the reported `input_tokens` crosses 70% of it. The library's
    // own `ContextCompacted` event is unaffected and still fires only
    // at the real model window.
    let test_window: Option<u64> = std::env::args().nth(1).and_then(|s| s.parse().ok());
    if let Some(w) = test_window {
        let threshold = w.saturating_mul(7) / 10;
        eprintln!(
            "{}test context window: {w} tokens (synthetic warning at {threshold} tokens used){}",
            style.dim, style.reset,
        );
    }

    let role = ROLE.trim();

    let user_prompt = format!("\n{}you ›{} ", style.user, style.reset);

    let event_style = style.clone();
    // `midstream` tracks whether the last byte written was streamed
    // model text (no trailing newline). Stderr event lines consult it
    // to break out of the stream exactly once, instead of every
    // `eprintln!` doubling up newlines.
    let midstream = Arc::new(AtomicBool::new(false));
    let handler_midstream = Arc::clone(&midstream);
    let handler: Arc<dyn Fn(Event) + Send + Sync> =
        Arc::new(move |e: Event| print_event(&e, &event_style, test_window, &handler_midstream));

    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
    let store_dir = cwd.join(".agentwerk");
    let tickets = TicketSystem::load(&store_dir).expect("open ticket store");
    tickets.max_turns(40);

    let knowledge = Knowledge::load(&store_dir).expect("open knowledge store");

    tickets.event_handler(move |e| handler(e));
    let _agent = tickets.agent(
        Agent::new()
            .name("orchestrator")
            .interactive()
            .from_env()
            .role(role)
            .dir(&cwd)
            .tool(GlobTool)
            .tool(GrepTool)
            .tool(ListDirectoryTool)
            .tool(ReadFileTool)
            .tool(WriteFileTool)
            .tool(ReadTicketsTool)
            .tool(ManageTicketsTool)
            .knowledge(&knowledge),
    );

    let mut prev_turns: u64 = 0;
    let mut prev_requests: u64 = 0;
    let mut prev_tool_calls: u64 = 0;
    let mut prev_input: u64 = 0;
    let mut prev_output: u64 = 0;

    // Resume the newest open chat ticket from disk if one exists,
    // so a previous session's transcript carries into this one.
    let mut chat_key: Option<String> = tickets
        .find_tickets(|t| {
            t.status.to_string() == "in_progress" && t.labels.iter().any(|l| l == "orchestrator")
        })
        .last()
        .map(|t| t.key.to_string());
    if let Some(k) = &chat_key {
        eprintln!("{}resumed chat ticket {k}{}", style.dim, style.reset,);
    }

    // One long-running loop drives every turn; each user input flips the
    // ticket out of the gate's pause and the next iteration redraws the
    // prompt once the assistant has spoken.
    tickets.start();

    loop {
        let line = tokio::select! {
            line = read_line(&user_prompt) => line,
            _ = tokio::signal::ctrl_c() => {
                eprintln!("\n{}^C{}", style.dim, style.reset);
                std::process::exit(130);
            }
        };
        let Some(line) = line else {
            std::process::exit(0)
        };
        if line.is_empty() {
            continue;
        }
        if line == "/new" {
            let k = tickets.task("<new chat>");
            eprintln!("{}new chat {k}{}", style.dim, style.reset);
            chat_key = Some(k);
            continue;
        }
        if line == "/list" {
            let all = tickets.tickets();
            if all.is_empty() {
                eprintln!("{}(no tickets){}", style.dim, style.reset);
            } else {
                for t in all {
                    let preview = match &t.task {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let mut preview: String = preview.chars().take(60).collect();
                    if preview.len() < t.task.to_string().len() {
                        preview.push('…');
                    }
                    let active = chat_key.as_deref() == Some(t.key.as_str());
                    let mark = if active { "▸ " } else { "  " };
                    eprintln!(
                        "{}{mark}{} [{}] · {} replies · {}{}",
                        style.dim,
                        t.key,
                        t.status,
                        t.replies.len(),
                        preview,
                        style.reset,
                    );
                }
            }
            continue;
        }
        if line == "/stats" {
            let s = tickets.stats();
            eprintln!(
                "{}{} turns · {} requests · {} tools · {} in / {} out · {} created / {} done / {} failed{}",
                style.dim,
                s.turns(),
                s.requests(),
                s.tool_calls(),
                s.input_tokens(),
                s.output_tokens(),
                s.tickets_created(),
                s.tickets_finished(),
                s.tickets_failed(),
                style.reset,
            );
            continue;
        }
        if line == "/clear" {
            knowledge.clear().ok();
            eprintln!("{}knowledge cleared{}", style.dim, style.reset);
            continue;
        }

        announce_assistant(&style);
        // "agent › " left stdout mid-line; mark so the first event
        // breaks out before its own content.
        midstream.store(true, Ordering::Relaxed);
        let key = match chat_key.as_deref() {
            Some(k)
                if tickets
                    .tickets()
                    .iter()
                    .any(|t| t.key == k && t.status.to_string() == "in_progress") =>
            {
                tickets.reply(k, line);
                k.to_string()
            }
            _ => tickets.task(line),
        };
        chat_key = Some(key.clone());

        let cancelled = tokio::select! {
            _ = wait_for_assistant_pause(&tickets, &key) => false,
            _ = tokio::signal::ctrl_c() => {
                tickets.cancel();
                if midstream.swap(false, Ordering::Relaxed) {
                    eprintln!();
                }
                eprintln!("{}cancelling…{}", style.dim, style.reset);
                let drain_fut = tickets.finish();
                tokio::pin!(drain_fut);
                tokio::select! {
                    _ = &mut drain_fut => {}
                    _ = tokio::signal::ctrl_c() => std::process::exit(130),
                }
                tickets.start();
                true
            }
        };

        let stats = tickets.stats();
        let outcome = {
            let status = chat_key.as_deref().and_then(|k| {
                tickets
                    .tickets()
                    .into_iter()
                    .find(|t| t.key == k)
                    .map(|t| t.status.to_string())
            });
            match status.as_deref() {
                Some("done") => {
                    chat_key = None;
                    "completed"
                }
                Some("failed") => {
                    chat_key = None;
                    "failed"
                }
                _ if cancelled => "cancelled",
                _ => "incomplete",
            }
        };

        let turns = stats.turns().saturating_sub(prev_turns);
        let requests = stats.requests().saturating_sub(prev_requests);
        let tool_calls = stats.tool_calls().saturating_sub(prev_tool_calls);
        let input = stats.input_tokens().saturating_sub(prev_input);
        let output = stats.output_tokens().saturating_sub(prev_output);
        prev_turns = stats.turns();
        prev_requests = stats.requests();
        prev_tool_calls = stats.tool_calls();
        prev_input = stats.input_tokens();
        prev_output = stats.output_tokens();

        if midstream.swap(false, Ordering::Relaxed) {
            eprintln!();
        }
        eprintln!(
            "{}{outcome} · {turns} turns · {requests} requests · {tool_calls} tools · {input} in / {output} out{}",
            style.dim, style.reset,
        );
    }
}

fn announce_assistant(style: &Style) {
    print!("\n{}agent ›{} ", style.agent, style.reset);
    let _ = io::stdout().flush();
}

fn print_event(event: &Event, style: &Style, test_window: Option<u64>, midstream: &AtomicBool) {
    // Emit a single leading newline only when streamed model text just
    // landed on stdout without a trailing newline; subsequent events
    // print directly on their own line.
    let break_stream = || {
        if midstream.swap(false, Ordering::Relaxed) {
            eprintln!();
        }
    };
    match &event.kind {
        EventKind::TextChunkReceived { content } => {
            print!("{content}");
            let _ = io::stdout().flush();
            midstream.store(true, Ordering::Relaxed);
        }
        EventKind::ToolCallStarted {
            tool_name, input, ..
        } => {
            break_stream();
            let arg = input["pattern"]
                .as_str()
                .or_else(|| input["path"].as_str())
                .or_else(|| input["query"].as_str())
                .unwrap_or("");
            if arg.is_empty() {
                eprintln!("{}· {tool_name}{}", style.dim, style.reset);
            } else {
                eprintln!("{}· {tool_name}({arg}){}", style.dim, style.reset);
            }
        }
        EventKind::ToolCallFailed {
            tool_name, message, ..
        } => {
            break_stream();
            eprintln!("{}✗ {tool_name}: {message}{}", style.red, style.reset);
        }
        EventKind::RequestFinished { usage, .. } => {
            if let Some(window) = test_window {
                break_stream();
                let used = usage.input_tokens;
                let remaining = window.saturating_sub(used);
                let threshold = window.saturating_mul(7) / 10;
                let (marker, color) = if used >= threshold {
                    ("⚠", style.red)
                } else {
                    ("·", style.dim)
                };
                eprintln!(
                    "{color}{marker} {used} / {window} tokens used ({remaining} left, warn at {threshold}){reset}",
                    reset = style.reset,
                );
            }
        }
        EventKind::CompactionStarted { reason } => {
            break_stream();
            eprintln!(
                "{}… compacting context ({reason:?}){}",
                style.dim, style.reset,
            );
        }
        EventKind::CompactionFinished { reason } => {
            break_stream();
            eprintln!(
                "{}✓ context compacted ({reason:?}){}",
                style.dim, style.reset,
            );
        }
        EventKind::CompactionFailed { reason, message } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "{}✗ compaction failed ({reason:?}): {short}{}",
                style.red, style.reset,
            );
        }
        EventKind::RequestFailed { message, .. } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!("{}✗ request failed: {short}{}", style.red, style.reset);
        }
        EventKind::RequestRetried {
            attempt,
            max_attempts,
            message,
            ..
        } => {
            break_stream();
            let short = message.split_once(':').map(|(h, _)| h).unwrap_or(message);
            eprintln!(
                "{}↻ retry {attempt}/{max_attempts}: {short}{}",
                style.dim, style.reset,
            );
        }
        EventKind::SchemaRetried { .. } => {}
        EventKind::PolicyViolated { kind, limit } => {
            break_stream();
            eprintln!(
                "{}✗ policy {kind:?} (limit {limit}){}",
                style.red, style.reset,
            );
        }
        _ => {}
    }
}

/// Block until the chat ticket is either terminal (`finished`/`failed`)
/// or sitting at the gate's text-only pause. A mid-turn assistant reply
/// carrying a tool call doesn't count: tool execution is still pending
/// and the prompt would race the user against the loop.
async fn wait_for_assistant_pause(tickets: &TicketSystem, key: &str) {
    loop {
        match tickets.get_ticket(key) {
            None => return,
            Some(t) if is_terminal(&t) => return,
            Some(t) if is_paused_on_text(&t) => return,
            _ => {}
        }
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    }
}

fn is_terminal(ticket: &Ticket) -> bool {
    let s = ticket.status.to_string();
    s == "finished" || s == "failed"
}

fn is_paused_on_text(ticket: &Ticket) -> bool {
    ticket.replies.last().is_some_and(|r| {
        r.author == "assistant" && r.content.iter().all(|c| matches!(c, ReplyContent::Text(_)))
    })
}

async fn read_line(prompt: &str) -> Option<String> {
    print!("{prompt}");
    io::stdout().flush().ok()?;
    tokio::task::spawn_blocking(|| io::stdin().lines().next()?.ok().map(|s| s.trim().into()))
        .await
        .ok()
        .flatten()
}

#[derive(Clone)]
struct Style {
    dim: &'static str,
    user: &'static str,
    agent: &'static str,
    red: &'static str,
    reset: &'static str,
}

impl Style {
    fn detect() -> Self {
        if io::stdout().is_terminal() && io::stderr().is_terminal() {
            Self {
                dim: "\x1b[2m",
                user: "\x1b[1;33m",
                agent: "\x1b[1;36m",
                red: "\x1b[31m",
                reset: "\x1b[0m",
            }
        } else {
            Self {
                dim: "",
                user: "",
                agent: "",
                red: "",
                reset: "",
            }
        }
    }
}
