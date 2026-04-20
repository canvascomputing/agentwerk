//! Terminal REPL: chat with a long-lived local-search agent.
//!
//! Usage: terminal-repl
//!
//! Type instructions at the `> ` prompt. The agent has read-only search tools
//! (glob, grep, list_directory, read_file) scoped to the working directory.
//! Type `/exit` or `/quit` (or press Ctrl-D) to end the session.
//! Press Ctrl-C to cancel in-flight work and exit.
//!
//! Environment:
//!   ANTHROPIC_API_KEY   (or other provider env vars)

use std::io::{self, Write};
use std::sync::Arc;

use agentwerk::{
    Agent, AgentEvent, AgentEventKind, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool,
};
use tokio::sync::Notify;

const IDENTITY_PROMPT: &str =
    "You are a local-repo search assistant. Help the user explore this repository \
     using the provided read-only tools: glob, grep, list_directory, and read_file. \
     Cite file:line when referencing code. Do not invent facts — if unsure, search first.";

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    eprintln!("agentwerk REPL — type /exit to quit, Ctrl-C to cancel.\n");

    let Some(first) = read_line_async("> ").await else { return };
    if first.is_empty() || first == "/exit" || first == "/quit" {
        return;
    }

    let idle = Arc::new(Notify::new());
    let idle_for_handler = idle.clone();

    let running = Agent::new()
        .name("orchestrator")
        .provider_from_env()
        .expect("LLM provider required")
        .identity_prompt(IDENTITY_PROMPT)
        .instruction_prompt(first)
        .tool(GlobTool)
        .tool(GrepTool)
        .tool(ListDirectoryTool)
        .tool(ReadFileTool)
        .keep_alive()
        .event_handler(Arc::new(move |e: AgentEvent| {
            print_event(&e, &idle_for_handler)
        }))
        .create();

    loop {
        tokio::select! {
            _ = idle.notified() => {}
            _ = tokio::signal::ctrl_c() => break,
        }
        if running.is_stopped() {
            break;
        }
        let line = tokio::select! {
            line = read_line_async("> ") => line,
            _ = tokio::signal::ctrl_c() => { eprintln!("^C"); None }
        };
        let Some(line) = line else { break };
        if line.is_empty() {
            continue;
        }
        if line == "/exit" || line == "/quit" {
            break;
        }
        running.send(line);
    }

    running.cancel();
    match running.run().await {
        Ok(output) => eprintln!(
            "[session ended: {:?}, turns={}, tokens={}in/{}out]",
            output.status,
            output.statistics.turns,
            output.statistics.input_tokens,
            output.statistics.output_tokens,
        ),
        Err(e) => eprintln!("Error: {e}"),
    }

    // The stdin-reading spawn_blocking thread cannot be cancelled and would
    // block runtime drop. Exit directly.
    std::process::exit(0);
}

// ---------------------------------------------------------------------------
// Event handler
// ---------------------------------------------------------------------------

fn print_event(event: &AgentEvent, idle: &Arc<Notify>) {
    match &event.kind {
        AgentEventKind::ResponseTextChunk { content } => {
            print!("{content}");
            let _ = io::stdout().flush();
        }
        AgentEventKind::ToolCallStart {
            tool_name, input, ..
        } => {
            eprintln!("· {tool_name}({})", compact_input(tool_name, input));
        }
        AgentEventKind::ToolCallError { tool_name, error, .. } => {
            eprintln!("✗ {tool_name}: {error}");
        }
        AgentEventKind::AgentIdle | AgentEventKind::AgentEnd { .. } => {
            println!();
            idle.notify_one();
        }
        _ => {}
    }
}

fn compact_input(tool_name: &str, input: &serde_json::Value) -> String {
    let key = match tool_name {
        "glob" | "grep" => "pattern",
        "list_directory" | "read_file" => "path",
        _ => return serde_json::to_string(input).unwrap_or_default(),
    };
    input[key].as_str().unwrap_or("").to_string()
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

async fn read_line_async(prompt: &str) -> Option<String> {
    print!("{prompt}");
    io::stdout().flush().ok()?;
    tokio::task::spawn_blocking(|| {
        let mut buf = String::new();
        match io::stdin().read_line(&mut buf) {
            Ok(0) => None,
            Ok(_) => Some(buf.trim().to_string()),
            Err(_) => None,
        }
    })
    .await
    .ok()
    .flatten()
}
