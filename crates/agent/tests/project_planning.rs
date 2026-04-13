//! Integration test: Agent-driven task management with all TaskTool features.
//!
//! Exercises: create, update, list, get, delete, claim, add_dependency.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use agent::{
    AgentBuilder, AnthropicProvider, Event, LiteLlmProvider, LlmProvider, TaskTool,
};

fn build_provider() -> Option<(Arc<dyn LlmProvider>, String)> {
    let client = reqwest::Client::new();
    if let Ok(url) = std::env::var("LITELLM_API_URL") {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url(url)), model));
    }
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        let mut p = AnthropicProvider::new(key, client);
        if let Ok(url) = std::env::var("ANTHROPIC_BASE_URL") {
            p = p.base_url(url);
        }
        let model = std::env::var("ANTHROPIC_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(p), model));
    }
    if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
        let key = std::env::var("LITELLM_API_KEY").unwrap_or_else(|_| "unused".into());
        let model = std::env::var("LITELLM_MODEL").unwrap_or_else(|_| "claude-sonnet-4-20250514".into());
        return Some((Arc::new(LiteLlmProvider::new(key, client).base_url("http://localhost:4000".into())), model));
    }
    None
}

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let Some((provider, model)) = build_provider() else { eprintln!("SKIPPED: no provider"); return Ok(()); };

    let tmp = tempfile::tempdir()?;

    let output = AgentBuilder::new()
        .name("planner")
        .model(&model)
        .identity_prompt(
            "You are a project planner. Use the task tool to manage work items. \
             The task tool supports these actions: create, update, list, get, delete, claim, add_dependency. \
             Be concise. Always use the task tool, never simulate results.",
        )
        .max_turns(15)
        .tool(TaskTool::open(tmp.path()))
        .provider(provider)
        .instruction_prompt(
            "Do the following steps in order:\n\
             1. Create three tasks: 'Design API', 'Write tests', 'Deploy'\n\
             2. Add a dependency: 'Design API' blocks 'Write tests'\n\
             3. Claim 'Design API' as agent 'alice'\n\
             4. Mark 'Design API' as Completed\n\
             5. Delete 'Deploy'\n\
             6. List all remaining tasks\n\
             7. Get details of 'Write tests' by ID\n\
             8. Summarize what you did",
        )
        .event_handler(Arc::new(|event| match &event {
            Event::TextChunk { content, .. } => print!("{content}"),
            Event::RequestStart { model, .. } => eprintln!("[requesting {model}...]"),
            Event::ToolCallStart { tool_name, .. } => eprintln!("\n[tool] {tool_name}"),
            Event::ToolCallEnd { tool_name, output, is_error, .. } => {
                if *is_error {
                    eprintln!("[error] {tool_name}: {output}");
                } else {
                    eprintln!("[result] {}", &output[..output.len().min(120)]);
                }
            }
            Event::AgentError { message, .. } => eprintln!("[agent error] {message}"),
            Event::AgentEnd { turns, .. } => eprintln!("\n[done in {turns} turn(s)]"),
            _ => {}
        }))
        .cancel_signal(Arc::new(AtomicBool::new(false)))
        .run()
        .await?;

    eprintln!("\n--- Response ---");
    eprintln!("Raw: {:?}", &output.response_raw[..output.response_raw.len().min(200)]);
    eprintln!("Turns: {}, Requests: {}, Tool calls: {}, Cost: ${:.4}",
        output.statistics.turns, output.statistics.requests,
        output.statistics.tool_calls, output.statistics.costs);

    Ok(())
}
