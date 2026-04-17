mod common;

use std::sync::Arc;

use agentwerk::{AgentBuilder, Event, EventKind, SpawnAgentTool};

#[tokio::test]
async fn test() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let (provider, model) = common::build_provider();

    let event_handler = Arc::new(|event: Event| match &event.kind {
        EventKind::ResponseTextChunk { content } => {
            if event.agent_name == "orchestrator" {
                print!("{content}")
            }
        }
        EventKind::ToolCallStart { tool_name, .. } => {
            eprintln!("\n[{}] tool: {tool_name}", event.agent_name)
        }
        EventKind::AgentStart => eprintln!("[{}] started", event.agent_name),
        EventKind::AgentEnd { turns } => eprintln!("[{}] done ({turns} turns)", event.agent_name),
        _ => {}
    });

    let researcher = AgentBuilder::new()
        .name("researcher")
        .model(&model)
        .identity_prompt("You are a research assistant. Answer the given question concisely in 1-2 sentences.")
        .max_turns(1)
        .build()?;

    let spawn_tool = SpawnAgentTool::new()
        .sub_agent(researcher)
        .default_model(&model);

    let output = AgentBuilder::new()
        .provider(provider)
        .model(&model)
        .name("orchestrator")
        .identity_prompt(
            "You coordinate research tasks. Use spawn_agent with agent: \"researcher\" to delegate questions. \
             Summarize the results. Be concise.",
        )
        .instruction_prompt("What is the capital of France? Use the researcher agent to find out, then tell me.")
        .tool(spawn_tool)
        .max_turns(10)
        .event_handler(event_handler)
        .run()
        .await?;

    common::print_result(&output);

    assert!(output.statistics.tool_calls >= 1);

    Ok(())
}
