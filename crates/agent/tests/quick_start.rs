use std::sync::Arc;

use agent::{
    AgentBuilder, AnthropicProvider, LiteLlmProvider, LlmProvider, ReadFileTool, GlobTool,
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
    let Some((provider, model)) = build_provider() else {
        eprintln!("SKIPPED: no provider");
        return Ok(());
    };

    let output = AgentBuilder::new()
        .provider(provider)
        .model(&model)
        .instruction_prompt("Find all Rust source files and describe what this project does.")
        .tool(ReadFileTool)
        .tool(GlobTool)
        .max_turns(5)
        .run()
        .await?;

    println!("{}", output.response_raw);

    assert!(!output.response_raw.is_empty());
    assert!(output.statistics.tool_calls >= 1);
    Ok(())
}
