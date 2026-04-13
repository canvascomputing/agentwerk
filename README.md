<div align="center">

# 🤖 `agent`

```
                             __   
    .---.-.-----.-----.-----|  |_ 
    |  _  |  _  |  -__|     |   _|
    |___._|___  |_____|__|__|____|
          |_____|                 

  A minimal Rust crate that gives any
  application agentic capabilities.
```

</div>

<p align="center"><code>Agentic execution loop</code> · <code>Basic tool implementations</code> · <code>Sub-agent orchestration</code> · <code>Anthropic, Mistral, OpenAI integration</code> · <code>Schema-based output</code> · <code>Cost tracking</code></p>

## Use Cases

Every agentic application, like OpenClaw or Claude Code, reimplements the same core functionality. This crate extracts that shared foundation into a minimal, dependency-light library.

Here are example applications built with this project.

> Consider setting your LLM provider's environment variables for key, model or base URL.

### [Project Scanner](crates/use-cases/src/project_scanner/)

Scans a directory and outputs a JSON summary with project description and languages used.

```bash
make use-case name=project-scanner -- ./
```

Output:
```json
{
  "summary": "A minimal Rust framework for building agentic LLM applications with tool use",
  "languages": ["Rust"]
}
```

### [Deep Research](crates/use-cases/src/deep_research/)

Spawns three researcher sub-agents in parallel, then aggregates their findings into a structured decision. Requires `BRAVE_API_KEY` for web search.

```bash
make use-case name=deep-research args="What constitutes a good life?"
```

Output:
```json
{
  "title": "What Constitutes a Good Life: A Multi-Perspective Analysis",
  "research": "A good life emerges from the convergence of philosophical wisdom, scientific research, and cultural understanding. Key elements include meaningful relationships and social connections, a sense of purpose and personal growth, physical and mental well-being, contributing to something beyond oneself, and living in accordance with personal values. While cultural contexts vary, common themes across traditions emphasize virtue, balance, gratitude, and the cultivation of both inner fulfillment and positive impact on others."
}
```

## API

Create an `LlmProvider`, define your `Tool`s, wire them into an `AgentBuilder`, call `agent.run()` with an `InvocationContext`, stream `Event`s during execution, and get back an `AgentOutput`.

### AgentBuilder

Configure an agent with a model, prompt, tools, and guardrails.

```rust
use agent::AgentBuilder;

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("Help with {topic}")   // {key} replaced from template_variables
    .tool(ReadFileTool)
    .tool(GrepTool)
    .max_turns(10)
    .max_budget(1.0)                      // USD spend limit
    .output_schema(json!({...}))          // force structured JSON response
    .sub_agent(researcher)                // available to SpawnAgentTool
    .build()?;

let explorer = AgentBuilder::new()
    .name("explorer")
    .model("claude-haiku-4-5-20251001")   // fast model for search
    .system_prompt("Search the codebase.")
    .read_only()                          // minimal prompts, lower max_tokens
    .build()?;

let child = AgentBuilder::new()
    .name("child")                        // no .model() — inherits from parent
    .system_prompt("Help the parent.")
    .build()?;
```

Agents include default behavior prompts for task execution, tool usage, action safety, and output efficiency. Replace any default with `behavior_prompt()`:

```rust
use agent::BehaviorPrompt;

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are helpful.")
    .behavior_prompt(BehaviorPrompt::TaskExecution, "- Follow instructions exactly.")
    .build()?;
```

Add environment context, memory, and instruction files:

```rust
use agent::EnvironmentContext;

let agent = AgentBuilder::new()
    .name("assistant")
    .model("claude-sonnet-4-20250514")
    .system_prompt("You are helpful.")
    .environment_context(&EnvironmentContext::collect(&cwd))
    .instruction_files(&cwd)
    .memory(&memory_dir)
    .build()?;
```

### InvocationContext

Runtime state for a single agent run. Pass it to `agent.run(ctx)`.

```rust
use agent::InvocationContext;

let ctx = InvocationContext::new(provider)
    .prompt("Find all TODOs");

let ctx = InvocationContext::new(provider)
    .prompt("Analyze {topic}")
    .template_var("topic", json!("rust"))
    .model("claude-sonnet-4-20250514")           // sub-agents using inherit_model() resolve to this
    .working_directory(PathBuf::from("./src"))
    .event_handler(Arc::new(|e| { /* ... */ }))
    .cancel_signal(cancel)
    .session_store(Arc::new(Mutex::new(store)))
    .command_queue(Arc::new(CommandQueue::new()));
```

### AgentOutput

The result of `agent.run()` — text, structured data, and token usage.

```rust
let output = agent.run(ctx).await?;

println!("{}", output.response_raw);           // free-form LLM text
println!("{:?}", output.response);             // Some(Value) if output_schema was set
println!("{}", output.statistics.input_tokens); // accumulated token counts
```

With `.output_schema()`, the agent returns validated JSON in `response`:

```rust
AgentBuilder::new()
    .output_schema(json!({
        "type": "object",
        "properties": { "category": { "type": "string" } },
        "required": ["category"]
    }))
    ...

output.response.unwrap()["category"]  // "billing"
```

### Event

Lifecycle and progress notifications emitted during execution.

```rust
use agent::Event;

ctx.event_handler = Arc::new(|event| match &event {
    Event::TextChunk { content, .. } => print!("{content}"),
    Event::RequestStart { agent_name, model } => eprintln!("{agent_name} calling {model}..."),
    Event::RequestEnd { agent_name, .. } => eprintln!("{agent_name} response received"),
    Event::ToolCallStart { tool_name, input, .. } => eprintln!("[{tool_name}] {input}"),
    Event::ToolCallEnd { tool_name, is_error, .. } if *is_error => eprintln!("[error] {tool_name}"),
    Event::AgentEnd { agent_name, turns } => eprintln!("{agent_name} done in {turns} turns"),
    Event::TokenUsage { model, usage, .. } => eprintln!("{model}: {} tokens", usage.input_tokens),
    _ => {}
});
```

### LlmProvider

Connect to any LLM. Providers own a `reqwest::Client` for connection pooling and SSE streaming.

```rust
use agent::{AnthropicProvider, MistralProvider, LiteLlmProvider};

let provider = AnthropicProvider::from_api_key(key);
let provider = MistralProvider::from_api_key(key);
let provider = LiteLlmProvider::from_api_key(key);

let client = reqwest::Client::new();                        // share a connection pool
let provider = AnthropicProvider::new(key, client);
```

### Tool

Define what the agent can do. Read-only tools run concurrently.

```rust
use agent::{ToolBuilder, ToolResult};

let tool = ToolBuilder::new("greet", "Say hello")
    .schema(json!({...}))
    .read_only(true)
    .handler(|input, ctx| Box::pin(async move {
        Ok(ToolResult::success("Hello!"))
    }))
    .build();
```

Built-in tools:

- `ReadFileTool`, `WriteFileTool`, `EditFileTool` — file operations
- `GlobTool`, `GrepTool` — search by pattern or content
- `ListDirectoryTool` — directory listing with type and size
- `BashTool` — shell command execution
- `ToolSearchTool` — discover available tools by keyword
- `SpawnAgentTool` — delegate work to a sub-agent

## Development

```bash
make                   # build
make test              # unit tests
make test_integration  # integration tests (requires LLM provider)
make fmt               # format
make use-case          # list use cases
make litellm           # start LiteLLM proxy
```

### Environment

Auto-detect the provider from environment variables:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Use Anthropic directly |
| `ANTHROPIC_BASE_URL` | API URL (default: `https://api.anthropic.com`) |
| `ANTHROPIC_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
| `MISTRAL_API_KEY` | Use Mistral directly |
| `MISTRAL_BASE_URL` | API URL (default: `https://api.mistral.ai`) |
| `MISTRAL_MODEL` | Model (default: `mistral-medium-2508`) |
| `LITELLM_API_KEY` | Auth key (optional) |
| `LITELLM_API_URL` | Use LiteLLM proxy (default: `http://localhost:4000`) |
| `LITELLM_MODEL` | Model (default: `claude-sonnet-4-20250514`) |
