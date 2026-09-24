# Tools

Add tools to let an agent read and write files, run commands, fetch URLs, manage tasks, and use shared knowledge.

```rust
use agentwerk::tools::{CommandTool, GrepTool, ReadFileTool};

let git = CommandTool("git").allow("git *");

let agent = Agent()
    .tool(ReadFileTool)
    .tool(GrepTool)
    .tool(git);
```

## Built-in tools

The built-in tools cover files, search, commands, web access, events, tasks, and shared knowledge.

### File tools

These tools read, create, and edit files.

| Tool | Purpose |
| --- | --- |
| `ReadFileTool` | Read a file with line numbers, offset, and limit. |
| `WriteFileTool` | Create or overwrite a file. |
| `EditFileTool` | Replace text in a file. |

### Search tools

These tools find files and search their contents.

| Tool | Purpose |
| --- | --- |
| `GlobTool` | Find files by pattern. |
| `GrepTool` | Search file contents by regular expression or code shape. |
| `ListDirectoryTool` | List files and directories. |

### Other built-in tools

These tools run commands, fetch URLs, publish events, manage tasks, and use shared knowledge.

| Tool | Purpose |
| --- | --- |
| `CommandTool` | Grant access to specific commands. |
| `FetchTool` | Fetch a URL and read its body. |
| `EventTool` | Publish an event and optionally finish the current task. |
| `FinishTool` | Write the current task's result and mark it finished. |
| `TaskTool` | Read the Werk and create or edit tasks. |
| `KnowledgeTool` | Write, read, remove, or list pages in a knowledge store. |

## Timeouts

Call `timeout(duration)` to override a tool's limit. Use zero to disable it.

```rust
use std::time::Duration;
use agentwerk::tools::FetchTool;

let quick_fetch = FetchTool.timeout(Duration::from_secs(15));
let patient_fetch = FetchTool.timeout(Duration::ZERO);
```

These defaults apply unless a tool overrides its timeout.

| Tool | Default timeout |
| --- | --- |
| `FetchTool` | 60 seconds. |
| `GrepTool` | 180 seconds. |
| `CommandTool` | The call's `timeout_ms`, or 120 seconds if omitted. |
| All other tools | No timeout. |

## FinishTool

An agent calls `FinishTool` to finish its task and return a result:

```json
{
  "answer": "The configuration is loaded in src/config.rs.",
  "confidence": 0.9
}
```

An agent returns a result through `FinishTool`, which validates any task schema. A non-interactive task without a schema may instead return plain text; [interactive agents](agents.md#interactive-agents) have no `FinishTool` unless you explicitly add `.tool(FinishTool)`.

## CommandTool

Use `CommandTool` to allow or deny specific commands and flags.

```rust
let git = CommandTool("git")
    .allow("git status")
    .allow("git log *")
    .deny("git push*")
    .deny_flag("--force");
```

With an `allow_flag` set, a command carrying any other flag is refused:

```rust
let cargo = CommandTool("cargo")
    .allow("cargo test*")
    .allow_flag("--all-features");
```

## FetchTool

Use `FetchTool` to fetch a URL as text. It sends the user agent `agentwerk/<version>`. `impersonate()` uses a browser's headers and HTTP/2 settings.

```rust
let web = FetchTool.impersonate();
```

## EventTool

Add `EventTool` when an agent needs to publish custom events:

```rust
use agentwerk::tools::EventTool;

let agent = Agent().tool(EventTool);
```

The model supplies a name and optional JSON data:

```json
{
  "name": "...",
  "data": {}
}
```

Events carry the current task and agent context; see [Events](events.md) for hooks and queries. Names are unrestricted, with lowercase snake case conventional. Only `task_finished` completes the current task, using its `data` as the result object:

```json
{
  "name": "task_finished",
  "data": { "answer": "..." }
}
```

Set a [corrective template](tasks.md#corrective-templates) under the event name to customize the text returned to the model.

## Custom tools

Set `concurrent(true)` only for a side-effect-free tool that can safely run beside other calls, then define its description and implementation:

```rust
use agentwerk::{Event, tools::Tool};
use serde_json::Value;

let greet_schema = json!({
    "type": "object",
    "properties": { "name": { "type": "string" } },
    "required": ["name"]
});

let greet = Tool("greet")
    .description("Say hello")
    .schema(greet_schema)
    .concurrent(true)
    .timeout(std::time::Duration::from_secs(5))
    .handler(|input: Value| async move {
        let name = input["name"].as_str().unwrap_or("world");
        Event::tool_call_finished(format!("Hello, {name}!"))
    });
```

Return a `tool_call_failed` event with a string `message` for a failure the model should work around.
