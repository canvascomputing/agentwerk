# Tools

Add tools to let an agent read and write files, run commands, fetch URLs, manage tasks, and use shared knowledge.

```python
from agentwerk import Agent, CommandTool, GrepTool, ReadFileTool

git = CommandTool("git").allow("git *")

agent = (
    Agent()
    .tool(ReadFileTool())
    .tool(GrepTool())
    .tool(git)
)
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
| `GrepTool` | Search file contents by regular expression, or by code shape with `syntax: "code"`. |
| `ListDirectoryTool` | List files and directories. |

### Other built-in tools

These tools run commands, fetch URLs, publish events, manage tasks, and use shared knowledge.

| Tool | Purpose |
| --- | --- |
| `CommandTool` | Give access to specific commands. |
| `FetchTool` | Fetch a URL and read its body. |
| `EventTool` | Publish an event. `task_finished` also completes the current task. |
| `FinishTool` | Write the result for the current task and mark it finished. |
| `TaskTool` | Read the Werk and create or edit tasks. |
| `KnowledgeTool` | Write, read, remove, or list pages in a knowledge store. |

## Timeouts

Call `timeout(seconds)` to override a tool's limit. Use zero to disable it.

```python
from agentwerk import FetchTool

quick_fetch = FetchTool().timeout(15)
patient_fetch = FetchTool().timeout(0)
```

When a Python tool times out, the agent stops waiting, but its worker thread may continue in the background. These defaults apply unless a tool overrides its timeout.

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

An agent returns a result through `FinishTool`, which validates any task schema. A non-interactive task without a schema may instead return plain text; [interactive agents](agents.md#interactive-agents) have no `FinishTool` unless you explicitly add `.tool(FinishTool())`.

## CommandTool

Use `CommandTool` to allow or deny specific commands and flags.

```python
git = (
    CommandTool("git")
    .allow("git status")
    .allow("git log *")
    .deny("git push*")
    .deny_flag("--force")
)
```

With an `allow_flag` set, a command carrying any other flag is refused:

```python
cargo = CommandTool("cargo").allow("cargo test*").allow_flag("--all-features")
```

## FetchTool

Use `FetchTool` to fetch a URL as text. It sends the user agent `agentwerk/<version>`. `impersonate()` uses a browser's headers and HTTP/2 settings.

```python
web = FetchTool().impersonate()
```

## EventTool

Add `EventTool` when an agent needs to publish custom events:

```python
from agentwerk import EventTool

agent = Agent().tool(EventTool())
```

The model supplies a name and optional JSON data:

```json
{
  "name": "...",
  "data": {}
}
```

Events carry the current task and agent context; see [Events](events.md) for hooks and queries. Names are unrestricted, with lowercase snake case conventional. Only `task_finished` completes the current task, using its `data` as the result dictionary:

```json
{
  "name": "task_finished",
  "data": { "answer": "..." }
}
```

Set a [corrective template](tasks.md#corrective-templates) under the event name to customize the text returned to the model.

## Custom tools

Set `concurrent=True` only for a side-effect-free tool that can safely run beside other calls. agentwerk derives model arguments from type annotations: arguments without defaults are required, and lists, dictionaries, tuples, `Literal`, `Optional`, and unions are supported. Use `schema=` when annotations are not enough.

```python
from agentwerk import tool


@tool(concurrent=True, timeout=5)
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
```
