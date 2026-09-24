# Events

Use events to inspect a run. Publish custom events through the Werk, adding agent or task context when relevant:

```rust
use agentwerk::Event;
use serde_json::json;

let document_indexed = Event("document_indexed")
    .data(json!({ "documents": 42 }))
    .task_id("t-1")
    .agent_id("indexer-1");

let index_refreshed = Event("index_refreshed");

werk.emit_event(document_indexed);
werk.emit_event(index_refreshed);
```

`Werk::emit_event` does not change task status; use [EventTool](tools.md#eventtool) for model-driven completion through `task_finished`. Events are saved to `.agentwerk/events.jsonl`, except `text_chunk_received`, and can be queried with [AQL](aql.md).

## Event catalog

Built-in events are grouped by the part of execution that emits them.

### Run events

These events describe the start, end, or forced stop of a run.

| Event | Emitted when |
| --- | --- |
| `run_started` | Execution began. |
| `run_finished` | Execution ended, carrying its outcome. |
| `policy_violated` | A limit was breached and execution stopped. |

### Task events

These events describe task creation, execution, completion, and validation.

| Event | Emitted when |
| --- | --- |
| `task_started` | An agent claimed a task. |
| `task_created` | A task was added to the Werk. |
| `task_finished` | A task finished successfully, carrying its result when it has one. |
| `task_failed` | A task failed. |
| `turn_started` | The agent began another turn on its task. |
| `schema_retried` | A tool call or result the model created was invalid. |

### Provider events

These events describe model requests and streamed text.

| Event | Emitted when |
| --- | --- |
| `request_started` | A request went out to the model. |
| `request_finished` | A request finished and reported its token usage. |
| `request_failed` | A request failed and was not retried. |
| `prompt_render_failed` | A prompt or template could not render. |
| `request_retried` | A temporary LLM provider error triggered a retry. |
| `text_chunk_received` | Part of the reply arrived. |

### Tool events

These events describe tool approval, repair, execution, and failure.

| Event | Emitted when |
| --- | --- |
| `tool_call_declined` | A tool call proposed by the model was declined. |
| `tool_call_repaired` | A tool call or value the model created was invalid and was corrected. |
| `tool_call_started` | A tool invocation began, carrying its registered name, call ID, and raw input. |
| `tool_call_finished` | A tool invocation finished. |
| `tool_call_failed` | A tool invocation failed but the task continues. |

### Knowledge events

These events describe operations against the shared knowledge store.

| Event | Emitted when |
| --- | --- |
| `knowledge_written` | A page was written. |
| `knowledge_read` | A page was read. |
| `knowledge_removed` | A page was removed. |
| `knowledge_listed` | The pages were listed. |
| `knowledge_failed` | An action against the store did not go through. |

### Compaction events

These events describe each stage of context compaction.

| Event | Emitted when |
| --- | --- |
| `compaction_started` | Compaction is about to rewrite the older messages. |
| `compaction_progress` | Compaction finished part of the work. |
| `compaction_finished` | Compaction replaced the older messages. |
| `compaction_failed` | Compaction could not finish. |

### Custom events

- Application-defined event: Published with `emit_event` under the name chosen by the application.

## Event objects

Use these members to create a custom event and inspect its context.

| Member | Purpose |
| --- | --- |
| `Event(name)` | Create a custom event. |
| `get_name()` | Read the event name. |
| `get_data()` | Read the event payload. |
| `get_task_id()` | Read the associated task ID. |
| `get_agent_id()` | Read the associated agent ID. |
| `get_label()` | Read the associated task's label. |
| `get_created_at()` | Read the timestamp in epoch milliseconds. |

## Hooks

Use hooks to run code when an event arrives, a task finishes, or a task changes.

```rust
werk.on_event(|_, event| eprintln!("event: {}", event.get_name()));
werk.on_result(|_, task, result| println!("{}: {result}", task.get_id()));
werk.on_task(|_, event, task| eprintln!("{}: {}", task.get_id(), event.get_name()));
```

`on_result` is synchronous; keep it brief or use `on_result_async` for awaited work. Async hooks run only while a completion method waits and finish before it returns; `start()` alone does not run them. Calling `finish`, `finish_task`, or `finish_tasks` inside one can deadlock. Without an event hook, `event::default_logger()` logs events.
