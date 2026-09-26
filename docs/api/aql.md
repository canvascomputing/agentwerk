# AQL

Use Agent Query Language (AQL) to find tasks and events. Pass an AQL string directly, or compile it with `Query(text)` to reuse it.

```rust
// Find tasks labeled `scan`.
werk.find_tasks("scan");

// Find failed tool calls from scan tasks.
werk.find_events("scan AND event.name = tool_call_failed");

// Find the result produced by task `t-3`.
werk.find_results("t-3");
```

## AQL operators

Use these operators to build an AQL expression.

| Operation | Syntax | Meaning |
| --- | --- | --- |
| Match | `field = value`, `field != value` | Include or exclude one exact value. |
| Match | `field IN (a, b)`, `field NOT IN (a, b)` | Include or exclude a list. |
| Presence | `field IS EMPTY`, `field IS NOT EMPTY` | Test whether an optional field has a value. |
| Search | `field ~ text`, `field !~ text` | Include or exclude case-insensitive text. |
| Compare | `field > value`, `>=`, `<`, `<=` | Compare a time field. |
| Combine | `A AND B`, `A OR B`, `NOT A`, `(A OR B)` | Combine or group conditions. |
| Task label | `scan`, `"needs review"` | Short for `task.label = scan`. Quote labels containing spaces or query words. |
| Task ID | `t-3` | Short for `task.id = t-3`. IDs take precedence over labels. |
| Sort | `ORDER BY field DESC` | Sort matches. `ASC` is the default. |

## Queryable fields

Filter tasks and events through these fields.

| Origin | Fields |
| --- | --- |
| Task | `task.id`, `task.label`, `task.status`, `task.pending`, `task.cancelled`, `task.assignee`, `task.content`, `task.result`, `task.errors`, `task.created`, `task.started`, `task.finished`, `task.failed` |
| Event | `event.name`, `event.agent_id`, `event.task_id`, `event.label`, `event.created`, `event.data` |

Mixed task and event queries join each event to its referenced task; events without a task do not match. Results retain event-log order unless `ORDER BY` names a task or event field. Result finders default to finished tasks with results; an explicit status overrides this. `finish_task`, `finish_tasks`, and `cancel_tasks` accept AQL. Cancellation lasts only for the current run; `start()` clears it.
