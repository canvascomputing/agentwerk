# Werk

A `Werk` assigns tasks to agents and collects their results and events.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/werk.gif" width="600" alt="A Werk coordinating agents and tasks" />

```rust
use agentwerk::{Agent, Task, Werk};

let analyst = Agent::from_env()
    .label("analysis");

let writer = Agent::from_env()
    .label("report");

let analysis = Task("Rank all products by value.").label("analysis");
let report = Task("Write up the ranking.").label("report");

let werk = Werk(".agentwerk")?;
werk.add_agent(analyst);
werk.add_agent(writer);

werk.add_task(analysis);
werk.add_task(report);
```

`start()` keeps processing tasks in the background. `finish()` runs tasks and waits for results.

```rust
let task = werk.add_task("Write a report.");

if let Some(answer) = werk.finish_task(task).await {
    println!("{answer}");
}
```

## Construct a Werk

```rust
let werk = Werk::new();
let persisted_werk = Werk(".agentwerk")?;
```

## Configure a Werk

Use these members to set policy, inspect the session directory, and register agents or conditions.

| Member | Purpose |
| --- | --- |
| `set_policy(policy)` | Set execution limits and retry settings. |
| `get_policy()` | Get the active policy. |
| `get_dir()` | Get the session directory. |
| `add_agent(agent)` | Add an agent to the Werk. |
| `add_condition(condition)` | Add a runtime AQL condition and return its ID. |

## Submit and interact

Use these members to create tasks, continue conversations, and set task outcomes.

| Member | Purpose |
| --- | --- |
| `add_task(task)` | Submit a task and return its ID. |
| `add_reply(id, content)` | Add a reply to a task. |
| `edit_replies(id, editor)` | Rewrite a task's replies. |
| `set_task_finished(id, result)` | Finish a task with a result. |
| `set_task_failed(id)` | Mark a task as failed. |

## Observe changes

Use these members to handle events, finished results, and task state changes.

| Member | Purpose |
| --- | --- |
| `on_event(handler)` | Read every event as it is emitted. |
| `on_event_async(handler)` | Read every event in an asynchronous hook. |
| `on_result(handler)` | Read every finished task and its result. |
| `on_result_async(handler)` | Read every finished task and result in an asynchronous hook. |
| `on_task(handler)` | Read task state changes. |
| `on_task_async(handler)` | Read task state changes in an asynchronous hook. |

## Run work

Use these members to start execution and wait for matching results.

| Member | Purpose |
| --- | --- |
| `start()` | Process tasks in the background. |
| `finish_task(query)` | Wait for all matches and return the first result in query order. |
| `finish_tasks(query)` | Wait for matching tasks and return their results. |
| `finish()` | Run tasks and return their results. |

## Cancel work

- `cancel_tasks(query)`: Stop work on matching tasks.
- `cancel()`: Stop work on every task.

## Inspect tasks

Use these members to get tasks directly or find them with AQL.

| Member | Purpose |
| --- | --- |
| `get_task(id)` | Get one task by ID. |
| `get_tasks()` | Get every task in creation order. |
| `find_task(query)` | Get the first matching task. |
| `find_tasks(query)` | Get every matching task. |

## Inspect results and events

Use these members to read completed results and recorded events.

| Member | Purpose |
| --- | --- |
| `get_results()` | Get every finished task result in creation order. |
| `find_result(query)` | Get the first result in query order. |
| `find_results(query)` | Get every result in query order. |
| `find_event(query)` | Get the first recorded event in query order. |
| `find_events(query)` | Get every recorded event in query order. |

## Inspect execution

Use these members to read the run outcome, model choices, token usage, and duration.

| Member | Purpose |
| --- | --- |
| `get_finish_reason()` | Get the reason the last execution ended. |
| `get_model_for_agent(agent_id)` | Get the model used by an agent. |
| `get_input_tokens()` | Get input tokens across finished requests. |
| `get_output_tokens()` | Get output tokens across finished requests. |
| `get_duration()` | Get the elapsed execution duration. |

## Configuration

Use a `Policy` to set turn, token, and time limits, retry behavior, and compaction.

```rust
let policy = Policy {
    max_turns: Some(40),
    max_time: Some(std::time::Duration::from_secs(300)),
    ..Default::default()
};

werk.set_policy(policy);
```

The policy exposes these fields.

| Field | Purpose |
| --- | --- |
| `max_turns` | Limit the total number of turns. |
| `max_time` | Limit the total elapsed duration. |
| `max_input_tokens` | Limit the total input tokens. |
| `max_output_tokens` | Limit the total output tokens. |
| `max_request_tokens` | Limit the output tokens of a single request. |
| `max_schema_retries` | Limit consecutive failed tool calls or silent replies. A successful call resets the count. |
| `max_request_retries` | Limit how often a failing request is retried. |
| `request_retry_delay` | Set the base delay for exponential backoff between request retries. |
| `compaction_threshold` | Compact once the next request would fill this share of the window. |

`set_policy(policy)` replaces the whole configuration, and `get_policy()` reads it back. A violated limit emits `Event::POLICY_VIOLATED`. `compaction_threshold` is the exception, see [Compaction](#compaction).

## Compaction

Compaction replaces older messages with a summary as a task approaches the model's context limit or after the provider reports an overflow.

```rust
let policy = Policy {
    compaction_threshold: Some(0.7),
    ..Default::default()
};

werk.set_policy(policy);
```

`compaction_threshold` is a fraction of the model's context window, `0.85` by default. Reaching it summarizes older messages so the agent can continue. Compaction also runs after the LLM provider reports an overflow; `compaction_started`, `compaction_progress`, `compaction_finished`, and `compaction_failed` report each step, see [Events](events.md).

```rust
werk.on_event(|_, event| {
    if event.get_name() == Event::COMPACTION_FINISHED {
        eprintln!("[{}] compacted {}", event.get_task_id(), event.get_data()["trigger"]);
    }
});
```

Each compaction event carries the trigger: `proactive` before a context-window error or `reactive` after one. A failure also carries a stable `kind` and human-readable `message`.

## Sessions

Use a session directory to save tasks, replies, and recorded events, then resume them in a later run.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/sessions.gif" width="600" alt="A persisted agentwerk session" />

The session directory is `./.agentwerk` by default.

```rust
let werk = Werk(".agentwerk")?;
werk.add_agent(my_agent);
werk.start();
```

Sessions use this file layout.

```
.agentwerk/
├── events.jsonl                        recorded events, one per line
├── tasks/
│   └── t-1/
│       ├── task.json                   task metadata and input
│       ├── result.json                 task result
│       ├── replies.jsonl               model messages, one per line
│       └── outputs/<tool_use_id>.txt   full tool outputs
└── knowledge/
    ├── pages/<slug>.md                 knowledge pages
    └── index.md                        knowledge index
```
