# agentwerk API

Use this guide to configure a Rust agent, add tools and tasks, run several agents together, and share knowledge.

## Agents

Create an agent with a role, a model, and the tools it can call.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/agents.gif" width="600" alt="An agent processing tasks" />

```rust
use agentwerk::tools::ReadFileTool;

let agent = Agent::from_env()
    .role("You are a release manager who prepares release notes.")
    .tool(ReadFileTool);

agent.add_task("Read CHANGELOG.md and summarize the entries added since the last release.");

let results = agent.finish().await;
```

<details>
<summary>Agent reference</summary>

| Area | Method | Description |
|------|--------|-------------|
| **Configure** | `role(role)` | Define who the agent is and how it should work. |
| | `tool(tool)` | Register a tool the agent may call. |
| | `tools(tools)` | Register several tools together. |
| | `label(label)` | Restrict the agent to tasks carrying this label. |
| | `dir(dir)` | Set the directory the agent can access. |
| | `template(key, value)` | Set a shared template value. |
| | `templates(variables)` | Set several shared template values together. |
| | `directive(key, template)` | Set a directive template. |
| | `directives(overrides)` | Set several directive templates together. |
| | `knowledge(store)` | Share a knowledge store and register its `KnowledgeTool`. |
| | `interactive()` | Keep a task in progress while waiting for new instructions. |
| **Work** | `add_task(task)` | Submit a task and return its ID. |
| | `start()` | Process tasks in the background. |
| | `finish_task(query)` | Wait for all matches and return the first result in query order. |
| | `finish_tasks(query)` | Wait for matching tasks and return their results. |
| | `finish()` | Run tasks and return their results. |
| | `get_id()` | Get the agent's unique identifier. |

</details>

### Providers

Send an agent's model requests to Anthropic, OpenAI, Mistral, or a LiteLLM proxy.

```rust
use agentwerk::providers::Anthropic;

let agent = Agent::new()
    .provider(Anthropic::new(key))
    .model("claude-sonnet-4-20250514");
```

You can also read the model or provider individually: `.provider(Provider::from_env()?)` or `.model(Model::from_env()?)`.

Set a model's context window or reasoning level when the defaults do not fit. Claude, GPT, Mistral, and Qwen families have built-in settings.

Configure a custom model:

```rust
use agentwerk::providers::{Model, ReasoningEffort};

let agent = Agent::new().model(
    Model::new("my-local-model")
        .context_window(128_000)
        .reasoning_effort(ReasoningEffort::High),
);
```

<details>
<summary>Provider reference</summary>

Provider methods:

| Method | Description |
|--------|-------------|
| `provider(provider)` | Set the LLM provider. |
| `model(model)` | Set the model. |
| `Agent::from_env()` | Read the provider and model from environment variables. |
| `verify(model)` | Verify that the provider can answer with a model. |
| `Anthropic::new(key).base_url(url).timeout(duration)` | Configure an Anthropic endpoint. OpenAI, Mistral, and LiteLLM expose the same methods. |

Provider environment variables:

| Variable | Description |
|----------|-------------|
| `LITELLM_PROVIDER` | Choose `anthropic`, `mistral`, `openai`, or `litellm` outright, ahead of the keys below. |
| `LITELLM_API_KEY`, `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | Authenticate with that vendor. The first one set picks the LLM provider, in this order. |
| `LITELLM_BASE_URL`, `MISTRAL_BASE_URL`, `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL` | Set a different API address for that vendor. |
| `SSL_CERT_FILE`, `SSL_CERT_DIR` | Trust these CA certificates instead of the built-in root store. |

Model methods:

| Method | Description |
|--------|-------------|
| `context_window(size)` | Set the context window size for a model. |
| `get_context_window()` | Get the configured window size. |
| `reasoning_effort(effort)` | Set the reasoning level. |
| `get_reasoning_effort()` | Get the configured effort. |

Model environment variables:

| Variable | Description |
|----------|-------------|
| `MODEL` | Set the model name. |
| `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `MISTRAL_MODEL`, `LITELLM_MODEL` | Set the model for the detected provider when `MODEL` is unset. |
| `MODEL_CONTEXT_WINDOW` | Set the context window size in tokens. |

</details>

<a id="interactive"></a>

### Interactive agents

Call `interactive()` to keep a task open for follow-up replies. Interactive agents have no completion tool by default.

```rust
let agent = Agent::from_env().interactive();
let id = agent.add_task("Where does the configuration get loaded?");

let werk = agent.start();
werk.finish().await;

werk.add_reply(&id, "And which environment variables override it?");
werk.finish().await;
```

<details>
<summary>Interactive agent reference</summary>

Replies pause the task in `in_progress`, and completion methods return when it pauses. Use `add_reply(id, content)` to resume and `set_task_finished(id, result)` to end the conversation. Intermediate replies arrive as [events](#events). `on_result` receives the final result.

</details>

## Tasks

Put each piece of work in a task. The task records its status and result.

```rust
use agentwerk::Task;

let task = Task::labeled("review", "Review the release notes.");
werk.add_task(task);
```

<details>
<summary>Task reference</summary>

| Area | Member | Description |
|------|--------|-------------|
| **Identity** | `get_id()` | Get the task ID in the form `t-N`. |
| | `get_task()` | Get the work assigned to the task. |
| | `get_label()` | Get the task's label. |
| | `get_reporter()` | Get the ID of the agent that created the task. |
| | `get_assignee()` | Get the ID of the agent that claimed the task. |
| **Outcome** | `get_status()` | Get the task's current status. |
| | `is_todo()` | Check whether the task is waiting to be claimed. |
| | `is_in_progress()` | Check whether an agent is working on the task. |
| | `is_finished()` | Check whether the task finished. |
| | `is_failed()` | Check whether the task failed. |
| | `is_pending()` | Check whether the task has work in this run. |
| | `is_cancelled()` | Check whether this run excluded the task from scheduling. |
| | `get_result()` | Get the task's result. |
| | `get_errors()` | Get failures recorded against the task as events. |
| | `get_replies()` | Get messages exchanged with the model. |
| | `get_schema()` | Get the task's optional result schema. |
| **Timestamps** | `get_created_at()` | Get the creation time in milliseconds. |
| | `get_started_at()` | Get the claim time in milliseconds. |
| | `get_finished_at()` | Get the finish time in milliseconds. |
| | `get_failed_at()` | Get the failure time in milliseconds. |

</details>

### Templates

Use templates to insert shared values, task results, and event data into roles and tasks. The [prompt skill](skills/prompt/SKILL.md) provides a compact template for writing agent roles.

Define an agent with template expressions, then set their values before adding its task:

```rust
use agentwerk::{Agent, Task};

let writer = Agent::from_env()
    .label("report")
    .role("Write for {{ company }} using:\n{{ results: research }}");

werk.add_agent(writer);
werk.set_template("company", "Canvas Computing");
werk.finish_tasks("research").await;
werk.add_task(Task::labeled("report", "Write the board report."));
```

<details>
<summary id="template-reference">Template reference</summary>

agentwerk renders templates in the role and task just before each task's first model request. Newly added tasks use the latest template values and results.

Use `{{ context }}` in a prompt to include the current task and execution limits:

```markdown
- Task: t-7
- Date: 2026-05-06
- Working directory: /Users/caro
- Platform: darwin 25.1.0
- Turns remaining: 8
- Input tokens remaining: 95000
- Output tokens remaining: 12000
- Time remaining: 240s
```

Each context field is also available separately: `{{ task_id }}`, `{{ date }}`, `{{ dir }}`, `{{ platform }}`, `{{ os_version }}`, `{{ turns_remaining }}`, `{{ input_tokens_remaining }}`, `{{ output_tokens_remaining }}`, and `{{ time_remaining }}`.

| Expression | Output |
|---|---|
| `{{ name }}` | The value assigned to `name`. |
| `{{ name \| JSONPath }}` | A value selected from the template variable after parsing it as JSON. |
| `{{ result: AQL }}` | The first matching result. Strings appear as text and other values as compact JSON. |
| `{{ results: AQL }}` | Matching results as a compact JSON array. |
| `{{ result: AQL \| JSONPath }}` | A value selected from the first result. |
| `{{ results: AQL \| JSONPath }}` | A value selected from the array of matching results. |
| `{{ task: AQL }}` | The first matching task as compact JSON. |
| `{{ tasks: AQL }}` | Matching tasks as a compact JSON array. |
| `{{ task: AQL \| JSONPath }}` | A value selected from the first matching task. |
| `{{ tasks: AQL \| JSONPath }}` | A value selected from the array of matching tasks. |
| `{{ event: AQL }}` | The first matching event as compact JSON. |
| `{{ events: AQL }}` | Matching events as a compact JSON array. |
| `{{ event: AQL \| JSONPath }}` | A value selected from the first matching event. |
| `{{ events: AQL \| JSONPath }}` | A value selected from the array of matching events. |

`null`, empty arrays, and unmatched selectors render to an empty string.

| Record | JSONPath properties |
|---|---|
| Task | `task`, `label`, `schema`, `id`, `status`, `reporter`, `assignee`, `created_at`, `started_at`, `finished_at`, `failed_at` |
| Event | `name`, `directive`, `data`, `task_id`, `agent_id`, `label`, `created_at` |

`task` is the original JSON input. Unset `label`, `schema`, `assignee`, and `directive` properties are omitted. Task results, errors, replies, and cancellation state are not serialized.

For example, given this `research` result:

```json
{
  "company": {"name": "Canvas Computing"},
  "findings": [{"summary": "one"}, {"summary": "two"}]
}
```

This template:

```text
{{ result: research | company.name }}
```

Renders as:

```text
Canvas Computing
```

Select each task input or event's data:

```text
{{ tasks: research | [*].task }}
{{ events: event.name = tool_call_failed | [*].data }}
```

`research` is shorthand for `task.label = research`.

| Path | Selects |
|---|---|
| `company.name` | A nested field. |
| `metadata."build-id"` | A field that requires JSON quoting. |
| `findings[0]`, `findings[-1]` | An array element. |
| `findings[1:4]`, `findings[::-1]` | An array slice. |
| `findings[*].summary` | The `summary` field from each array element. |
| `authors.*.name` | The `name` field from each object value, in unspecified order. |
| `groups[].members` | The `members` field after flattening one array level. |

Missing fields, incompatible types, and out-of-range indexes produce `null`. Wildcards, slices, and flattening omit null values when more path steps follow. Filters, comparisons, logical expressions, literals, multi-selects, functions, and additional pipes are not supported.

`{ name }` stays unchanged. To output the literal text `{{ name }}`, write `{{{{ name }}}}`.

</details>

### Schemas

Attach a `Schema` when a task must return a specific JSON object structure.

```rust
use agentwerk::schemas::Schema;

let schema = Schema::new(json!({
    "type": "object",
    "properties": { "title": { "type": "string" } },
    "required": ["title"]
}))?;

werk.add_task(Task::new("Write a report.").schema(schema));
```

<details>
<summary>Schema reference</summary>

agentwerk corrects common result-formatting mistakes, such as a quoted number or a nested object encoded as JSON text. Schema-bound results must be objects. Remaining schema violations trigger a retry, subject to `max_schema_retries`. Without a schema, a task may return any JSON value.

Use shallow, focused schemas for small models. Split complex work into tasks with separate schemas.

| Area | Method | Description |
|------|--------|-------------|
| **Schema** | `Schema::new(document)` | Create a schema. |
| | `validate(value)` | Return the validated value and JSON pointers to repaired values, or report violations. |
| | `get_raw_schema()` | Read the JSON Schema document the schema was built from. |

</details>

### Directives

Directives tell the model what to do when an operation fails or returns invalid data. Override their wording for your model or environment.

```rust
let agent = Agent::from_env()
    .directive("grep_failed", "The search did not run. Narrow `path`.")
    .directives([
        ("tool_timed_out", "Reduce the command scope."),
        ("cache_miss", "No cache entry exists for {{ path }}."),
    ]);
```

<details>
<summary>Directive reference</summary>

Built-in keys override recovery text. Keys without overrides retain their defaults. Templates accept runtime values such as `{{ detail }}`, `{{ attempt }}`, and `{{ path }}`. Expressions without a value remain unchanged.

See [prompts/directives](https://github.com/canvascomputing/agentwerk/tree/main/crates/agentwerk/src/prompts/directives) for the built-in text.

</details>

## Tools

Add tools to let an agent read and write files, run commands, fetch URLs, manage tasks, and use shared knowledge.

```rust
use agentwerk::tools::{CommandTool, GrepTool, ReadFileTool};

let agent = Agent::new()
    .tool(ReadFileTool)
    .tool(GrepTool)
    .tool(CommandTool::new("git").allow("git *"));
```

<details>
<summary>Tool reference</summary>

| Area | Tool | Description |
|------|------|-------------|
| **File** | `ReadFileTool` | Read a file with line numbers, offset, and limit. |
| | `WriteFileTool` | Create or overwrite a file. |
| | `EditFileTool` | Replace text in a file. |
| **Search** | `GlobTool` | Find files by pattern. |
| | `GrepTool` | Search file contents by regular expression or code shape. |
| | `ListDirectoryTool` | List files and directories. |
| **Command** | `CommandTool` | Grant access to specific commands. |
| **Web** | `FetchTool` | Fetch a URL and read its body. |
| **Events** | `EventTool` | Publish an event and optionally finish the current task. |
| **Tasks** | `FinishTool` | Write the current task's result and mark it finished. |
| | `TaskTool` | Read the Werk and create or edit tasks. |
| **Knowledge** | `KnowledgeTool` | Write, read, remove, or list pages in a knowledge store. |

</details>

#### FinishTool

An agent calls `FinishTool` to finish its task and return a result:

```json
{
  "answer": "The configuration is loaded in src/config.rs.",
  "confidence": 0.9
}
```

To return a result, the agent must call `FinishTool`. If the task has a result schema, the tool validates the object against it. For a non-interactive task without a schema, the agent can instead finish by responding with plain text.

[Interactive agents](#interactive-agents) are the exception: they have no `FinishTool` unless you add one explicitly with `.tool(FinishTool)`.

#### Timeouts

Call `timeout(duration)` to override a tool's limit. Use zero to disable it.

```rust
use std::time::Duration;
use agentwerk::tools::FetchTool;

let quick_fetch = FetchTool::new().timeout(Duration::from_secs(15));
let patient_fetch = FetchTool::new().timeout(Duration::ZERO);
```

<details>
<summary>Timeout reference</summary>

| Tool | Default timeout |
|------|-----------------|
| `FetchTool` | 60 seconds |
| `GrepTool` | 180 seconds |
| `CommandTool` | The call's `timeout_ms`, or 120 seconds if omitted |
| All other tools | None |

</details>

#### EventTool

Add `EventTool` when an agent needs to publish custom events:

```rust
use agentwerk::tools::EventTool;

let agent = Agent::new().tool(EventTool);
```

The model supplies a name and optional JSON data:

```json
{
  "name": "...",
  "data": {}
}
```

Events carry the current task and agent context. See [Events](#events) for hooks and queries. Names are unrestricted. Lowercase snake case is conventional.

Only `task_finished` completes the current task. Its `data` is the result object:

```json
{
  "name": "task_finished",
  "data": { "answer": "..." }
}
```

Use [Directives](#directives) to customize the acknowledgement sent to the model.

#### CommandTool

Use `CommandTool` to allow or deny specific commands and flags.

```rust
let git = CommandTool::new("git")
    .allow("git status")
    .allow("git log *")
    .deny("git push*")
    .deny_flag("--force");
```

With an `allow_flag` set, a command carrying any other flag is refused:

```rust
let cargo = CommandTool::new("cargo")
    .allow("cargo test*")
    .allow_flag("--all-features");
```

#### FetchTool

Use `FetchTool` to fetch a URL as text. It sends the user agent `agentwerk/<version>`. `impersonate()` uses a browser's headers and HTTP/2 settings.

```rust
let web = FetchTool::new().impersonate();
```

#### Custom tools

Mark a custom tool as concurrent with `concurrent(true)` only when it has no side effects and can safely run beside other calls.

Describe the tool, then hand it the code it runs:

```rust
use agentwerk::{Event, tools::Tool};
use serde_json::Value;

let greet = Tool::new("greet")
    .description("Say hello")
    .schema(json!({
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }))
    .concurrent(true)
    .timeout(std::time::Duration::from_secs(5))
    .handler(|input: Value| async move {
        let name = input["name"].as_str().unwrap_or("world");
        Event::tool_call_finished(format!("Hello, {name}!"))
    });
```

Return a `tool_call_failed` event with a string `message` for a failure the model should work around.

## Werk

A `Werk` assigns tasks to agents and collects their results and events.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/werk.gif" width="600" alt="A Werk coordinating agents and tasks" />

```rust
use agentwerk::{Agent, Task, Werk};

let analyst = Agent::from_env()
    .label("analysis");

let writer = Agent::from_env()
    .label("report");

let werk = Werk::new();
werk.add_agent(analyst).add_agent(writer);

werk.add_task(Task::labeled("analysis", "Rank all products by value."));
werk.add_task(Task::labeled("report", "Write up the ranking."));
```

`start()` keeps processing tasks in the background. `finish()` runs tasks and waits for results.

```rust
let task = werk.add_task("Write a report.");

if let Some(answer) = werk.finish_task(task).await {
    println!("{answer}");
}
```

<details>
<summary>Werk reference</summary>

| Area | Method | Description |
|------|--------|-------------|
| **Configure** | `set_policy(policy)` | Set execution limits and retry settings. |
| | `get_policy()` | Get the active policy. |
| | `set_dir(dir)` | Set the session directory. |
| | `get_dir()` | Get the session directory. |
| | `add_agent(agent)` | Add an agent to the Werk. |
| | `add_condition(condition)` | Add a runtime AQL condition and return its ID. |
| **Submit and interact** | `add_task(task)` | Submit a task and return its ID. |
| | `add_reply(id, content)` | Add a reply to a task. |
| | `edit_replies(id, editor)` | Rewrite a task's replies. |
| | `set_task_finished(id, result)` | Finish a task with a result. |
| | `set_task_failed(id)` | Mark a task as failed. |
| **Observe** | `on_event(handler)` | Read every event as it is emitted. |
| | `on_event_async(handler)` | Read every event in an asynchronous hook. |
| | `on_result(handler)` | Read every finished task and its result. |
| | `on_result_async(handler)` | Read every finished task and result in an asynchronous hook. |
| | `on_task(handler)` | Read task state changes. |
| | `on_task_async(handler)` | Read task state changes in an asynchronous hook. |
| **Run** | `start()` | Process tasks in the background. |
| | `finish_task(query)` | Wait for all matches and return the first result in query order. |
| | `finish_tasks(query)` | Wait for matching tasks and return their results. |
| | `finish()` | Run tasks and return their results. |
| **Cancel** | `cancel_tasks(query)` | Stop work on matching tasks. |
| | `cancel()` | Stop work on every task. |
| **Inspect tasks** | `get_task(id)` | Get one task by ID. |
| | `get_tasks()` | Get every task in creation order. |
| | `find_task(query)` | Get the first matching task. |
| | `find_tasks(query)` | Get every matching task. |
| **Inspect results and events** | `get_results()` | Get every finished task result in creation order. |
| | `find_result(query)` | Get the first result in query order. |
| | `find_results(query)` | Get every result in query order. |
| | `find_event(query)` | Get the first recorded event in query order. |
| | `find_events(query)` | Get every recorded event in query order. |
| **Inspect execution** | `get_finish_reason()` | Get the reason the last execution ended. |
| | `get_model_for_agent(agent_id)` | Get the model used by an agent. |
| | `get_input_tokens()` | Get input tokens across finished requests. |
| | `get_output_tokens()` | Get output tokens across finished requests. |
| | `get_duration()` | Get the elapsed execution duration. |

</details>

### AQL

Use Agent Query Language (AQL) to find tasks and events. Pass an AQL string
directly, or compile it with `Query::new` to reuse it.

```rust
// Find tasks labeled `scan`.
werk.find_tasks("scan");

// Find failed tool calls from scan tasks.
werk.find_events("scan AND event.name = tool_call_failed");

// Find the result produced by task `t-3`.
werk.find_results("t-3");
```

<details>
<summary>AQL reference</summary>

Use these operators to build an AQL expression.

| Operation | Syntax | Meaning |
|-----------|--------|---------|
| **Match** | `field = value`, `field != value` | Include or exclude one exact value. |
| | `field IN (a, b)`, `field NOT IN (a, b)` | Include or exclude a list. |
| **Presence** | `field IS EMPTY`, `field IS NOT EMPTY` | Test whether an optional field has a value. |
| **Search** | `field ~ text`, `field !~ text` | Include or exclude case-insensitive text. |
| **Compare** | `field > value`, `>=`, `<`, `<=` | Compare a time field. |
| **Combine** | `A AND B`, `A OR B`, `NOT A`, `(A OR B)` | Combine or group conditions. |
| **Task label** | `scan`, `"needs review"` | Short for `task.label = scan`. Quote labels containing spaces or query words. |
| **Task ID** | `t-3` | Short for `task.id = t-3`. IDs take precedence over labels. |
| **Sort** | `ORDER BY field DESC` | Sort matches. `ASC` is the default. |

Filter tasks and events through these fields.

| Origin | Fields |
|--------|--------|
| **Task** | `task.id`, `task.label`, `task.status`, `task.pending`, `task.cancelled`, `task.assignee`, `task.input`, `task.result`, `task.errors`, `task.created`, `task.started`, `task.finished`, `task.failed` |
| **Event** | `event.name`, `event.agent_id`, `event.task_id`, `event.label`, `event.created`, `event.data` |

Queries using both namespaces match events with their referenced tasks. Events without an existing task do not match. Joined matches default to event-log order. `ORDER BY` accepts task or event fields.

Result finders return raw results where `task.result` is present. They select finished tasks unless the query specifies another status.

Completion methods and `cancel_tasks` also accept AQL. Event and joined queries snapshot matching task IDs when the operation starts. Task-only cancellation also applies to later matching tasks.

Cancellation affects only the current execution, not persisted task status. Starting a new run with `start()` clears cancellation. Inspect it with `task.is_cancelled()`, or query execution state with `task.cancelled = true` and `task.pending = true`.

</details>

### Collaboration

Agents can pass work and results in these ways:

1. **Follow-up routing**: [hooks](#hooks) or a condition creates follow-up tasks.
2. **[Task templates](#templates)**: interpolate shared values, results, tasks, and events.
3. **[Knowledge](#knowledge)**: shares durable pages between agents.
4. **[TaskTool](#tools)**: reads any finished task's result by ID.
5. **[ReadFileTool](#tools)**: opens a task's `result.json` in the session directory.

#### Result hook

Use a hook to create a new task when a matching result arrives:

```rust
werk.on_result(|werk, done, result| {
    if done.get_label() == Some("research") {
        werk.add_task(Task::labeled("report", result.clone()));
    }
});
```

#### Conditions

Use a condition to create follow-up tasks or add agents when an AQL query matches. It activates once per run by default. Call `.times(3)` to set a finite count, or `.times(None)` or `.times(0)` to activate for every match. Counts reset at the start of each run.

```rust
use agentwerk::Condition;

werk.add_condition(
    Condition::new("task.label = research AND task.status = finished")
        .agent(Agent::from_env().label("report"))
        .task(Task::labeled(
            "report",
            "Write {{ result: task.label = research AND task.status = finished }}",
        )),
);
```

#### Task templates

Wait for the research task, then insert its result into the report task:

```rust
werk.add_task(Task::labeled("research", "Rank all products by value."));
werk.finish_task("research").await;

werk.add_task(Task::labeled(
    "report",
    "Write the board report from:\n\n{{ result: research }}",
));
```

#### Knowledge

Give agents the same knowledge store so either can write pages that the other reads:

```rust
use agentwerk::Knowledge;

let store = Knowledge::load("./notes")?;

let researcher = Agent::from_env()
    .label("research")
    .knowledge(&store);

let writer = Agent::from_env()
    .label("report")
    .knowledge(&store);
```

#### TaskTool

Give an agent `TaskTool` to read a finished task's result from the same Werk. Here, `t-1` is the completed research task:

```rust
use agentwerk::tools::TaskTool;

let writer = Agent::from_env()
    .label("report")
    .tool(TaskTool);

werk.add_agent(writer);
werk.add_task(Task::labeled(
    "report",
    "Read the result of t-1 with the task tool, then write the board report.",
));
```

#### ReadFileTool

An agent with `ReadFileTool` can instead open the persisted result in the session directory:

```rust
use agentwerk::tools::ReadFileTool;

let writer = Agent::from_env()
    .label("report")
    .tool(ReadFileTool);

werk.add_agent(writer);
werk.add_task(Task::labeled(
    "report",
    "Read .agentwerk/tasks/t-1/result.json, then write the board report.",
));
```

### Configuration

Use a `Policy` to set turn, token, and time limits, retry behavior, and compaction.

```rust
werk.set_policy(Policy {
    max_turns: Some(40),
    max_time: Some(std::time::Duration::from_secs(300)),
    ..Default::default()
});
```

<details>
<summary>Configuration reference</summary>

| Field | Description |
|-------|-------------|
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

</details>

### Compaction

Compaction replaces older messages with a summary as a task approaches the model's context limit or after the provider reports an overflow.

```rust
werk.set_policy(Policy {
    compaction_threshold: Some(0.7),
    ..Default::default()
});
```

<details>
<summary>Compaction reference</summary>

`compaction_threshold` is a fraction of the model's context window, `0.85` by default. Reaching it summarizes the older messages and the agent continues its task.

Compaction also runs after the LLM provider reports that the window was exceeded. `compaction_started`, `compaction_progress`, `compaction_finished`, and `compaction_failed` report each step, see [Events](#events).

```rust
werk.on_event(|_, event| {
    if event.get_name() == Event::COMPACTION_FINISHED {
        eprintln!("[{}] compacted {}", event.get_task_id(), event.get_data()["trigger"]);
    }
});
```

Each compaction event carries the trigger: `proactive` before a context-window error or `reactive` after one. A failure also carries a stable `kind` and human-readable `message`.

</details>

### Sessions

Use a session directory to save tasks, replies, and recorded events, then resume them in a later run.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/sessions.gif" width="600" alt="A persisted agentwerk session" />

The session directory is `./.agentwerk` by default.

```rust
let werk = Werk::load(".agentwerk")?;
werk.add_agent(my_agent);
werk.start();
```

<details>
<summary>Session file reference</summary>

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

</details>

## Events

Use events to inspect what happened during a run.

Publish custom events through the Werk. Add agent or task context when relevant:

```rust
use agentwerk::Event;
use serde_json::json;

werk.emit_event(
    Event::new("document_indexed")
        .data(json!({ "documents": 42 }))
        .task_id("t-1")
        .agent_id("indexer-1"),
);

werk.emit_event(Event::new("index_refreshed"));
```

`Werk::emit_event` does not change task status. Use [EventTool](#eventtool) for model-driven completion through `task_finished`.

Events are saved to `.agentwerk/events.jsonl`, except `text_chunk_received`.

Query recorded events with [AQL](#aql).

<details>
<summary>Event reference</summary>

Event names:

| Area | Name | Description |
|------|------|-------------|
| **Run** | `run_started` | Execution began. |
| | `run_finished` | Execution ended, carrying its outcome. |
| | `policy_violated` | A limit was breached and execution stopped. |
| **Task** | `task_started` | An agent claimed a task. |
| | `task_created` | A task was added to the Werk. |
| | `task_finished` | A task finished successfully, carrying its result when it has one. |
| | `task_failed` | A task failed. |
| | `turn_started` | The agent began another turn on its task. |
| | `schema_retried` | A tool call or result the model created was invalid. |
| **LLM provider** | `request_started` | A request went out to the model. |
| | `request_finished` | A request finished and reported its token usage. |
| | `request_failed` | A request failed and was not retried. |
| | `prompt_render_failed` | A role or task expression could not render before its request. |
| | `request_retried` | A temporary LLM provider error triggered a retry. |
| | `text_chunk_received` | Part of the reply arrived. |
| **Tool** | `tool_call_declined` | A tool call proposed by the model was declined. |
| | `tool_call_repaired` | A tool call or value the model created was invalid and was corrected. |
| | `tool_call_started` | A tool invocation began, carrying its registered name, call ID, and raw input. |
| | `tool_call_finished` | A tool invocation finished. |
| | `tool_call_failed` | A tool invocation failed but the task continues. |
| **Knowledge** | `knowledge_written` | A page was written. |
| | `knowledge_read` | A page was read. |
| | `knowledge_removed` | A page was removed. |
| | `knowledge_listed` | The pages were listed. |
| | `knowledge_failed` | An action against the store did not go through. |
| **Compaction** | `compaction_started` | Compaction is about to rewrite the older messages. |
| | `compaction_progress` | Compaction finished part of the work. |
| | `compaction_finished` | Compaction replaced the older messages. |
| | `compaction_failed` | Compaction could not finish. |
| **Custom** | name chosen by your application | An event published with `emit_event`. |

Event methods:

| Event method | Description |
|--------------|-------------|
| `get_name()` | Read the event name. |
| `get_data()` | Read the event payload. |
| `get_task_id()` | Read the associated task ID. |
| `get_agent_id()` | Read the associated agent ID. |
| `get_label()` | Read the associated task's label. |
| `get_created_at()` | Read the timestamp in epoch milliseconds. |
| `directive(value)` | Set directive metadata. This does not send an instruction to the model. |
| `get_directive()` | Read the directive metadata. |

</details>

### Hooks

Use hooks to run code when an event arrives, a task finishes, or a task changes.

```rust
werk.on_event(|_, event| eprintln!("event: {}", event.get_name()));
werk.on_result(|_, task, result| println!("{}: {result}", task.get_id()));
werk.on_task(|_, event, task| eprintln!("{}: {}", task.get_id(), event.get_name()));
```

`on_result` runs synchronously on the agent. Keep it brief. Use `on_result_async` for work that needs to await.

Async hooks run while a completion method is waiting, and finish before it returns. `start()` alone does not run them. Do not call `finish`, `finish_task`, or `finish_tasks` inside an async hook: it can deadlock.

When no event hook is installed, `event::default_logger()` logs events.

## Knowledge

Use `Knowledge` to store pages on disk and share them between agents and tasks.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/knowledge.gif" width="600" alt="Agents sharing knowledge" />

```rust
use agentwerk::Knowledge;

let store = Knowledge::load("./notes")?;
let alice = Agent::new().knowledge(&store);
let bob = Agent::new().knowledge(&store);
```

Calling `.knowledge(&store)` registers a `KnowledgeTool` bound to that store, so the agent can read and update its shared pages.

Pages use the Open Knowledge Format (OKF) and are stored at `./notes/pages/<slug>.md`. Each has an entry in `./notes/index.md`, which is included in the prompts of agents sharing the store.

By default, prompts include up to 12,000 characters of the index. Agents can read the rest from `index.md`. Pages are always saved in full.

Create entries in code:

```rust
use agentwerk::agents::knowledge::Page;

store.get_pages().save(Page {
    slug: "build-command".into(),
    kind: String::new(),
    description: "How the project is built.".into(),
    content: "Run `make` to compile.".into(),
    tags: vec!["build".into()],
})?;

let page = store.get_pages().get_page("build-command")?;
store.get_pages().remove("build-command")?;
```

<details>
<summary>Knowledge reference</summary>

| Method | Description |
|--------|-------------|
| `get_index()` | Get the index injected into the agent prompt. |
| `set_index_char_limit(count)` | Limit how much of the index is injected into the prompt. |
| `get_index_char_limit()` | Get the active index size limit. |
| `get_pages()` | Get the page collection. |
| `get_pages().get_all()` | Get every page in the store. |
| `clear()` | Remove every page from the store. |

</details>
