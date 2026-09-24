# agentwerk API

Use this guide to configure a Rust agent, add tools and tasks, run several agents together, and share knowledge.

## Contents

Use these sections to move from configuring an agent to coordinating several agents.

| Section | Covers |
| --- | --- |
| [Agents](#agents) | Configure model providers and agent behavior. |
| [Tools](#tools) | Give agents access to files, commands, web pages, events, tasks, and knowledge. |
| [Tasks](#tasks) | Define work, result schemas, and templates. |
| [Werk](#werk) | Coordinate execution, policies, compaction, and sessions. |
| [AQL](#aql) | Find and order tasks, results, and events. |
| [Events](#events) | Publish and inspect runtime activity. |
| [Knowledge](#knowledge) | Share durable pages between agents and tasks. |
| [Collaboration](#collaboration) | Pass work and results between agents. |

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

### Construct an agent

```rust
let agent = Agent();
```

### Configure an agent

Use these members to define the agent's role, tools, routing, and shared context.

| Member | Purpose |
| --- | --- |
| `role(role)` | Define who the agent is and how it should work. |
| `tool(tool)` | Register a tool the agent may call. |
| `tools(tools)` | Register several tools together. |
| `label(label)` | Restrict the agent to tasks carrying this label. |
| `dir(dir)` | Set the directory the agent can access. |
| `template(key, value)` | Set a shared template value. |
| `templates(variables)` | Set several shared template values together. |
| `knowledge(store)` | Share a knowledge store and register its `KnowledgeTool`. |
| `interactive()` | Keep a task in progress while waiting for new instructions. |

### Run an agent

Use these members to submit work, start execution, and collect results.

| Member | Purpose |
| --- | --- |
| `add_task(task)` | Submit a task and return its ID. |
| `start()` | Process tasks in the background. |
| `finish_task(query)` | Wait for all matches and return the first result in query order. |
| `finish_tasks(query)` | Wait for matching tasks and return their results. |
| `finish()` | Run tasks and return their results. |
| `get_id()` | Get the agent's unique identifier. |

### Providers

Send an agent's model requests to Anthropic, OpenAI, Mistral, or a LiteLLM proxy.

```rust
use agentwerk::providers::Anthropic;

let provider = Anthropic(key);
let agent = Agent()
    .provider(provider)
    .model("claude-sonnet-4-20250514");
```

Load the provider or model separately from environment variables with `.provider(Provider::from_env()?)` or `.model(Model::from_env()?)`. Claude, GPT, Mistral, and Qwen families have built-in context-window and reasoning settings; override them or configure a custom model when needed:

```rust
use agentwerk::providers::{Model, ReasoningEffort};

let model = Model("my-local-model")
    .context_window(128_000)
    .reasoning_effort(ReasoningEffort::High);

let agent = Agent().model(model);
```

#### Provider configuration

Use these members to select, configure, and verify a provider.

| Member | Purpose |
| --- | --- |
| `provider(provider)` | Set the LLM provider. |
| `model(model)` | Set the model. |
| `Provider(provider)` | Wrap a provider for sharing. |
| `Model(name)` | Configure a model by name. |
| `Agent::from_env()` | Read the provider and model from environment variables. |
| `verify(model)` | Verify that the provider can answer with a model. |
| `Anthropic(key)` | Configure Anthropic, with optional base URL and timeout overrides. |
| `OpenAi(key)` | Configure OpenAI, with optional base URL and timeout overrides. |
| `Mistral(key)` | Configure Mistral, with optional base URL and timeout overrides. |
| `LiteLlm(key)` | Configure LiteLLM, with optional base URL and timeout overrides. |

#### Provider environment

Set `LITELLM_PROVIDER` to choose a provider explicitly. Otherwise, API keys are checked in the order shown below.

| Variable | Purpose |
| --- | --- |
| `LITELLM_PROVIDER` | Choose `anthropic`, `mistral`, `openai`, or `litellm` outright, ahead of the keys below. |
| `LITELLM_API_KEY` | Authenticate with LiteLLM. |
| `MISTRAL_API_KEY` | Authenticate with Mistral. |
| `ANTHROPIC_API_KEY` | Authenticate with Anthropic. |
| `OPENAI_API_KEY` | Authenticate with OpenAI. |
| `LITELLM_BASE_URL` | Set a different LiteLLM API address. |
| `MISTRAL_BASE_URL` | Set a different Mistral API address. |
| `ANTHROPIC_BASE_URL` | Set a different Anthropic API address. |
| `OPENAI_BASE_URL` | Set a different OpenAI API address. |
| `SSL_CERT_FILE` | Trust the CA certificates in this file instead of the built-in root store. |
| `SSL_CERT_DIR` | Trust the CA certificates in this directory instead of the built-in root store. |

#### Model configuration

Use these members to configure and inspect model limits.

| Member | Purpose |
| --- | --- |
| `context_window(size)` | Set the context window size for a model. |
| `get_context_window()` | Get the configured window size. |
| `reasoning_effort(effort)` | Set the reasoning level. |
| `get_reasoning_effort()` | Get the configured effort. |

#### Model environment

Set `MODEL` to override provider-specific model variables.

| Variable | Purpose |
| --- | --- |
| `MODEL` | Set the model name. |
| `ANTHROPIC_MODEL` | Set the Anthropic model when `MODEL` is unset. |
| `OPENAI_MODEL` | Set the OpenAI model when `MODEL` is unset. |
| `MISTRAL_MODEL` | Set the Mistral model when `MODEL` is unset. |
| `LITELLM_MODEL` | Set the LiteLLM model when `MODEL` is unset. |
| `MODEL_CONTEXT_WINDOW` | Set the context window size in tokens. |

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

Replies pause the task in `in_progress`, and completion methods return when it pauses. Use `add_reply(id, content)` to resume and `set_task_finished(id, result)` to end the conversation. Intermediate replies arrive as [events](#events). `on_result` receives the final result.

## Tools

Add tools to let an agent read and write files, run commands, fetch URLs, manage tasks, and use shared knowledge.

```rust
use agentwerk::tools::{CommandTool, GrepTool, ReadFileTool};

let git = CommandTool("git").allow("git *");

let agent = Agent()
    .tool(ReadFileTool)
    .tool(GrepTool)
    .tool(git);
```

### Built-in tools

The built-in tools cover files, search, commands, web access, events, tasks, and shared knowledge.

#### File tools

These tools read, create, and edit files.

| Tool | Purpose |
| --- | --- |
| `ReadFileTool` | Read a file with line numbers, offset, and limit. |
| `WriteFileTool` | Create or overwrite a file. |
| `EditFileTool` | Replace text in a file. |

#### Search tools

These tools find files and search their contents.

| Tool | Purpose |
| --- | --- |
| `GlobTool` | Find files by pattern. |
| `GrepTool` | Search file contents by regular expression or code shape. |
| `ListDirectoryTool` | List files and directories. |

#### Other built-in tools

These tools run commands, fetch URLs, publish events, manage tasks, and use shared knowledge.

| Tool | Purpose |
| --- | --- |
| `CommandTool` | Grant access to specific commands. |
| `FetchTool` | Fetch a URL and read its body. |
| `EventTool` | Publish an event and optionally finish the current task. |
| `FinishTool` | Write the current task's result and mark it finished. |
| `TaskTool` | Read the Werk and create or edit tasks. |
| `KnowledgeTool` | Write, read, remove, or list pages in a knowledge store. |

### Timeouts

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

### FinishTool

An agent calls `FinishTool` to finish its task and return a result:

```json
{
  "answer": "The configuration is loaded in src/config.rs.",
  "confidence": 0.9
}
```

An agent returns a result through `FinishTool`, which validates any task schema. A non-interactive task without a schema may instead return plain text; [interactive agents](#interactive-agents) have no `FinishTool` unless you explicitly add `.tool(FinishTool)`.

### CommandTool

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

### FetchTool

Use `FetchTool` to fetch a URL as text. It sends the user agent `agentwerk/<version>`. `impersonate()` uses a browser's headers and HTTP/2 settings.

```rust
let web = FetchTool.impersonate();
```

### EventTool

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

Events carry the current task and agent context; see [Events](#events) for hooks and queries. Names are unrestricted, with lowercase snake case conventional. Only `task_finished` completes the current task, using its `data` as the result object:

```json
{
  "name": "task_finished",
  "data": { "answer": "..." }
}
```

Set a [corrective template](#corrective-templates) under the event name to customize the text returned to the model.

### Custom tools

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

## Tasks

Put each piece of work in a task. The task records its status and result.

```rust
use agentwerk::Task;

let task = Task("Review the release notes.").label("review");
werk.add_task(task);
```

### Construct a task

```rust
let task = Task("Review the release notes.");
```

### Identify a task

Use these members to inspect a task's work, label, reporter, and assignee.

| Member | Purpose |
| --- | --- |
| `get_id()` | Get the task ID in the form `t-N`. |
| `get_task()` | Get the work assigned to the task. |
| `get_label()` | Get the task's label. |
| `get_reporter()` | Get the ID of the agent that created the task. |
| `get_assignee()` | Get the ID of the agent that claimed the task. |

### Inspect task outcomes

Use these members to inspect execution state, results, errors, replies, and schemas.

| Member | Purpose |
| --- | --- |
| `get_status()` | Get the task's current status. |
| `is_todo()` | Check whether the task is waiting to be claimed. |
| `is_in_progress()` | Check whether an agent is working on the task. |
| `is_finished()` | Check whether the task finished. |
| `is_failed()` | Check whether the task failed. |
| `is_pending()` | Check whether the task has work in this run. |
| `is_cancelled()` | Check whether this run excluded the task from scheduling. |
| `get_result()` | Get the task's result. |
| `get_errors()` | Get failures recorded against the task as events. |
| `get_replies()` | Get messages exchanged with the model. |
| `get_schema()` | Get the task's optional result schema. |

### Read task timestamps

Use these members to inspect when a task was created, started, finished, or failed.

| Member | Purpose |
| --- | --- |
| `get_created_at()` | Get the creation time in milliseconds. |
| `get_started_at()` | Get the claim time in milliseconds. |
| `get_finished_at()` | Get the finish time in milliseconds. |
| `get_failed_at()` | Get the failure time in milliseconds. |

### Schemas

Attach a `Schema` when a task must return a specific JSON object structure.

```rust
use agentwerk::schemas::Schema;

let schema_document = json!({
    "type": "object",
    "properties": { "title": { "type": "string" } },
    "required": ["title"]
});

let schema = Schema(schema_document)?;
let report = Task("Write a report.").schema(schema);

werk.add_task(report);
```

Schema-bound results must be objects. agentwerk repairs quoted numbers and objects encoded as JSON text, then retries remaining violations up to `max_schema_retries`. Without a schema, any JSON value is valid. Prefer focused schemas for small models and separate tasks for complex work. These members create, validate, and inspect a schema.

| Member | Purpose |
| --- | --- |
| `Schema(document)` | Create a schema. |
| `validate(value)` | Return the validated value and JSON pointers to repaired values, or report violations. |
| `get_raw_schema()` | Read the JSON Schema document the schema was built from. |

### Templates

Templates insert shared values, task results, and event data into roles and tasks. The [prompt skill](skills/prompt/SKILL.md) provides a compact role template. Set template values before adding the task:

```rust
use agentwerk::{Agent, Task};

let writer = Agent::from_env()
    .label("report")
    .role("Write for {{ company }} using:\n{{ find_results(research) }}");

let report = Task("Write the board report.").label("report");

werk.add_agent(writer);
werk.set_template("company", "Canvas Computing");
werk.finish_tasks("research").await;
werk.add_task(report);
```

<a id="template-reference"></a>

#### Named template values

agentwerk renders templates in the role and task just before each task's first model request. Newly added tasks use the latest template values and results.

| Template | Output |
| --- | --- |
| `{{ name }}` | The value assigned to `name`. |

#### Runtime context

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

#### Selecting results

Use these templates to select one result, several results, or fields from matching results.

| Template | Output |
| --- | --- |
| `{{ find_result(AQL) }}` | The first matching result. Strings appear as text and other values as compact JSON. |
| `{{ find_results(AQL) }}` | Matching results as a compact JSON array. |
| `{{ find_result(AQL).field }}` | A field selected from the first result. |
| `{{ find_results(AQL)[*].field }}` | Fields selected from the array of matching results. |

#### Selecting tasks

Use these templates to select one task, several tasks, or fields from matching tasks.

| Template | Output |
| --- | --- |
| `{{ find_task(AQL) }}` | The first matching task as compact JSON. |
| `{{ find_tasks(AQL) }}` | Matching tasks as a compact JSON array. |
| `{{ find_task(AQL).field }}` | A field selected from the first matching task. |
| `{{ find_tasks(AQL)[*].field }}` | Fields selected from the array of matching tasks. |

#### Selecting events

Use these templates to select one event, several events, or fields from matching events.

| Template | Output |
| --- | --- |
| `{{ find_event(AQL) }}` | The first matching event as compact JSON. |
| `{{ find_events(AQL) }}` | Matching events as a compact JSON array. |
| `{{ find_event(AQL).field }}` | A field selected from the first matching event. |
| `{{ find_events(AQL)[*].field }}` | Fields selected from the array of matching events. |

`null`, empty arrays, and unmatched selectors render to an empty string.

#### Available record fields

Task and event templates can select these fields.

| Record | Available fields |
| --- | --- |
| Task | `task`, `label`, `schema`, `id`, `status`, `reporter`, `assignee`, `created_at`, `started_at`, `finished_at`, `failed_at` |
| Event | `name`, `data`, `task_id`, `agent_id`, `label`, `created_at` |

#### Selecting nested values

For example, given this `research` result:

```json
{
  "company": {"name": "Canvas Computing"},
  "findings": [{"summary": "one"}, {"summary": "two"}]
}
```

This template:

```text
{{ find_result(research).company.name }}
```

Renders as:

```text
Canvas Computing
```

Select each task input or event's data:

```text
{{ find_tasks(research)[*].task }}
{{ find_events(event.name = tool_call_failed)[*].data }}
```

`research` is shorthand for `task.label = research`.

| Path | Selects |
| --- | --- |
| `company.name` | A nested field. |
| `metadata."build-id"` | A field that requires JSON quoting. |
| `findings[0]`, `findings[-1]` | An array element. |
| `findings[1:4]`, `findings[::-1]` | An array slice. |
| `findings[*].summary` | The `summary` field from each array element. |
| `authors.*.name` | The `name` field from each object value, in unspecified order. |
| `groups[].members` | The `members` field after flattening one array level. |

Missing fields, incompatible types, and out-of-range indexes produce `null`. Wildcards, slices, and flattening omit null values when more path steps follow. Filters, comparisons, logical expressions, literals, multi-selects, functions, and additional pipes are not supported.

#### Escaping templates

`{ name }` stays unchanged. To output the literal text `{{ name }}`, write `{{{{ name }}}}`.

### Corrective templates

Corrective templates tell agents how to recover from failed tool calls or invalid output. Agentwerk provides [built-in templates](https://github.com/canvascomputing/agentwerk/tree/main/crates/agentwerk/src/prompts/templates) for these failures that you can override.

```rust
let corrective_templates = [
    ("tool_timed_out", "Reduce the command scope."),
    ("cache_miss", "No cache entry exists for {{ path }}."),
];

let agent = Agent::from_env()
    .template("grep_failed", "The search did not run. Narrow `path`.")
    .templates(corrective_templates);
```

## Werk

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

### Construct a Werk

```rust
let werk = Werk::new();
let persisted_werk = Werk(".agentwerk")?;
```

### Configure a Werk

Use these members to set policy, inspect the session directory, and register agents or conditions.

| Member | Purpose |
| --- | --- |
| `set_policy(policy)` | Set execution limits and retry settings. |
| `get_policy()` | Get the active policy. |
| `get_dir()` | Get the session directory. |
| `add_agent(agent)` | Add an agent to the Werk. |
| `add_condition(condition)` | Add a runtime AQL condition and return its ID. |

### Submit and interact

Use these members to create tasks, continue conversations, and set task outcomes.

| Member | Purpose |
| --- | --- |
| `add_task(task)` | Submit a task and return its ID. |
| `add_reply(id, content)` | Add a reply to a task. |
| `edit_replies(id, editor)` | Rewrite a task's replies. |
| `set_task_finished(id, result)` | Finish a task with a result. |
| `set_task_failed(id)` | Mark a task as failed. |

### Observe changes

Use these members to handle events, finished results, and task state changes.

| Member | Purpose |
| --- | --- |
| `on_event(handler)` | Read every event as it is emitted. |
| `on_event_async(handler)` | Read every event in an asynchronous hook. |
| `on_result(handler)` | Read every finished task and its result. |
| `on_result_async(handler)` | Read every finished task and result in an asynchronous hook. |
| `on_task(handler)` | Read task state changes. |
| `on_task_async(handler)` | Read task state changes in an asynchronous hook. |

### Run work

Use these members to start execution and wait for matching results.

| Member | Purpose |
| --- | --- |
| `start()` | Process tasks in the background. |
| `finish_task(query)` | Wait for all matches and return the first result in query order. |
| `finish_tasks(query)` | Wait for matching tasks and return their results. |
| `finish()` | Run tasks and return their results. |

### Cancel work

- `cancel_tasks(query)`: Stop work on matching tasks.
- `cancel()`: Stop work on every task.

### Inspect tasks

Use these members to get tasks directly or find them with AQL.

| Member | Purpose |
| --- | --- |
| `get_task(id)` | Get one task by ID. |
| `get_tasks()` | Get every task in creation order. |
| `find_task(query)` | Get the first matching task. |
| `find_tasks(query)` | Get every matching task. |

### Inspect results and events

Use these members to read completed results and recorded events.

| Member | Purpose |
| --- | --- |
| `get_results()` | Get every finished task result in creation order. |
| `find_result(query)` | Get the first result in query order. |
| `find_results(query)` | Get every result in query order. |
| `find_event(query)` | Get the first recorded event in query order. |
| `find_events(query)` | Get every recorded event in query order. |

### Inspect execution

Use these members to read the run outcome, model choices, token usage, and duration.

| Member | Purpose |
| --- | --- |
| `get_finish_reason()` | Get the reason the last execution ended. |
| `get_model_for_agent(agent_id)` | Get the model used by an agent. |
| `get_input_tokens()` | Get input tokens across finished requests. |
| `get_output_tokens()` | Get output tokens across finished requests. |
| `get_duration()` | Get the elapsed execution duration. |

### Configuration

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

### Compaction

Compaction replaces older messages with a summary as a task approaches the model's context limit or after the provider reports an overflow.

```rust
let policy = Policy {
    compaction_threshold: Some(0.7),
    ..Default::default()
};

werk.set_policy(policy);
```

`compaction_threshold` is a fraction of the model's context window, `0.85` by default. Reaching it summarizes older messages so the agent can continue. Compaction also runs after the LLM provider reports an overflow; `compaction_started`, `compaction_progress`, `compaction_finished`, and `compaction_failed` report each step, see [Events](#events).

```rust
werk.on_event(|_, event| {
    if event.get_name() == Event::COMPACTION_FINISHED {
        eprintln!("[{}] compacted {}", event.get_task_id(), event.get_data()["trigger"]);
    }
});
```

Each compaction event carries the trigger: `proactive` before a context-window error or `reactive` after one. A failure also carries a stable `kind` and human-readable `message`.

### Sessions

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

## AQL

Use Agent Query Language (AQL) to find tasks and events. Pass an AQL string directly, or compile it with `Query(text)` to reuse it.

```rust
// Find tasks labeled `scan`.
werk.find_tasks("scan");

// Find failed tool calls from scan tasks.
werk.find_events("scan AND event.name = tool_call_failed");

// Find the result produced by task `t-3`.
werk.find_results("t-3");
```

### AQL operators

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

### Queryable fields

Filter tasks and events through these fields.

| Origin | Fields |
| --- | --- |
| Task | `task.id`, `task.label`, `task.status`, `task.pending`, `task.cancelled`, `task.assignee`, `task.input`, `task.result`, `task.errors`, `task.created`, `task.started`, `task.finished`, `task.failed` |
| Event | `event.name`, `event.agent_id`, `event.task_id`, `event.label`, `event.created`, `event.data` |

Mixed task and event queries join each event to its referenced task; events without a task do not match. Results retain event-log order unless `ORDER BY` names a task or event field. Result finders default to finished tasks with results; an explicit status overrides this. `finish_task`, `finish_tasks`, and `cancel_tasks` accept AQL. Cancellation lasts only for the current run; `start()` clears it.

## Events

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

`Werk::emit_event` does not change task status; use [EventTool](#eventtool) for model-driven completion through `task_finished`. Events are saved to `.agentwerk/events.jsonl`, except `text_chunk_received`, and can be queried with [AQL](#aql).

### Event catalog

Built-in events are grouped by the part of execution that emits them.

#### Run events

These events describe the start, end, or forced stop of a run.

| Event | Emitted when |
| --- | --- |
| `run_started` | Execution began. |
| `run_finished` | Execution ended, carrying its outcome. |
| `policy_violated` | A limit was breached and execution stopped. |

#### Task events

These events describe task creation, execution, completion, and validation.

| Event | Emitted when |
| --- | --- |
| `task_started` | An agent claimed a task. |
| `task_created` | A task was added to the Werk. |
| `task_finished` | A task finished successfully, carrying its result when it has one. |
| `task_failed` | A task failed. |
| `turn_started` | The agent began another turn on its task. |
| `schema_retried` | A tool call or result the model created was invalid. |

#### Provider events

These events describe model requests and streamed text.

| Event | Emitted when |
| --- | --- |
| `request_started` | A request went out to the model. |
| `request_finished` | A request finished and reported its token usage. |
| `request_failed` | A request failed and was not retried. |
| `prompt_render_failed` | A prompt or template could not render. |
| `request_retried` | A temporary LLM provider error triggered a retry. |
| `text_chunk_received` | Part of the reply arrived. |

#### Tool events

These events describe tool approval, repair, execution, and failure.

| Event | Emitted when |
| --- | --- |
| `tool_call_declined` | A tool call proposed by the model was declined. |
| `tool_call_repaired` | A tool call or value the model created was invalid and was corrected. |
| `tool_call_started` | A tool invocation began, carrying its registered name, call ID, and raw input. |
| `tool_call_finished` | A tool invocation finished. |
| `tool_call_failed` | A tool invocation failed but the task continues. |

#### Knowledge events

These events describe operations against the shared knowledge store.

| Event | Emitted when |
| --- | --- |
| `knowledge_written` | A page was written. |
| `knowledge_read` | A page was read. |
| `knowledge_removed` | A page was removed. |
| `knowledge_listed` | The pages were listed. |
| `knowledge_failed` | An action against the store did not go through. |

#### Compaction events

These events describe each stage of context compaction.

| Event | Emitted when |
| --- | --- |
| `compaction_started` | Compaction is about to rewrite the older messages. |
| `compaction_progress` | Compaction finished part of the work. |
| `compaction_finished` | Compaction replaced the older messages. |
| `compaction_failed` | Compaction could not finish. |

#### Custom events

- Application-defined event: Published with `emit_event` under the name chosen by the application.

### Event objects

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

### Hooks

Use hooks to run code when an event arrives, a task finishes, or a task changes.

```rust
werk.on_event(|_, event| eprintln!("event: {}", event.get_name()));
werk.on_result(|_, task, result| println!("{}: {result}", task.get_id()));
werk.on_task(|_, event, task| eprintln!("{}: {}", task.get_id(), event.get_name()));
```

`on_result` is synchronous; keep it brief or use `on_result_async` for awaited work. Async hooks run only while a completion method waits and finish before it returns; `start()` alone does not run them. Calling `finish`, `finish_task`, or `finish_tasks` inside one can deadlock. Without an event hook, `event::default_logger()` logs events.

## Knowledge

Use `Knowledge` to store pages on disk and share them between agents and tasks.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/knowledge.gif" width="600" alt="Agents sharing knowledge" />

```rust
use agentwerk::Knowledge;

let store = Knowledge("./notes")?;
let alice = Agent().knowledge(&store);
let bob = Agent().knowledge(&store);
```

Calling `.knowledge(&store)` registers a `KnowledgeTool` for shared pages. OKF pages live at `./notes/pages/<slug>.md` and appear in `./notes/index.md`. Shared prompts include the first 12,000 index characters by default; agents can read the rest from `index.md`. Pages are always saved in full. Create entries in code:

```rust
use agentwerk::agents::knowledge::Page;

let build_page = Page {
    slug: "build-command".into(),
    kind: String::new(),
    description: "How the project is built.".into(),
    content: "Run `make` to compile.".into(),
    tags: vec!["build".into()],
};

store.get_pages().save(build_page)?;

let page = store.get_pages().get_page("build-command")?;
store.get_pages().remove("build-command")?;
```

Use these members to inspect and manage the knowledge store.

| Member | Purpose |
| --- | --- |
| `get_index()` | Get the index injected into the agent prompt. |
| `set_index_char_limit(count)` | Limit how much of the index is injected into the prompt. |
| `get_index_char_limit()` | Get the active index size limit. |
| `get_pages()` | Get the page collection. |
| `get_pages().get_all()` | Get every page in the store. |
| `clear()` | Remove every page from the store. |

## Collaboration

Agents can pass work and results in these ways:

1. **[Conditions](#conditions)**: create agents or tasks when an AQL query matches.
2. **[Result hooks](#result-hook)**: create follow-up tasks in application code.
3. **[Task templates](#task-templates)**: interpolate shared values, results, tasks, and events.
4. **[Shared knowledge](#shared-knowledge)**: shares durable pages between agents.
5. **[TaskTool](#tasktool)**: reads any finished task's result by ID.
6. **[ReadFileTool](#readfiletool)**: opens a task's `result.json` in the session directory.

### Conditions

Use a condition to create tasks or add agents when AQL matches. It activates once per run by default; `.times(3)` sets a finite count, while `.times(None)` or `.times(0)` allows every match. Counts reset each run.

```rust
use agentwerk::Condition;

let report_agent = Agent::from_env().label("report");
let report_task = Task(
    "Write {{ find_result(task.label = research AND task.status = finished) }}",
).label("report");

let report_condition = Condition("task.label = research AND task.status = finished")
    .agent(report_agent)
    .task(report_task);

werk.add_condition(report_condition);
```

### Result hook

Use a hook to create a new task when a matching result arrives:

```rust
werk.on_result(|werk, done, result| {
    if done.get_label() != Some("research") {
        return;
    }

    let report = Task(result.clone()).label("report");
    werk.add_task(report);
});
```

### Task templates

Wait for the research task, then insert its result into the report task:

```rust
let research = Task("Rank all products by value.").label("research");
werk.add_task(research);
werk.finish_task("research").await;

let report = Task(
    "Write the board report from:\n\n{{ find_result(research) }}",
).label("report");
werk.add_task(report);
```

### Shared knowledge

Give agents the same knowledge store so either can write pages that the other reads:

```rust
use agentwerk::Knowledge;

let store = Knowledge("./notes")?;

let researcher = Agent::from_env()
    .label("research")
    .knowledge(&store);

let writer = Agent::from_env()
    .label("report")
    .knowledge(&store);
```

### TaskTool

Give an agent `TaskTool` to read a finished task's result from the same Werk. Here, `t-1` is the completed research task:

```rust
use agentwerk::tools::TaskTool;

let writer = Agent::from_env()
    .label("report")
    .tool(TaskTool);

let report = Task(
    "Read the result of t-1 with the task tool, then write the board report.",
).label("report");

werk.add_agent(writer);
werk.add_task(report);
```

### ReadFileTool

An agent with `ReadFileTool` can instead open the persisted result in the session directory:

```rust
use agentwerk::tools::ReadFileTool;

let writer = Agent::from_env()
    .label("report")
    .tool(ReadFileTool);

let report = Task(
    "Read .agentwerk/tasks/t-1/result.json, then write the board report.",
).label("report");

werk.add_agent(writer);
werk.add_task(report);
```
