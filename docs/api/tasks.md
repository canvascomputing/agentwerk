# Tasks

Put each piece of work in a task. The task records its status and result.

```rust
use agentwerk::Task;

let task = Task("Review the release notes.").label("review");
werk.add_task(task);
```

## Construct a task

```rust
let task = Task("Review the release notes.");
```

## Identify a task

Use these members to inspect a task's work, label, reporter, and assignee.

| Member | Purpose |
| --- | --- |
| `get_id()` | Get the task ID in the form `t-N`. |
| `get_task()` | Get the work assigned to the task. |
| `get_label()` | Get the task's label. |
| `get_reporter()` | Get the ID of the agent that created the task. |
| `get_assignee()` | Get the ID of the agent that claimed the task. |

## Inspect task outcomes

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

## Read task timestamps

Use these members to inspect when a task was created, started, finished, or failed.

| Member | Purpose |
| --- | --- |
| `get_created_at()` | Get the creation time in milliseconds. |
| `get_started_at()` | Get the claim time in milliseconds. |
| `get_finished_at()` | Get the finish time in milliseconds. |
| `get_failed_at()` | Get the failure time in milliseconds. |

## Schemas

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

## Templates

Templates insert shared values, task results, and event data into roles and tasks. The [prompt skill](../../skills/prompt/SKILL.md) provides a compact role template. Set template values before adding the task:

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

### Named template values

agentwerk renders templates in the role and task just before each task's first model request. Newly added tasks use the latest template values and results.

| Template | Output |
| --- | --- |
| `{{ name }}` | The value assigned to `name`. |

### Runtime context

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

### Selecting results

Use these templates to select one result, several results, or fields from matching results.

| Template | Output |
| --- | --- |
| `{{ find_result(AQL) }}` | The first matching result. Strings appear as text and other values as compact JSON. |
| `{{ find_results(AQL) }}` | Matching results as a compact JSON array. |
| `{{ find_result(AQL).field }}` | A field selected from the first result. |
| `{{ find_results(AQL)[*].field }}` | Fields selected from the array of matching results. |

### Selecting tasks

Use these templates to select one task, several tasks, or fields from matching tasks.

| Template | Output |
| --- | --- |
| `{{ find_task(AQL) }}` | The first matching task as compact JSON. |
| `{{ find_tasks(AQL) }}` | Matching tasks as a compact JSON array. |
| `{{ find_task(AQL).field }}` | A field selected from the first matching task. |
| `{{ find_tasks(AQL)[*].field }}` | Fields selected from the array of matching tasks. |

### Selecting events

Use these templates to select one event, several events, or fields from matching events.

| Template | Output |
| --- | --- |
| `{{ find_event(AQL) }}` | The first matching event as compact JSON. |
| `{{ find_events(AQL) }}` | Matching events as a compact JSON array. |
| `{{ find_event(AQL).field }}` | A field selected from the first matching event. |
| `{{ find_events(AQL)[*].field }}` | Fields selected from the array of matching events. |

`null`, empty arrays, and unmatched selectors render to an empty string.

### Available record fields

Task and event templates can select these fields.

| Record | Available fields |
| --- | --- |
| Task | `task`, `label`, `schema`, `id`, `status`, `reporter`, `assignee`, `created_at`, `started_at`, `finished_at`, `failed_at` |
| Event | `name`, `data`, `task_id`, `agent_id`, `label`, `created_at` |

### Selecting nested values

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

### Escaping templates

`{ name }` stays unchanged. To output the literal text `{{ name }}`, write `{{{{ name }}}}`.

## Corrective templates

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
