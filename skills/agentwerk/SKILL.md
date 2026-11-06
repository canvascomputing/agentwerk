---
name: agentwerk
description: Design, implement, or review agentwerk applications using agents, label-routed tasks, Conditions, AQL, results, hooks, and events for orchestration. Use the prompt skill for detailed role and tool wording.
---

# Agentwerk

A Werk schedules tasks for agents and records their results and events.
Agents claim tasks through matching labels, while Conditions and hooks create dependent work.
Build on these records and scheduling mechanisms rather than maintaining a second orchestration system.

- For a review, report findings without editing files.
- Implement an application only when explicitly requested.
- Preserve the application's language, provider, deployment, and integration choices.
- Use the [prompt skill](../prompt/SKILL.md) for roles, task instructions, tool contracts, and result schemas.

## Start Here: Parallel Summaries and One Report

This example summarizes two supplied notes in parallel, then combines their results for an external audience.
The notes are task inputs, the audience is event data, and the report consumes task results.
Both languages require configured provider credentials and a model.
Use a fresh session directory for each example run.

Rust:

```rust
use agentwerk::{Agent, Condition, Event, Task, Werk};
use serde_json::json;

async fn summarize_notes() -> Result<(), Box<dyn std::error::Error>> {
    let werk = Werk("./summary-session-rust")?;
    let worker = Agent::from_env()
        .label("summarize")
        .role("Summarize the supplied note for the report writer. Return the summary as plain text.");
    for _ in 0..2 {
        werk.add_agent(worker.clone());
    }
    werk.add_agent(Agent::from_env().label("report").role(
        "Combine supplied summaries into one report for the supplied audience. Return the report as plain text.",
    ));

    werk.add_condition(
        Condition("event.name = summaries_ready").task(
            Task(concat!(
                "Write a report for this audience:\n",
                "{{ find_event(event.name = customer_brief).data.audience }}\n",
                "Use these summaries:\n{{ find_results(summarize) }}",
            )).label("report"),
        ),
    );
    werk.on_result(|werk, done, _result| {
        if done.get_label() != Some("summarize") {
            return;
        }
        let tasks = werk.find_tasks("summarize");
        if !tasks.is_empty() && tasks.iter().all(|task| task.is_finished()) {
            werk.emit_event(Event("summaries_ready"));
        }
    });

    werk.emit_event(Event("customer_brief").data(json!({"audience": "engineering"})));
    for note in [
        "The API supports bulk imports. Summarize this capability.",
        "The importer validates each record before writing. Summarize this behavior.",
    ] {
        werk.add_task(Task(note).label("summarize"));
    }
    werk.finish().await;

    let report = werk.find_result("report").ok_or("No report was produced")?;
    println!("{report}");
    Ok(())
}
```

Python:

```python
import asyncio
from agentwerk import Agent, Condition, Event, Task, Werk


async def summarize_notes():
    werk = Werk("./summary-session-python")
    worker = Agent.from_env().label("summarize").role(
        "Summarize the supplied note for the report writer. Return the summary as plain text."
    )
    for _ in range(2):
        werk.add_agent(worker)
    werk.add_agent(Agent.from_env().label("report").role(
        "Combine supplied summaries into one report for the supplied audience. Return the report as plain text."
    ))

    werk.add_condition(
        Condition("event.name = summaries_ready").task(Task(
            "Write a report for this audience:\n"
            "{{ find_event(event.name = customer_brief).data.audience }}\n"
            "Use these summaries:\n{{ find_results(summarize) }}",
            label="report",
        ))
    )

    def launch_report(host, done, result):
        if done.get_label() != "summarize":
            return
        tasks = host.find_tasks("summarize")
        if tasks and all(task.is_finished() for task in tasks):
            host.emit_event(Event("summaries_ready"))

    werk.on_result(launch_report)
    werk.emit_event(Event("customer_brief").data({"audience": "engineering"}))
    for note in [
        "The API supports bulk imports. Summarize this capability.",
        "The importer validates each record before writing. Summarize this behavior.",
    ]:
        werk.add_task(Task(note, label="summarize"))
    await werk.finish()

    report = werk.find_result("report")
    if report is None:
        raise RuntimeError("No report was produced")
    print(report)


if __name__ == "__main__":
    asyncio.run(summarize_notes())
```

All producer tasks are registered before execution starts.
The hook derives readiness from task records without storing results or counting completions separately.
Overlapping hooks can emit readiness more than once, but the Condition creates only one report per run.
A failed worker leaves the report absent, which the final check detects.
Applications with blocked domain outcomes must also validate those outcomes before emitting readiness.

## 1. Ground the Workflow

- Read the application entrypoint, dependency version, agent definitions, task creation, and result consumers.
- Check the [README](../../README.md) and relevant [API documentation](../../docs/api/werk.md) against the application's installed version.
- MUST verify available tools, files, commands, and network access before assigning work that requires them.
- Identify the input, required output, consumer, and observable success criteria.
- Ask about missing requirements only when their answers change the workflow.

Documentation links refer to this skill's source checkout, including installation through a repository symlink.
The target application's installed API is the authority for compatibility.

## 2. Initialize a Werk and Register Agents

- Use one Werk as the orchestration boundary.
- Give each agent a stable responsibility and the tools needed for its tasks.
- Keep changing objectives and inputs in tasks.
- Replicate an agent configuration when independent tasks can run in parallel.
- MUST match task and agent labels exactly, including leaving both unlabelled when no label is needed.
- NEVER implement custom threads, worker queues, polling loops, or executors to schedule agentwerk tasks.

Agents sharing a label form a worker pool.
Each agent handles one claimed task at a time, and each task is claimed once.
Rust registers cloned configurations; Python's `add_agent` copies the configuration on each registration.
Nested Werks are unsupported.

### Let an Agent Delegate with TaskTool

Use `TaskTool` when choosing follow-up work requires agent judgment.
Conditions and host hooks remain the default for predetermined handoffs.

Rust:

```rust
use agentwerk::tools::TaskTool;

werk.add_agent(Agent::from_env().label("plan").tool(TaskTool));
```

Python:

```python
from agentwerk import TaskTool

werk.add_agent(Agent.from_env().label("plan").tool(TaskTool()))
```

An agent can call the `task` tool to create work for registered agents carrying the `review` label:

```json
{"action": "create", "label": "review", "task": "Review the proposed API for compatibility risks."}
```

It can inspect existing work before deciding what to create:

```json
{"action": "list", "aql": "task.label = review ORDER BY task.created DESC"}
```

- Give the delegating agent explicit task boundaries and eligible worker labels.
- Have the delegating agent finish its own task after submitting its assignments.
- Use `TaskTool` to inspect or change task content, not to transition task status.

`TaskTool` also reads finished results by task ID.
Its task listing is capped at 50 entries, so queries should select the relevant work.

## 3. Pass Task Results Through the Werk

- Store an agent's output as its task result.
- Avoid custom dictionaries, shared structs, files, or counters used solely to pass results or track workflow progress.
- Query previous tasks, results, and events through AQL instead of duplicating their state.
- Attach an object `Schema` when downstream code requires a validated result shape.
- MUST align schema-bound output with the fields accepted by the task's completion tool.
- Define a blocked result when domain work can fail without a framework failure.
- Check domain outcomes before launching work that requires success.

A finished task can report a blocked domain outcome.
Task completion alone does not prove domain success.
See [Tasks](../../docs/api/tasks.md) and [AQL](../../docs/api/aql.md).

### Results in Code and Templates

Rust:

```rust
let summaries = werk.find_results("task.label = summarize AND task.status = finished");
let report = werk.find_result("task.label = report AND task.status = finished");
```

Python:

```python
summaries = werk.find_results("task.label = summarize AND task.status = finished")
report = werk.find_result("task.label = report AND task.status = finished")
```

The same result selectors work inside role and task strings:

```text
Summaries supplied by the workers:
{{ find_results(task.label = summarize AND task.status = finished) }}

The summary field from one result:
{{ find_result(task.label = summarize AND task.status = finished).summary }}

Summary fields from all matching results:
{{ find_results(task.label = summarize AND task.status = finished)[*].summary }}
```

Field selectors assume results contain the named fields.
The singular finder returns the first match in query order.

- NEVER use a result selector as a substitute for a dependency.
- Select the intended task IDs or ordering when labels cover multiple batches or repeated runs.
- Ensure required results exist before launching their consumers.
- Set shared template values before the task's first model request.
- Supply later observations through tools, replies, or new tasks.

Templates render before the first request and do not wait for results.
Missing values can render as empty strings, and the system prompt remains frozen during the task.

## 4. Connect Phases with Conditions and Hooks

- Use Conditions for declarative task or event triggers.
- Use `on_result` when code must inspect a result or compute follow-up tasks.
- Query task records when a consumer must wait for every producer.
- Register Conditions and hooks before the activity they must observe.
- Define trigger repetition deliberately to prevent recursive or duplicate follow-up work.
- Keep synchronous hooks brief and use async hooks for awaited work.
- NEVER call a completion wait inside an asynchronous hook, because it can deadlock.

A Condition activates once per run by default.
`times(n)` sets a finite limit; zero or `None` allows every match.
A task query matches individual tasks, so one finished worker does not establish that every worker finished.
Conditions do not replay earlier events and are not restored from persisted sessions.
Async hooks run while a completion method waits; `start()` alone does not drive them.
See [Collaboration](../../docs/api/collaboration.md) and [Events](../../docs/api/events.md).

### A Single-Producer Handoff

Here, `draft` identifies exactly one producer task.
Register this Condition before submitting that task; the downstream agent carries the label `review`.

Rust:

```rust
werk.add_condition(
    Condition("task.label = draft AND task.status = finished")
        .task(Task("Review this draft:\n{{ find_result(draft) }}").label("review")),
);
```

Python:

```python
werk.add_condition(
    Condition("task.label = draft AND task.status = finished")
    .task(Task("Review this draft:\n{{ find_result(draft) }}", label="review"))
)
```

When readiness depends on structured result fields, inspect them in a result hook rather than inventing unsupported AQL field expressions.

### Create a Task Directly from a Result Hook

For a per-result review, the hook receives the finished task and its result directly.
This example schedules a review only when the producer reports `needs_review: true`.
The `draft` task must have an object schema requiring a boolean `needs_review`, and an agent must carry the `review` label.

Rust:

```rust
werk.on_result(|werk, done, result| {
    if done.get_label() == Some("draft") && result["needs_review"] == true {
        werk.add_task(Task(format!("Review this draft:\n{result}")).label("review"));
    }
});
```

Python:

```python
def review_draft(host, done, result):
    if done.get_label() == "draft" and result["needs_review"]:
        host.add_task(Task(f"Review this draft:\n{result}", label="review"))


werk.on_result(review_draft)
```

Use either this hook or the unconditional draft Condition above, according to the required behavior.

### Handle Failed Tasks with a Lifecycle Hook

`on_result` does not observe task failures.
Use `on_task` for task lifecycle changes or `on_event` for broader events, including policy violations and tool failures.
This hook cancels report work and records which task failed, without introducing a custom failure registry.
Register it before execution starts.

Rust:

```rust
werk.on_task(|werk, event, task| {
    if event.get_name() == Event::TASK_FAILED {
        werk.cancel_tasks("report");
        werk.emit_event(Event("workflow_blocked").data(json!({"task_id": task.get_id()})));
    }
});
```

Python:

```python
def stop_report_on_failure(host, event, task):
    if event.get_name() == Event.TASK_FAILED:
        host.cancel_tasks("report")
        host.emit_event(Event("workflow_blocked").data({"task_id": task.get_id()}))


werk.on_task(stop_report_on_failure)
```

Cancellation applies to the current run and clears when a new run starts.
A failed tool call can be recoverable within a task; it does not automatically mean the task failed.

## 5. Publish External Data as Events

- Emit an event when agents need an observation or input that no task produced.
- Read recorded event payloads through AQL in code or template selectors in prompts.
- Use an event Condition when the observation should launch work.
- Keep authoritative domain state and deterministic validation in the application or tool handlers.

Events expose application data without a shared workflow-state object.
Emitting a custom event does not change task status.
Event-only queries can find events without a task; mixed task/event queries require an associated task.

Rust:

```rust
werk.emit_event(Event("customer_brief").data(json!({"audience": "engineering"})));
let brief = werk.find_event("event.name = customer_brief ORDER BY event.created DESC");
```

Python:

```python
werk.emit_event(Event("customer_brief").data({"audience": "engineering"}))
brief = werk.find_event("event.name = customer_brief ORDER BY event.created DESC")
```

Prompt:

```text
Write for this audience:
{{ find_event(event.name = customer_brief ORDER BY event.created DESC).data.audience }}
```

### Let an Agent Publish with EventTool

Register `EventTool` when an agent must signal an intermediate milestone before finishing its task.

Rust:

```rust
use agentwerk::tools::EventTool;

werk.add_agent(Agent::from_env().label("research").tool(EventTool));
```

Python:

```python
from agentwerk import EventTool

werk.add_agent(Agent.from_env().label("research").tool(EventTool()))
```

The agent calls the `event` tool with a name and payload:

```json
{"name": "source_verified", "data": {"url": "https://example.com/report"}}
```

Conditions can react to `event.name = source_verified`, and templates can read the event's `data.url`.
The event includes the emitting agent and current task context automatically.
Custom milestones leave the task open; `task_finished` is special and completes it using the event payload as its result.

### Share Durable Reference Pages with Knowledge

Use shared [Knowledge](../../docs/api/knowledge.md) for reusable findings or project notes that later tasks should discover and read.
Both agents must receive the same store explicitly.

Rust:

```rust
use agentwerk::Knowledge;

let notes = Knowledge("./shared-notes")?;
werk.add_agent(Agent::from_env().label("research").knowledge(&notes));
werk.add_agent(Agent::from_env().label("report").knowledge(&notes));
```

Python:

```python
from agentwerk import Knowledge

notes = Knowledge("./shared-notes")
werk.add_agent(Agent.from_env().label("research").knowledge(notes))
werk.add_agent(Agent.from_env().label("report").knowledge(notes))
```

The attachment registers `KnowledgeTool` and adds the store's index to the agent prompt.
Full pages are read on demand; index changes appear in subsequent tasks rather than refreshing an active prompt.
Keep bounded handoff outputs in task results and use Conditions or hooks to establish readiness even when agents share knowledge.

## 6. Run the Workflow with One Final Wait

- Prefer one `werk.finish()` call after registering agents, Conditions, hooks, and initial tasks.
- Use Conditions or hooks to advance phases instead of calling completion methods between phases.
- Inspect expected outputs, failed tasks, and the run's finish reason before reporting success.
- Choose policy limits and tool timeouts proportionate to the requested work.
- Rebuild agents, tools, templates, Conditions, and hooks when resuming a persisted session.

An unmet Condition does not keep `finish()` waiting.
Interactive workflows may require repeated waits as the user supplies replies.
An agent's `finish` tool completes its current task; the host's `werk.finish()` drives the overall run.

## 7. Verify and Report

- Trace one input through task assignment, result creation, dependent work, and the final output.
- Check that every task has an eligible agent and every required input has an available source.
- Verify parallel branches wait for all required producers before combining results.
- Check missing results, blocked outcomes, failed tasks, and duplicate triggers.
- Verify every example API against the application's language and dependency version.
- Report changes or findings with the checks performed and any unresolved limitations.
