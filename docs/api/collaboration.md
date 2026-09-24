# Collaboration

Agents can pass work and results in these ways:

1. **[Conditions](#conditions)**: create agents or tasks when an AQL query matches.
2. **[Result hooks](#result-hook)**: create follow-up tasks in application code.
3. **[Task templates](#task-templates)**: interpolate shared values, results, tasks, and events.
4. **[Shared knowledge](#shared-knowledge)**: shares durable pages between agents.
5. **[TaskTool](#tasktool)**: reads any finished task's result by ID.
6. **[ReadFileTool](#readfiletool)**: opens a task's `result.json` in the session directory.

## Conditions

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

## Result hook

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

## Task templates

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

## Shared knowledge

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

## TaskTool

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

## ReadFileTool

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
