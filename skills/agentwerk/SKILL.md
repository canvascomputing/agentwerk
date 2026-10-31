---
name: agentwerk
description: Design or review applications built with agentwerk, including agent responsibilities, task contracts, tool access, orchestration, shared state, and completion behavior. Use for application architecture rather than prompt wording alone.
---

# Agentwerk

Design applications from the user's goal and the agentwerk capabilities available to the application.

- For a review, report findings without editing files.
- For a design, edit the named destination or write the design in chat when none is named.
- Implement the application only when explicitly requested.
- Preserve the user's language, provider, deployment, and integration choices.

## 1. Ground the Design

- Read the application's entrypoint, dependency version, existing agents, and code that submits tasks and consumes results.
- Read the [README](../../README.md) and the relevant API documentation before recommending an agentwerk feature.
- Compare the closest coding, research, or event-driven example with the application's actual needs.
- Check the installed API or source when documentation and the application's version differ.
- MUST verify available tools, files, commands, and network access before designing work that requires them.

The links in this skill refer to its source checkout, including when installed through a repository symlink.
Use the target application's version as the authority for compatibility.

## 2. Define the Application's Job

- Identify the user input, its source, and the artifact or action the application must produce.
- Name the output consumer and how it decides whether the work succeeded.
- Establish the workplace, available data, permitted effects, and relevant time or token constraints.
- Choose batch execution or an interactive conversation based on whether follow-up replies must continue the same task.
- Ask about missing requirements only when the answer changes the design.

Keep application startup, UI, logging, and domain state in the consuming application.
Use agentwerk to compose agents, tools, tasks, and execution state.

## 3. Assign Responsibilities and Work

### Agent Boundaries

- Start with one agent unless separate tools, expertise, concurrency, or independent verification justify more.
- Give each agent a concrete responsibility and the tools needed to perform it.
- Keep stable identity and operating rules in the role, and changing objectives and inputs in tasks.
- Use a unique label to target one agent or a shared label for a worker pool.
- MUST match task and agent labels exactly, including leaving both unlabelled when no label is needed.

Agents sharing a label claim different tasks; a task is claimed once.
Each agent processes one claimed task at a time.
See [Agents](../../docs/api/agents.md) for configuration and interactive behavior.

### Task Contracts

- Define each task's input sources, expected output, consumer, and stopping condition.
- Attach a focused object `Schema` when downstream code requires a machine-checked result shape.
- MUST align schema-bound output with the fields accepted by the task's completion tool.
- Define a blocked outcome when domain work can fail without a framework failure.
- Ensure downstream code checks the outcome before treating a finished task as successful domain work.
- Use the [prompt skill](../prompt/SKILL.md), when available, for detailed role, task, and tool wording.

A batch agent normally receives `FinishTool`.
A schema-less batch task can also finish with plain text.
Interactive agents have no completion tool by default; replies pause the task in `in_progress`.
The host resumes it with `add_reply` and ends it with `set_task_finished`.
See [Tasks](../../docs/api/tasks.md) for schemas and templates.

## 4. Make Dependencies Explicit

### Scheduling and Handoffs

- Use one `Werk` as the orchestration boundary; nested Werks are unsupported.
- Start dependent tasks through a matching `Condition`, a result hook, or an explicit completion wait in application code.
- Use conditions for declarative task or event matches and hooks when application logic must inspect a result or build task inputs.
- Specify the producer, readiness signal, transferred data, and consumer for each handoff.
- Distinguish a per-result follow-up from a fan-in step that must wait for every required producer.
- Register conditions before the activity they must observe.
- Define condition repetition deliberately and prevent follow-up work from recreating its own trigger indefinitely.

A condition activates once per run by default.
`times(n)` allows a finite count; zero or `None` allows every match.
Conditions are not restored from sessions, and earlier events are not replayed.
An unmet condition does not keep `finish()` waiting.
See [Collaboration](../../docs/api/collaboration.md) for composition patterns and [AQL](../../docs/api/aql.md) for matching syntax.

### Templates and Result Selection

- NEVER use a result selector as a substitute for a dependency; templates select available data without waiting.
- Set shared template values before the task's first model request.
- Select the intended task or result explicitly when repeated runs or multiple tasks share a label.
- Handle missing results before launching consumers that require them.
- Supply changing observations through tools, replies, or new tasks rather than expecting an active prompt to refresh.

Initial role and task templates render before the first request, and the system prompt remains frozen across later turns and reloads.
Later messages are literal.
Missing selectors can render as empty text, so a rendered prompt alone does not prove its inputs are ready.

## 5. Choose Tools and State

### Tool Access

- Prefer existing tools when their actual contracts fit the work.
- Give each custom tool an input schema, result contract, timeout where needed, and predictable recovery behavior.
- Keep deterministic domain validation and state transitions in application code or tool handlers.
- Define command permissions in `CommandTool` configuration rather than relying on prompt restrictions alone.
- Bind actor identity in application code when a tool must act for one specific agent.
- Enable `concurrent(true)` only for side-effect-free tools that safely overlap other calls.
- Distinguish tool rejection, recoverable failure, and completed effects in returned data.

See [Tools](../../docs/api/tools.md) for built-ins and custom handlers.
Publishing a custom event through `Werk::emit_event` does not change task status.
`EventTool` completes a task only for `task_finished`, using its data as the result.

### State Ownership

- Keep authoritative domain state in application code and expose deliberate read or mutation tools.
- Transfer bounded outputs through task results and use shared `Knowledge` when durable pages benefit later tasks.
- Share knowledge explicitly through the same store handle; constructing an agent does not grant knowledge access.
- Account for index-only prompt injection and on-demand page reads when designing knowledge-dependent work.
- Choose session storage deliberately and distinguish it from the agent's working directory and the knowledge bundle.
- Rebuild agents, tools, templates, conditions, and hooks when resuming a persisted session.
- Preserve agent registration order within labels when started tasks must resume on the same generated agent IDs.

Knowledge index changes become visible in the next task rather than refreshing an active prompt.
See [Knowledge](../../docs/api/knowledge.md) for page storage and [Werk](../../docs/api/werk.md) for sessions.

## 6. Define Run Ending and Recovery

- Select policy limits and tool timeouts proportionate to the work without inventing fixed budgets the user did not request.
- Distinguish provider request retries, schema or tool correction retries, and application-level recovery.
- Specify how blocked work, task failure, cancellation, and exhausted policy limits affect dependent tasks and the caller's answer.
- Inspect task outcomes and the run's finish reason before reporting application success.
- Keep synchronous hooks brief and use asynchronous hooks for awaited work.
- Drive asynchronous hooks through a completion method; `start()` alone does not execute them.
- NEVER call a completion wait inside an asynchronous hook, because it can deadlock.

Cancellation excludes tasks only for the current run; starting another run clears it.
An interactive pause can return control while its task remains open.
Compaction summarizes context; its threshold is not a terminal execution budget.
Use [Events](../../docs/api/events.md) to observe progress, failures, policy violations, and the terminal run outcome.

## 7. Walk Through and Report

### Design Verification

- Trace one representative input through assignment, tool use, state changes, handoffs, and the final consumed output.
- Check that every task has an eligible agent and every required input has an available source.
- Check parallel branches and fan-in readiness when the design includes concurrent work.
- Walk through missing inputs, rejected actions, invalid results, and the applicable blocked or failed outcome.
- Check pause, resume, repeated triggers, and session restoration only when the application uses them.
- Define acceptance checks for observable behavior rather than prompt wording or internal call order.
- Verify every API shown in code against the application's language and dependency version.

### Deliverable

- For a design, state the application goal and the chosen agent responsibilities, task contracts, and tool access.
- Describe execution order, handoff data, state ownership, completion, and failure behavior.
- Include acceptance checks and unresolved requirements or explicit assumptions.
- Use a small code sketch or flow diagram when it resolves ambiguity in the proposed design.
- For a review, report each finding's location, consequence, and concrete correction.
- State which walkthroughs or checks were performed and which remain unverified.

Keep instructions direct, with one sentence of at most 30 words per instruction bullet.
Keep instruction lists to seven sibling items or fewer, grouping longer material under meaningful headings.
