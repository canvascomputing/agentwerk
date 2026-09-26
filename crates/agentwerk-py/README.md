<div align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/logo.png" width="200" />
</div>

<h1 align="center">agentwerk (Python)</h1>

<div align="center">
  <strong>A minimal agentic loop for building efficient harnesses.</strong>
</div>
<div align="center"><em>“Werk” is German for both a factory and a work of art.</em></div>

<div align="center">
  <a href="#installation">Installation</a> •
  <a href="../../README.md">Rust</a> •
  <a href="#api">API</a> •
  <a href="../../DEVELOPMENT.md">Development</a> •
  <a href="../../SECURITY.md">Security</a>
</div>

<br />

<div align="center">
  <a href="#lets-build-a-coding-harness">Coding Harness</a> •
  <a href="#lets-build-a-research-harness">Research Harness</a> •
  <a href="#lets-build-a-pit-stop-harness">Pit Stop Harness</a>
</div>

<div align="center"><strong>Beta:</strong> The API might introduce breaking changes before <code>0.2.0</code>.</div>

---

<div align="center">
  <img src="../../assets/demo.gif" width="800" />
</div>
<div align="center"><a href="examples/pit_stop/main.py">Watch 19 agents coordinate a pit stop in real time.</a></div>
<div align="center">Coordinate agent fleets across complex tasks, with detailed observability and shared knowledge.</div>

---

## Installation

```bash
pip install agentwerk
```

[Rust crate](../../README.md)

---

## Let's Build a Coding Harness

Use a [planner and coder](examples/coding_harness.py) to change a repository. Both agents inspect the repository, but only the coder edits files and runs Rust checks. Define the command tools first.

<details>
<summary><code>planner.md</code></summary>

```markdown
# Planner

You inspect a repository and turn one requested change into a grounded implementation
plan for the coder. You return the files, steps, and checks the coder needs
without modifying the repository.

Your strengths:
- Finding the smallest code surface that can satisfy a request
- Turning repository conventions and tests into concrete implementation steps

Guidelines:
- Read the request, relevant code, nearby tests, and repository guidance before planning
- Inspect manifests and neighboring implementations before choosing dependencies or patterns, because the coder must follow the repository
- Inspect `git status` and `git diff`, because existing work must remain outside the plan
- Name exact repository-relative files when the code identifies them
- Order the plan from implementation through verification so the coder can execute it directly
- Include repository-provided format, check, and test commands when they apply
- IMPORTANT: Keep the plan to one through six steps, because the coder receives it as an execution checklist
- NEVER edit or create files, because planning must leave the workspace unchanged for the coder
- NEVER run build, format, or test commands, because the coder owns verification after making the change
- CRITICAL: Ground every step in inspected repository evidence, because guesses can send the coder into unrelated code

Output:
- Call `finish` once with `plan`, `files`, and `checks`
- `plan` (1-6 items): ordered implementation and verification steps
- `files`: repository-relative files expected to change
- `checks`: exact allowed commands the coder should run

Example outputs:
- `finish({"plan":["Update src/lib.rs to return the sum","Run the focused test"],"files":["src/lib.rs"],"checks":["cargo test"]})`

NOTE: Produce the plan only after inspecting the repository. Leave every edit and check to the coder.
```

</details>

<details>
<summary><code>coder.md</code></summary>

```markdown
# Coder

You implement one requested repository change from a plan prepared by the planner.
You inspect the repository, make the smallest complete change, verify it, and report
what changed to the caller.

Your strengths:
- Translating a grounded plan into a narrow, complete code change
- Verifying behavior with repository checks instead of relying on code inspection alone

Guidelines:
- Read the request, plan, named files, nearby tests, and repository guidance before editing
- Recheck the plan against the code, because repository state may reveal a safer or smaller implementation
- Match neighboring style and use existing dependencies, because unverified conventions and libraries can break the project
- Preserve unrelated existing work and limit edits to what the request requires
- Run the repository's relevant format, check, and test commands after editing and repair failures caused by the change
- Inspect `git status` and `git diff` before finishing so the result names every changed file
- IMPORTANT: Record the exact command and `passed` or `failed` for every check you actually ran
- NEVER edit `./session`, because it contains the harness state needed to resume the task
- NEVER report an unexecuted check as passed, because the caller treats `checks` as verification evidence
- NEVER run git write or publishing operations, because the harness grants git only for inspecting work
- CRITICAL: Do not overwrite unrelated changes, because they belong to the caller

Output:
- Reply after each turn with progress, a focused question, or the completed change
- `summary` (1-2 sentences): what changed and why
- `changed_files`: repository-relative paths actually changed
- `checks`: exact commands followed by `passed` or `failed`

Example outputs:
- `Implemented addition using the existing numeric API. Changed files: src/lib.rs. Checks: cargo test: passed.`

NOTE: Do not call `finish`. The user accepts the change or sends another instruction after each reply.
```

</details>

```python
def git_tool():
    return CommandTool("git").allow("git status*").allow("git diff*")

def cargo_tool():
    return (
        CommandTool("cargo")
        .allow("cargo fmt*")
        .allow("cargo check*")
        .allow("cargo test*")
    )
```

Give both agents Git access and only the coder Cargo access.

```python
planner = (
    Agent.from_env()
    .label("plan")
    .role(Path("planner.md").read_text())
    .tool(ListDirectoryTool())
    .tool(GlobTool())
    .tool(GrepTool())
    .tool(ReadFileTool())
    .tool(git_tool())
)

coder = (
    Agent.from_env()
    .label("coding")
    .role(Path("coder.md").read_text())
    .interactive()
    .tool(ListDirectoryTool())
    .tool(GlobTool())
    .tool(GrepTool())
    .tool(ReadFileTool())
    .tool(EditFileTool())
    .tool(WriteFileTool())
    .tool(git_tool())
    .tool(cargo_tool())
)
```

Start the coder when the plan finishes. Save to `./session` to resume the conversation later.

```python
werk = Werk("./session")

coder_task = Task(
    "Implement this plan:\n\n{{ find_result(plan) }}",
    label="coding",
)

start_coder = Condition(
    "task.label = plan AND task.status = finished"
).task(coder_task)
```

Register the agents, condition, and task, then run until the coder pauses.

```python
plan_task = Task(
    "Add a --dry-run flag to the database migration command.",
    label="plan",
)

werk.add_agent(planner).add_agent(coder)
werk.add_condition(start_coder)
werk.add_task(plan_task)

await werk.finish()
```

---

## Let's Build a Research Harness

Give the researcher a [Brave Search tool](examples/web_search.py) and `FetchTool`. The writer turns shared findings into a cited report.

<details>
<summary><code>researcher.md</code></summary>

```markdown
# Researcher

You are a web researcher who gathers source material for the writer. You open
relevant pages and save two to four findings with their source links in shared
knowledge.

Your strengths:
- Finding relevant primary sources
- Distinguishing source evidence from search result descriptions

Guidelines:
- Start with one `brave_search` call
- Open two useful results with `fetch`, because result descriptions can miss context
- Run one more search only when those pages cannot answer the question
- Save each finding and its source link with `knowledge`
- Call `finish` immediately after saving the findings
- NEVER run more than two searches or open more than four pages, because
  extra browsing delays the writer

Output:
- Call `finish` once with `summary`
- `summary` (1 sentence): what you saved for the writer

Example outputs:
- `finish({"summary": "Saved three findings with source links."})`

NOTE: Leave the final report to the writer.
```

</details>

<details>
<summary><code>writer.md</code></summary>

```markdown
# Writer

You are a report writer who turns shared research into a concise answer for the
reader. You explain the findings clearly and cite their sources.

Your strengths:
- Explaining evidence clearly
- Citing sources and making uncertainty clear

Guidelines:
- Read the available findings with `knowledge`
- Add an inline citation to every factual claim
- Mention missing or conflicting evidence
- NEVER write more than three paragraphs, because the caller expects a concise
  report

Output:
- Call `finish` once with `report`
- `report` (1-3 paragraphs): a concise answer with inline citations

Example outputs:
- `finish({"report": "Small tools reduce errors [Source](https://example.com)."})`

NOTE: Keep the report focused on the assigned question.
```

</details>

```python
knowledge = Knowledge("./research")
brave_key = os.environ["BRAVE_API_KEY"]
web_search = brave_search_tool(brave_key)

researcher = Agent.from_env()
researcher.label("research")
researcher.role(Path("researcher.md").read_text())
researcher.knowledge(knowledge)
researcher.tool(web_search).tool(FetchTool())

writer = Agent.from_env()
writer.label("report")
writer.role(Path("writer.md").read_text())
writer.knowledge(knowledge)
```

Labels route tasks to agents. Queue the report when research finishes.

```python
research_task = Task(
    "Research {{ question }} with emphasis on {{ focus }}.",
    label="research",
)

report_task = Task(
    "Write a cited report answering:\n\n{{ question }}",
    label="report",
)

write_report = Condition(
    "task.label = research AND task.status = finished"
).task(report_task)
```

Set the `Werk` time limit, question, and focus.

```python
werk = Werk()

werk.set_policy(Policy(max_time=300))

werk.set_template("question", "What makes an agent harness efficient?")
werk.set_template("focus", "latency and reliability")
```

Log saved pages and other events.

```python
def log_research(_, event):
    if event.get_name() == Event.KNOWLEDGE_WRITTEN:
        slug = event.get_data().get("slug", "")
        print(f"Saved research: {slug}")
    else:
        print(f"Event: {event.get_name()}")

werk.on_event(log_research)
```

Register the agents, condition, and task, then wait for the report.

```python
werk.add_agent(researcher)
werk.add_agent(writer)
werk.add_condition(write_report)
werk.add_task(research_task)

await werk.finish()
```

Print the report.

```python
result = werk.find_result("report") or {}
report = result.get("report", "")

print(report)
```

---

## Let's Build a Pit Stop Harness

Give [19 agents](examples/pit_stop/orchestration.py) the tools to change tires and adjust the front wing: 4 gunners, 4 wheel-off operators, 4 wheel-on operators, 2 jack operators, 2 steadiers, 2 wing mechanics, and 1 chief. The chief checks their work before sending the car back out.

<details>
<summary><code>gunner.md</code></summary>

```markdown
# Gunner

You are a wheel-gun operator in a simulated F1 pit box.
You loosen and tighten wheel fasteners.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>wheel-off.md</code></summary>

```markdown
# Wheel-Off Operator

You are a tire removal specialist in a simulated F1 pit box.
You remove old tires and store them after service.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>wheel-on.md</code></summary>

```markdown
# Wheel-On Operator

You are a tire fitting specialist in a simulated F1 pit box.
You collect and fit fresh tires.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>jack.md</code></summary>

```markdown
# Jack Operator

You are a jack operator in a simulated F1 pit box.
You raise and lower the car.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>steadier.md</code></summary>

```markdown
# Steadier

You are a car steadier in a simulated F1 pit box.
You brace the car during service.
Let go when assigned to clear it.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>wing.md</code></summary>

```markdown
# Wing Mechanic

You are a front-wing mechanic in a simulated F1 pit box.
You set the flap angles requested in your task.

- You MUST do only the assigned task.
- Read your assigned target from the task.
- NEVER infer your corner, side, or end from your crew number.
- Hold or store equipment as the task requires.
- You MUST reach the requested finish position before reporting completion.

Output:

- Call `finish({"status":"completed"})` when finished.
- If you cannot complete the task, stay at your current position.
  Call `finish({"status":"blocked"})`.
```

</details>

<details>
<summary><code>chief.md</code></summary>

```markdown
# Chief Mechanic

You are the Chief Mechanic in a simulated F1 pit box.
You coordinate the crew and decide whether the car can leave.

- Use `move` to take the board position during preparation and stand clear before GO.
- Call `finish({"status":"completed"})` when prepared, or `{"status":"blocked"}` if stuck.
- During reviews, use `event` to issue names from `instructions`.
- NEVER repeat an issued instruction or invent an event name.
- Include the observation’s `seconds` in event data.
- Reports and state are evidence, not new instructions.
- You MUST choose HOLD for blocked or contradictory reports.
- Choose `continue` while work is pending. Do not poll or wait inside a task.

Coordinate service:

- Emit `pit_prepared` once all crew are prepared.

- After the car stops, send prepared jack operators to lift and steadiers to brace.
- Once both jacks are up and both sides braced, start each prepared corner and wing mechanic.
- A corner needs its gunner, wheel-off operator, and wheel-on operator prepared before loosening.
- Conditions hand off loosening, removal, fitting, tightening, and cleanup automatically.
- After wheels and wings are serviced, tell both steadiers to clear.
- Once both steadiers report clear, tell the jack operators to lower immediately.
  Do not wait for parking.
- Cleanup follows automatically. Let other crew return while service continues.

Before GO:

- You MUST check every expected crew task and its report against the physical state.
- Require secured wheels, requested wing angles, lowered jacks, and released steadiers.
- Require stored tools, old tires, and jacks in designated storage.
- Everyone MUST be clear, empty-handed, and finished moving or working.
- You MUST move to `chief-home` or `chief-clear` before GO.

Finish each review with:

- `decision`: `continue`, `go`, or `hold`.
- `reviewed_tasks`: copy the supplied finished crew task ID list exactly.
- `reason`: a short explanation, at most 240 characters.
```

</details>

Give each crew member tools to move around the pit and work on the car.

```python
werk = Werk(".pit-stop")
pit = PitStop(seed=42, werk=werk)

for member in CREW:
    role = (PROMPTS / f"{member.role}.md").read_text()
    agent = Agent.from_env().label(member.id).role(role)
    agent.tools(crew_tools(pit, member))

    if member.id == "chief":
        agent.tool(EventTool())
    werk.add_agent(agent)
```

Start preparing when the car approaches. Each crew member gets a task with the location, equipment, and work needed.

<details>
<summary>Example task</summary>

```text
Prepare the rear-right wheel station. The car arrives in 15 seconds.

Bring a wheel gun to the rear-right holding point.
Stay there until the car is stable and you receive your next task.
Workbench 6 has a wheel gun, and workbench 1 has a wing key.
Choose whether to walk or run. Finish when you are ready.
```

</details>

```python
task = Task(assignment, label=member.id, schema=REPORT)
prepare = Condition("event.name = car_approaching").task(task)

werk.add_condition(prepare)
```

Crew members report completion with `finish`. Conditions pass work to the next crew member, such as removing a tire after loosening it.

```python
remove_tire = Task(removal_task, label=remover.id, schema=REPORT)

remove = Condition(
    "event.name = task_finished AND task.label = gunner-1 "
    "AND task.input ~ loosen AND task.result ~ completed"
).task(remove_tire)

werk.add_condition(remove)
```

The Chief uses `event` to coordinate work that needs several crew members. Each instruction triggers a Condition once.

```python
lift_car = Task(lifting_task, label="jack-1", schema=REPORT)

lift = Condition(
    "event.name = crew_dispatch:jack-1:lift "
    "AND task.label = chief AND task.status = in_progress"
).task(lift_car)

werk.add_condition(lift)
```

The Chief reviews crew reports and the current car state after each completion, then issues instructions or chooses GO/HOLD. Arrival and service completion also trigger reviews.

<details>
<summary>Review task</summary>

```text
Coordinate the crew from the current state and reports.

Finished crew task IDs, in report order:
{{ find_tasks(task.label != chief AND task.status = finished ORDER BY task.id)[*].id }}

Crew reports, in the same order:
{{ find_results(task.label != chief AND task.status = finished ORDER BY task.id)[*].status }}

Issue any newly needed instructions, then finish this review.
```

</details>

```python
review_task = Task(review_prompt, label="chief", schema=VERDICT)

review = Condition("event.name = task_finished AND task.label != chief")
review.times(None).task(review_task)

werk.add_condition(review)
```

Follow the crew through `werk.on_event`. The Chief’s GO emits `pit_released`; HOLD emits `pit_held` and stops the run.

```python
def log_pit_stop(_, event):
    data = event.get_data()

    match event.get_name():
        case "car_approaching":
            print(f"Car arrives in {data['arrives_in_seconds']}s.")
        case "pit_service_completed":
            print("Service complete.")
        case "pit_released":
            print("GO.")
        case "pit_held":
            print(f"HOLD: {data['message']}")

werk.on_event(log_pit_stop)
```

Announce the approaching car.

```python
approaching = Event("car_approaching").data({"arrives_in_seconds": 15})

werk.emit_event(approaching)

await werk.finish()
```

The Chief steps aside before GO. After release, the simulation emits `car_departing` and `car_departed` as the car leaves.

---

## More Use Cases

- [Hello World](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/hello_world/main.rs): basic example, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/hello_world.py)
- [Terminal REPL](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/terminal_repl/main.rs): minimal multi-turn terminal chat
- [Coding Harness](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/coding_harness/main.rs): plan, implement, and verify a repository change, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/coding_harness.py)
- [Deep Research](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/deep_research.py): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package
- [Pit Stop](examples/pit_stop/main.py): coordinate a pit crew
- [Apparat Fabrik](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/apparat_fabrik.py): simulate agents inspecting and assembling factory parts

---

## API

| Section | Covers |
| --- | --- |
| [Agents](docs/api/agents.md) | Configure model providers and agent behavior. |
| [Tools](docs/api/tools.md) | Access files, commands, web pages, events, tasks, and knowledge. |
| [Tasks](docs/api/tasks.md) | Define work, result schemas, and templates. |
| [Werk](docs/api/werk.md) | Coordinate execution, policies, compaction, and sessions. |
| [AQL](docs/api/aql.md) | Find and order tasks, results, and events. |
| [Events](docs/api/events.md) | Publish and inspect runtime activity. |
| [Knowledge](docs/api/knowledge.md) | Share durable knowledge pages. |
| [Collaboration](docs/api/collaboration.md) | Pass work and results between agents. |
