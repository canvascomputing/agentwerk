<div align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/logo.png" width="200" />
</div>

<h1 align="center">agentwerk (Python)</h1>

<div align="center">
  <strong>A minimal agentic loop for building efficient harnesses.</strong>
</div>

<div align="center">Coordinate agent fleets across complex tasks, with detailed observability and shared knowledge.</div>

<div align="center">
  <a href="#installation">Installation</a> •
  <a href="../../README.md">Rust</a> •
  <a href="#api">API</a> •
  <a href="../../DEVELOPMENT.md">Development</a> •
  <a href="../../SECURITY.md">Security</a>
</div>

<br />

<div align="center"><strong>Beta:</strong> The API might introduce breaking changes before <code>0.2.0</code>.</div>

---

<div align="center">
  <img src="../../assets/demo.gif" width="800" />
</div>
<div align="center"><a href="examples/pit_stop/README.md">Pit Stop — 19 agents, 52 tasks, one coordinated release</a></div>
<div align="center"><em>“Werk” is German for both a factory and a work of art.</em></div>

---

## Installation

```bash
pip install agentwerk
```

[Rust crate](../../README.md)

---

## Let's Build a Coding Harness

We’ll use a [planner and coder agent](examples/coding_harness.py) to make changes to a repository.

Give both agents read-only repository tools, then add editing tools only to the coder. The planner returns a grounded plan for the coder to implement.

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
planner = (
    Agent.from_env()
    .label("plan")
    .role(Path("planner.md").read_text())
    .tool(ListDirectoryTool())
    .tool(GlobTool())
    .tool(GrepTool())
    .tool(ReadFileTool())
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
)
```

Both agents can inspect the working tree, but only the coder can run Rust checks.

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

planner.tool(git_tool())

coder.tool(git_tool()).tool(cargo_tool())
```

A condition starts the coder with the saved plan as soon as the planner finishes.
Store the session in `./session` so you can stop the program and continue the same plan and coder conversation later.

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

Register the agents and condition, add the planning task, then run until the interactive coder pauses.

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

We'll research a question and write a report with citations. Start with a researcher and a writer, then give the researcher a [custom Brave Search tool](examples/web_search.py) and the built-in `FetchTool`.

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

Create one task for research and another for writing. Each label routes the task to the matching agent. A condition queues the report after the research finishes.

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

Create the `Werk`, limit the run to five minutes, and set the question and research focus.

```python
werk = Werk()

werk.set_policy(Policy(max_time=300))

werk.set_template("question", "What makes an agent harness efficient?")
werk.set_template("focus", "latency and reliability")
```

Observe each knowledge page as it is saved and log every other event by name.

```python
def log_research(_, event):
    if event.get_name() == Event.KNOWLEDGE_WRITTEN:
        slug = event.get_data().get("slug", "")
        print(f"Saved research: {slug}")
    else:
        print(f"Event: {event.get_name()}")

werk.on_event(log_research)
```

Add the two agents, the report condition, and the research task, then wait for the report.

```python
werk.add_agent(researcher)
werk.add_agent(writer)
werk.add_condition(write_report)
werk.add_task(research_task)

await werk.finish()
```

Read and print the writer's report.

```python
result = werk.find_result("report") or {}
report = result.get("report", "")

print(report)
```

---

## More Use Cases

Example projects built with agentwerk:

- [Hello World](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/hello_world/main.rs): basic example, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/hello_world.py)
- [Terminal REPL](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/terminal_repl/main.rs): minimal multi-turn terminal chat
- [Coding Harness](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/coding_harness/main.rs): plan, implement, and verify a repository change, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/coding_harness.py)
- [Deep Research](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/deep_research.py): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package
- [Pit Stop](examples/pit_stop/README.md): a 3D pit stop with 19 agents, 52 tasks, visible equipment handoffs, and a verified release
- [Apparat Fabrik](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/apparat_fabrik.py): simulate agents inspecting and assembling factory parts

---

## API

| Section | Covers |
| --- | --- |
| [Agents](docs/api/agents.md) | Configure model providers and agent behavior. |
| [Tools](docs/api/tools.md) | Give agents access to files, commands, web pages, events, tasks, and knowledge. |
| [Tasks](docs/api/tasks.md) | Define work, result schemas, and templates. |
| [Werk](docs/api/werk.md) | Coordinate execution, policies, compaction, and sessions. |
| [AQL](docs/api/aql.md) | Find and order tasks, results, and events. |
| [Events](docs/api/events.md) | Publish and inspect runtime activity. |
| [Knowledge](docs/api/knowledge.md) | Share durable pages between agents and tasks. |
| [Collaboration](docs/api/collaboration.md) | Pass work and results between agents. |
