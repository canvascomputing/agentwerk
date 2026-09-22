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
  <a href="API.md">API</a> •
  <a href="../../DEVELOPMENT.md">Development</a> •
  <a href="../../SECURITY.md">Security</a>
</div>

<br />

<div align="center"><strong>Beta:</strong> The API might introduce breaking changes before <code>0.2.0</code>.</div>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/demo.gif" width="800" />
</div>
<div align="center"><a href="examples/apparat_fabrik.py">Apparat Fabrik</a></div>
<div align="center"><em>“Werk” is German for both a factory and a work of art.</em></div>

---

# Installation

```bash
pip install agentwerk
```

[Rust crate](../../README.md)

# Let's Build a Research Harness

We'll use the [Brave Search Tool](examples/web_search.py) to research a question and write a report with citations.

## Agents

Create a researcher and a writer. `Agent.from_env()` reads the provider and model from environment variables. The researcher gathers sources, while the writer turns those findings into a report.

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
from pathlib import Path

researcher_role = Path("researcher.md").read_text()
writer_role = Path("writer.md").read_text()

researcher = Agent.from_env()
researcher.label("research")
researcher.role(researcher_role)

writer = Agent.from_env()
writer.label("report")
writer.role(writer_role)
```

APIs: [Agents](API.md#agents), [Providers](API.md#providers), and [Prompt Skill](../../skills/prompt/SKILL.md).

## Tools

The researcher uses a [custom Brave Search tool](examples/web_search.py) to find sources and the built-in `FetchTool` to open them.

<details>
<summary><code>web_search.py</code></summary>

```python
def brave_search_tool(api_key: str):
    endpoint = "https://api.search.brave.com/res/v1/web/search"
    description = "Search the web and return titles, URLs, and descriptions."

    @tool(
        name="brave_search",
        description=description,
        concurrent=True,
        timeout=60,
    )
    def brave_search(query: str, count: int = 5) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        result_count = max(1, min(count, 20))
        parameters = {"q": query, "count": result_count}
        url = f"{endpoint}?{urlencode(parameters)}"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        request = Request(url, headers=headers)
        with urlopen(request, timeout=60) as response:
            body = json.load(response)

        search_results = body.get("web", {}).get("results", [])
        if not search_results:
            return "No results found."

        rendered_results = (
            f"## {result.get('title', '')}\n"
            f"{result.get('url', '')}\n"
            f"{result.get('description', '')}"
            for result in search_results
        )
        return "\n\n".join(rendered_results)

    return brave_search
```

</details>

```python
brave_key = os.environ["BRAVE_API_KEY"]
web_search = brave_search_tool(brave_key)

researcher.tool(web_search).tool(FetchTool())
```

APIs: [Tools](API.md#tools), [FetchTool](API.md#fetchtool), and [Custom tools](API.md#custom-tools).

## Tasks

Create one task for research and another for writing. Each label routes the task to the matching agent. The `question` and `focus` templates insert shared values into the prompts.

```python
research_task = Task(
    "Research {{ question }} with emphasis on {{ focus }}.",
    label="research",
)

report_task = Task(
    "Write a cited report answering:\n\n{{ question }}",
    label="report",
)
```

APIs: [Tasks](API.md#tasks), [Templates](API.md#templates), [Schemas](API.md#schemas), and [Directives](API.md#directives).

## Knowledge

Assign both agents a shared `Knowledge` base. The researcher records sourced findings there, and the writer uses that evidence to produce the report.

```python
knowledge = Knowledge.load("./research")

researcher.knowledge(knowledge)
writer.knowledge(knowledge)
```

APIs: [Knowledge](API.md#knowledge).

## Werk

A `Werk` coordinates the agents, tasks, and conditions for one run. Set the shared template values, then add the parts of the research harness.

```python
werk = Werk()

werk.set_policy(Policy(max_time=300))

werk.set_template("question", "What makes an agent harness efficient?")
werk.set_template("focus", "latency and reliability")

werk.add_agent(researcher)
werk.add_agent(writer)

write_report = Condition(
    "task.label = research AND task.status = finished"
).task(report_task)

werk.add_condition(write_report)
werk.add_task(research_task)
```

APIs: [Werk](API.md#werk), [Policy](API.md#configuration), [AQL](API.md#aql), [Collaboration](API.md#collaboration), and [Conditions](API.md#conditions).

## Events

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

APIs: [Events](API.md#events) and [Hooks](API.md#hooks).

## Results

Wait for the workflow to finish, then print the writer's report.

```python
await werk.finish()

result = werk.find_result("report") or {}
report = result.get("report", "")

print(report)
```

---

# Let's Build a Coding Harness

We’ll use a [planner and coder agent](examples/coding_harness.py) to make changes to a repository. The coder is interactive, meaning its coding task remains active until you end it.

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
def read_tools():
    return [
        ListDirectoryTool(),
        GlobTool(),
        GrepTool(),
        ReadFileTool(),
    ]

planner = (
    Agent.from_env()
    .label("plan")
    .role(Path("planner.md").read_text())
    .tools(read_tools())
)

coder = (
    Agent.from_env()
    .label("coding")
    .role(Path("coder.md").read_text())
    .interactive()
    .tools(read_tools())
    .tool(EditFileTool())
    .tool(WriteFileTool())
)
```

Both agents can inspect the working tree, but only the coder can run Rust checks. These rules limit the commands available to each agent; they do not create an operating-system sandbox.

```python
def git_tool():
    return CommandTool("git").allow("git status*").allow("git diff*")

planner.tool(git_tool())

coder.tool(git_tool()).tool(
    CommandTool("cargo")
    .allow("cargo fmt*")
    .allow("cargo check*")
    .allow("cargo test*")
)
```

A condition starts the coder with the saved plan as soon as the planner finishes.
Store the session in `./session` so you can stop the program and continue the same plan and coder conversation later.

```python
werk = Werk()
werk.set_dir("./session")

coder_task = Task(
    "Implement this plan:\n\n{{ result: plan }}",
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

APIs: [CommandTool](API.md#commandtool), [Werk](API.md#werk), [Templates](API.md#templates), [Collaboration](API.md#collaboration), [Conditions](API.md#conditions), and [Interactive agents](API.md#interactive-agents).

## More Use Cases

Example projects built with agentwerk:

- [Hello World](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/hello_world/main.rs): basic example, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/hello_world.py)
- [Terminal REPL](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/terminal_repl/main.rs): minimal multi-turn terminal chat
- [Coding Harness](https://github.com/canvascomputing/agentwerk/blob/main/crates/use-cases/src/coding_harness/main.rs): plan, implement, and verify a repository change, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/coding_harness.py)
- [Deep Research](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/deep_research.py): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package
- [Apparat Fabrik](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/apparat_fabrik.py): simulate agents inspecting and assembling factory parts
