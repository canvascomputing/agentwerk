<div align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/logo.png" width="200" />
</div>

<h1 align="center">agentwerk</h1>

<div align="center">
  <strong>A minimal agentic loop for building efficient harnesses.</strong>
</div>

<div align="center">Coordinate agent fleets across complex tasks, with detailed observability and shared knowledge.</div>

<div align="center">
  <a href="#installation">Installation</a> •
  <a href="crates/agentwerk-py/README.md">Python</a> •
  <a href="API.md">API</a> •
  <a href="DEVELOPMENT.md">Development</a> •
  <a href="SECURITY.md">Security</a>
</div>

<br />

<div align="center"><strong>Beta:</strong> The API might introduce breaking changes before <code>0.2.0</code>.</div>

---

<div align="center">
  <img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/demo.gif" width="800" />
</div>
<div align="center"><a href="crates/agentwerk-py/examples/apparat_fabrik.py">Apparat Fabrik</a></div>
<div align="center"><em>“Werk” is German for both a factory and a work of art.</em></div>

---

## Installation

```bash
cargo add agentwerk
```

[Python bindings](crates/agentwerk-py/README.md)

---

## Let's Build a Coding Harness

We’ll use a [planner and coder agent](crates/use-cases/src/coding_harness/main.rs) to make changes to a repository.

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

```rust
fn read_tools() -> Vec<Tool> {
    vec![
        ListDirectoryTool.into(),
        GlobTool.into(),
        GrepTool.into(),
        ReadFileTool.into(),
    ]
}

let planner = Agent::from_env()
    .label("plan")
    .role(include_str!("planner.md"))
    .tools(read_tools());

let coder = Agent::from_env()
    .label("coding")
    .role(include_str!("coder.md"))
    .interactive()
    .tools(read_tools())
    .tool(EditFileTool)
    .tool(WriteFileTool);
```

Both agents can inspect the working tree, but only the coder can run Rust checks.

```rust
let git = || CommandTool("git")
    .allow("git status*")
    .allow("git diff*");

let cargo = || CommandTool("cargo")
    .allow("cargo fmt*")
    .allow("cargo check*")
    .allow("cargo test*");

let planner = planner.tool(git());

let coder = coder.tool(git()).tool(cargo());
```

A condition starts the coder with the saved plan as soon as the planner finishes.
Store the session in `./session` so you can stop the program and continue the same plan and coder conversation later.

```rust
let werk = Werk("./session")?;

let coder_task = Task("Implement this plan:\n\n{{ result: plan }}")
    .label("coding");

let start_coder = Condition("task.label = plan AND task.status = finished")
    .task(coder_task);
```

Register the agents and condition, add the planning task, then run until the interactive coder pauses.

```rust
let plan_task = Task("Add a --dry-run flag to the database migration command.")
    .label("plan");

werk.add_agent(planner);
werk.add_agent(coder);
werk.add_condition(start_coder);
werk.add_task(plan_task);

werk.finish().await;
```

APIs: [CommandTool](API.md#commandtool), [Templates](API.md#templates), [Collaboration](API.md#collaboration), [Conditions](API.md#conditions), [Sessions](API.md#sessions), and [Interactive agents](API.md#interactive-agents).

---

## Let's Build a Research Harness

We'll research a question and write a report with citations. Start with a researcher and a writer, then give the researcher a [custom Brave Search tool](crates/use-cases/src/deep_research/web_search.rs) and the built-in `FetchTool`.

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

```rust
let knowledge = Knowledge("./research")?;
let brave_key = std::env::var("BRAVE_API_KEY")?;
let web_search = brave_search_tool(brave_key);

let researcher = Agent::from_env()
    .label("research")
    .role(include_str!("researcher.md"))
    .knowledge(&knowledge)
    .tool(web_search)
    .tool(FetchTool);

let writer = Agent::from_env()
    .label("report")
    .role(include_str!("writer.md"))
    .knowledge(&knowledge);
```

Create one task for research and another for writing. Each label routes the task to the matching agent. A condition queues the report after the research finishes.

```rust
let research_task = Task("Research {{ question }} with emphasis on {{ focus }}.")
    .label("research");

let report_task = Task("Write a cited report answering:\n\n{{ question }}")
    .label("report");

let write_report = Condition("task.label = research AND task.status = finished")
    .task(report_task);
```

Create the `Werk`, limit the run to five minutes, and set the question and research focus.

```rust
let werk = Werk(".agentwerk")?;

werk.set_policy(Policy {
    max_time: Some(Duration::from_secs(300)),
    ..Default::default()
});

werk.set_template("question", "What makes an agent harness efficient?");
werk.set_template("focus", "latency and reliability");
```

Observe each knowledge page as it is saved and log every other event by name.

```rust
werk.on_event(|_, event| {
    if event.get_name() == Event::KNOWLEDGE_WRITTEN {
        let slug = event.get_data()["slug"].as_str().unwrap_or_default();
        eprintln!("Saved research: {slug}");
    } else {
        eprintln!("Event: {}", event.get_name());
    }
});
```

Add the two agents, the report condition, and the research task, then wait for the report.

```rust
werk.add_agent(researcher);
werk.add_agent(writer);

werk.add_condition(write_report);
werk.add_task(research_task);

werk.finish().await;
```

Read and print the writer's report.

```rust
let result = werk.find_result("report").unwrap();
let report = result["report"].as_str().unwrap_or_default();

println!("{report}");
```

APIs: [Agents](API.md#agents), [Tools](API.md#tools), [Tasks](API.md#tasks), [Knowledge](API.md#knowledge), [Werk](API.md#werk), [Events](API.md#events), [Conditions](API.md#conditions), and [Collaboration](API.md#collaboration).

---

## More Use Cases

Example projects built with agentwerk:

- [Hello World](crates/use-cases/src/hello_world/main.rs): basic example
- [Terminal REPL](crates/use-cases/src/terminal_repl/main.rs): minimal multi-turn terminal chat
- [Coding Harness](crates/use-cases/src/coding_harness/main.rs): plan, implement, and verify a repository change
- [Deep Research](crates/use-cases/src/deep_research/main.rs): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package
