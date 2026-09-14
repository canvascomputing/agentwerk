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

## Introduction

Create `Agents` with `Tools` and assign them `Tasks` to create results. Coordinate their work in a `Werk`, observe detailed `Events`, and derive `Knowledge`.

Build a file explorer by creating an `Agent`, giving it a `ListDirectoryTool`, adding a `Task`, and calling `finish_task` to collect the result.

```rust
use agentwerk::tools::ListDirectoryTool;
use agentwerk::Agent;

#[tokio::main]
async fn main() {
    let agent = Agent::from_env()
        .role("You are a filesystem explorer.")
        .tool(ListDirectoryTool);

    let task = agent.add_task("Explore this directory.");
    let result = agent.finish_task(task).await.unwrap();

    println!("{result}");
}
```

Further reading:

- [Agents](API.md#agents): [Providers](API.md#providers), [Interactive agents](API.md#interactive-agents)
- [Tools](API.md#tools) and [Tasks](API.md#tasks)

## Tools, Tasks, and Schemas

Build a release check by giving an `Agent` an allowlisted `CommandTool`, inserting the target branch through a template, and constraining its `Task` with a `Schema`.

```rust
use agentwerk::tools::CommandTool;
use agentwerk::{Agent, Task};

#[tokio::main]
async fn main() {
    let git = CommandTool::new("git")
        .allow("git branch --show-current")
        .allow("git status --short");

    let agent = Agent::from_env()
        .role("You are a release engineer.")
        .template("release_branch", "main")
        .tool(git);

    let schema = release_status_schema();
    let task = Task::new(
        "Check whether this repository is ready to release from \
         {{ release_branch }}.",
    )
    .schema(schema);

    let task = agent.add_task(task);
    let result = agent.finish_task(task).await.unwrap();

    println!("{result}");
}
```

<details>
<summary>Result schema</summary>

```rust
use agentwerk::schemas::Schema;
use serde_json::json;

fn release_status_schema() -> Schema {
    let schema = json!({
        "type": "object",
        "properties": {
            "branch": { "type": "string" },
            "clean": { "type": "boolean" },
            "ready": { "type": "boolean" }
        },
        "required": ["branch", "clean", "ready"],
        "additionalProperties": false
    });

    Schema::new(schema).expect("release status schema is valid")
}
```

</details>

Further reading:

- [Tools](API.md#tools): [CommandTool](API.md#commandtool), [FetchTool](API.md#fetchtool), [Custom tools](API.md#custom-tools), [Timeouts](API.md#timeouts)
- [Tasks](API.md#tasks): [Templates](API.md#templates), [Schemas](API.md#schemas), [Directives](API.md#directives)

## Werk, Events, and Knowledge

Build a two-stage research workflow by loading shared `Knowledge`, giving the research `Agent` [Brave Search](crates/use-cases/src/deep_research/web_search.rs) and `FetchTool`, coordinating both `Agents` in a `Werk`, observing their `Events`, and using an AQL `Condition` to start the writing `Task`.

<details>
<summary><code>researcher.md</code></summary>

```markdown
# Researcher

Research the question for the Writer. Save verified evidence and source
links in shared knowledge.

Your strengths:

- Finding and verifying primary sources

Guidelines:

- Search with Brave. Open every source with `fetch` before using it,
  because search summaries are only leads
- IMPORTANT: Record conflicts and gaps instead of guessing

Output:

- Save two to four knowledge pages, then call `finish` once with a
  `summary` of no more than three sentences

Example outputs:

- "Saved three verified findings and one unresolved gap."

NOTE: Stop when the Writer has enough evidence to answer the question.
```

</details>

<details>
<summary><code>writer.md</code></summary>

```markdown
# Writer

Answer the question for the reader using shared knowledge.

Your strengths:

- Turning evidence into a concise explanation

Guidelines:

- Read the available knowledge pages before drafting the report
- Link every factual claim to its source
- NEVER add facts absent from shared knowledge, because this agent has
  no research tools

Output:

- Call `finish` once with a `title` of no more than 80 characters and a
  `report` of three to five paragraphs

Example outputs:

- "The evidence supports lower latency, but not lower cost for every
  workload."

NOTE: Return one grounded report and nothing beyond it.
```

</details>

```rust
mod web_search;

use agentwerk::tools::FetchTool;
use agentwerk::{Agent, Condition, Knowledge, Task, Werk};
use web_search::brave_search_tool;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let researcher_role = include_str!("researcher.md");
    let writer_role = include_str!("writer.md");

    let question = "What makes an agent harness efficient?";
    let knowledge = Knowledge::load(".agentwerk/research")?;
    let brave_key = std::env::var("BRAVE_API_KEY")?;
    let brave_search = brave_search_tool(brave_key);

    let researcher = Agent::from_env()
        .label("research")
        .role(researcher_role)
        .knowledge(&knowledge)
        .tool(brave_search)
        .tool(FetchTool::new());

    let writer = Agent::from_env()
        .label("report")
        .role(writer_role)
        .knowledge(&knowledge);

    let werk = Werk::new();
    werk.on_event(|_, event| eprintln!("{}", event.get_name()));
    werk.add_agent(researcher);
    werk.add_agent(writer);

    let write_report = Condition::new("task.label = research AND task.status = finished")
        .task(Task::labeled("report", question));

    werk.add_condition(write_report);
    werk.add_task(Task::labeled("research", question));
    werk.finish().await;

    let result = werk.find_result("report").unwrap();

    println!("{}", result["report"].as_str().unwrap_or_default());
    Ok(())
}
```

Further reading:

- [Werk](API.md#werk): [AQL](API.md#aql), [Collaboration](API.md#collaboration), [Hooks](API.md#hooks), [Conditions](API.md#conditions), [Configuration](API.md#configuration), [Compaction](API.md#compaction), [Sessions](API.md#sessions)
- [Events](API.md#events) and [Knowledge](API.md#knowledge)
