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

## Let's Build a Research Workflow

We'll use the [Brave Search Tool](crates/use-cases/src/deep_research/web_search.rs) to research a question and write a report with citations.

## Agents

Create a researcher and a writer. `Agent::from_env()` reads the provider and model from environment variables. The researcher gathers sources, while the writer turns those findings into a report.

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
let researcher_role = include_str!("researcher.md");
let writer_role = include_str!("writer.md");

let researcher = Agent::from_env()
    .label("research")
    .role(researcher_role);

let writer = Agent::from_env()
    .label("report")
    .role(writer_role);
```

APIs: [Agents](API.md#agents), [Providers](API.md#providers), and [Prompt Skill](skills/prompt/SKILL.md).

## Tools

The researcher uses a [custom Brave Search tool](crates/use-cases/src/deep_research/web_search.rs) to find sources and the built-in `FetchTool` to open them.

<details>
<summary><code>web_search.rs</code></summary>

```rust
fn brave_search_tool(api_key: String) -> Tool {
    let endpoint = "https://api.search.brave.com/res/v1/web/search";
    let description = "Search the web and return titles, URLs, and descriptions.";
    let schema = json!({
        "type": "object",
        "properties": {
            "query": { "type": "string", "description": "The search query." },
            "count": { "type": "integer", "description": "Results to return, from 1 to 20. Defaults to 5." }
        },
        "required": ["query"]
    });
    let handler = move |input: Value| {
        let api_key = api_key.clone();
        async move {
            let query = input["query"].as_str().unwrap_or_default().trim();
            if query.is_empty() {
                return Event::tool_call_failed("query must not be empty");
            }
            let result_count = input["count"]
                .as_u64()
                .unwrap_or(5)
                .clamp(1, 20)
                .to_string();

            let request = reqwest::Client::new()
                .get(endpoint)
                .query(&[("q", query), ("count", &result_count)])
                .header("X-Subscription-Token", api_key)
                .header("Accept", "application/json");
            let response = match request.send().await {
                Ok(response) => response,
                Err(error) => {
                    return Event::tool_call_failed(format!("Brave search failed: {error}"));
                }
            };
            if !response.status().is_success() {
                return Event::tool_call_failed(format!(
                    "Brave search returned {}",
                    response.status()
                ));
            }

            let body = match response.json::<Value>().await {
                Ok(body) => body,
                Err(error) => {
                    return Event::tool_call_failed(format!(
                        "Brave returned invalid JSON: {error}"
                    ));
                }
            };
            let search_results = body["web"]["results"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let rendered_results = search_results
                .iter()
                .map(|result| {
                    format!(
                        "## {}\n{}\n{}",
                        result["title"].as_str().unwrap_or_default(),
                        result["url"].as_str().unwrap_or_default(),
                        result["description"].as_str().unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let output = if rendered_results.is_empty() {
                "No results found.".to_string()
            } else {
                rendered_results
            };

            Event::tool_call_finished(output)
        }
    };

    Tool::new("brave_search")
        .description(description)
        .schema(schema)
        .concurrent(true)
        .handler(handler)
}
```

</details>

```rust
let brave_key = std::env::var("BRAVE_API_KEY")?;
let web_search = brave_search_tool(brave_key);

let researcher = researcher
    .tool(web_search)
    .tool(FetchTool::new());
```

APIs: [Tools](API.md#tools), [FetchTool](API.md#fetchtool), and [Custom tools](API.md#custom-tools).

## Tasks

Create one task for research and another for writing. Each label routes the task to the matching agent. The `question` and `focus` templates insert shared values into the prompts.

```rust
let research_task = Task::labeled(
    "research",
    "Research {{ question }} with emphasis on {{ focus }}.",
);

let report_task = Task::labeled(
    "report",
    "Write a cited report answering:\n\n{{ question }}",
);
```

APIs: [Tasks](API.md#tasks), [Templates](API.md#templates), [Schemas](API.md#schemas), and [Directives](API.md#directives).

## Knowledge

Assign both agents a shared `Knowledge` base. The researcher records sourced findings there, and the writer uses that evidence to produce the report.

```rust
let knowledge = Knowledge::load("./research")?;

let researcher = researcher.knowledge(&knowledge);
let writer = writer.knowledge(&knowledge);
```

APIs: [Knowledge](API.md#knowledge).

## Werk

Add both agents and the research task to a `Werk`. Set the shared template values and use an AQL condition to queue the report task after the research finishes.

```rust
let werk = Werk::new();

werk.set_template("question", "What makes an agent harness efficient?");
werk.set_template("focus", "latency and reliability");

werk.add_agent(researcher);
werk.add_agent(writer);

let write_report = Condition::new("task.label = research AND task.status = finished")
    .task(report_task);

werk.add_condition(write_report);
werk.add_task(research_task);
```

APIs: [Werk](API.md#werk), [AQL](API.md#aql), [Collaboration](API.md#collaboration), and [Conditions](API.md#conditions).

## Events

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

APIs: [Events](API.md#events) and [Hooks](API.md#hooks).

## Results

Wait for the workflow to finish, then print the writer's report.

```rust
werk.finish().await;

let result = werk.find_result("report").unwrap();
let report = result["report"].as_str().unwrap_or_default();

println!("{report}");
```

APIs: [Werk](API.md#werk) and [AQL](API.md#aql).

## More Use Cases

Example projects built with agentwerk:

- [Hello World](crates/use-cases/src/hello_world/main.rs): basic example
- [Terminal REPL](crates/use-cases/src/terminal_repl/main.rs): minimal multi-turn terminal chat
- [Editorial Review](crates/use-cases/src/editorial_review/main.rs): route a draft through an editor with a result hook and AQL
- [Deep Research](crates/use-cases/src/deep_research/main.rs): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package

> Configure an LLM provider first (see [Environment](DEVELOPMENT.md#environment)).

```bash
make use_case name=hello-world
make use_case name=terminal-repl
make use_case name=editorial-review args='"Draft a short release announcement."'
make use_case name=deep-research args='"What makes an agent harness efficient?"'
make use_case name=malware-scanner args=./path/to/package
```
