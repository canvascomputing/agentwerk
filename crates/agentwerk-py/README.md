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

## Installation

```bash
pip install agentwerk
```

[Rust crate](../../README.md)

## Let's Build a Research Workflow

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

APIs: [Agents](API.md#agents), [Providers](API.md#providers), and [Interactive agents](API.md#interactive-agents).

## Tools

The researcher needs the [Brave Search Tool](examples/web_search.py) and `FetchTool` to gather web sources. The writer works from shared knowledge.

```python
brave_key = os.environ["BRAVE_API_KEY"]
brave_search = brave_search_tool(brave_key)
researcher.tool(brave_search).tool(FetchTool())
```

APIs: [Tools](API.md#tools), [FetchTool](API.md#fetchtool), and [Custom tools](API.md#custom-tools).

## Tasks

Create one task for research and another for writing. Each label routes the task to the matching agent. The `{{ question }}` placeholder inserts a shared value into both prompts.

```python
task_prompt = "{{ question }}"
research_task = Task(task_prompt, label="research")
report_task = Task(task_prompt, label="report")
```

APIs: [Tasks](API.md#tasks), [Templates](API.md#templates), [Schemas](API.md#schemas), and [Directives](API.md#directives).

## Knowledge

Use one `Knowledge` store for both agents, so the findings remain available between runs. The researcher writes the findings, and the writer reads them.

```python
knowledge = Knowledge.load(".agentwerk/research")
researcher.knowledge(knowledge)
writer.knowledge(knowledge)
```

APIs: [Knowledge](API.md#knowledge).

## Werk

Add both agents and the research task to a `Werk`. Set the shared template value, then use an AQL condition to queue the report task after the research finishes.

```python
werk = Werk()
werk.set_template("question", "What makes an agent harness efficient?")
werk.add_agent(researcher)
werk.add_agent(writer)

finished_research = "task.label = research AND task.status = finished"
write_report = Condition(finished_research)
write_report.task(report_task)

werk.add_condition(write_report)
werk.add_task(research_task)
```

APIs: [Werk](API.md#werk), [AQL](API.md#aql), [Collaboration](API.md#collaboration), and [Conditions](API.md#conditions).

## Events

Log each event while the workflow runs, then print the writer's report.

```python
werk.on_event(lambda _, event: print(event.get_name()))
await werk.finish()

result = werk.find_result("report") or {}
report = result.get("report", "")
print(report)
```

APIs: [Events](API.md#events) and [Hooks](API.md#hooks).

## Use Cases

Example projects built with agentwerk:

- [Hello World](https://github.com/canvascomputing/agentwerk/tree/main/crates/use-cases/src/hello_world/): basic example, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/hello_world.py)
- [Terminal REPL](https://github.com/canvascomputing/agentwerk/tree/main/crates/use-cases/src/terminal_repl/): minimal multi-turn terminal chat
- [Editorial Review](https://github.com/canvascomputing/agentwerk/tree/main/crates/use-cases/src/editorial_review/): route a draft through an editor with a result hook and AQL, also available as a [Python example](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/editorial_review.py)
- [Deep Research](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/deep_research.py): research across several sources (requires `BRAVE_API_KEY`)
- [Malware Scanner](https://github.com/canvascomputing/malwi): find signs of malware in a software package
- [Apparat Fabrik](https://github.com/canvascomputing/agentwerk/blob/main/crates/agentwerk-py/examples/apparat_fabrik.py): simulate agents inspecting and assembling factory parts

> Configure an LLM provider first (see [Environment](https://github.com/canvascomputing/agentwerk/blob/main/DEVELOPMENT.md#environment)).

```bash
python examples/editorial_review.py "Draft a short release announcement."
```
