# Knowledge

Use `Knowledge` to store pages on disk and share them between agents and tasks.

Agents receive the index from their Werk’s shared store, opened lazily at `<werk_dir>/knowledge` by default. A standalone agent uses `.agentwerk/knowledge`. The agent’s file-access directory does not change this location. A tool targeting an explicit store does not change the prompt index.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/knowledge.gif" width="600" alt="Agents sharing knowledge" />

```rust
use agentwerk::Knowledge;

let store = Knowledge("./notes")?;
let werk = agentwerk::Werk("./session")?;
werk.set_knowledge(std::sync::Arc::clone(&store));
let alice = Agent::from_env().tool(agentwerk::tools::KnowledgeTool::default());
let bob = Agent::from_env().tool(agentwerk::tools::KnowledgeTool::default());
werk.add_agent(alice).add_agent(bob);
```

Configure the store once on Werk and register `KnowledgeTool::default()` on agents that need page access. OKF pages live at `./notes/pages/<slug>.md` and appear in `./notes/index.md`. Shared prompts include the first 12,000 index characters by default; agents can read the rest from `index.md`. Pages are always saved in full. Create entries in code:

```rust
use agentwerk::agents::knowledge::Page;

let build_page = Page {
    slug: "build-command".into(),
    kind: String::new(),
    description: "How the project is built.".into(),
    content: "Run `make` to compile.".into(),
    tags: vec!["build".into()],
};

store.get_pages().save(build_page)?;

let page = store.get_pages().get_page("build-command")?;
store.get_pages().remove("build-command")?;
```

Use these members to inspect and manage the knowledge store.

| Member | Purpose |
| --- | --- |
| `get_index()` | Get the index injected into the agent prompt. |
| `set_index_char_limit(count)` | Limit how much of the index is injected into the prompt. |
| `get_index_char_limit()` | Get the active index size limit. |
| `get_pages()` | Get the page collection. |
| `get_pages().get_all()` | Get every page in the store. |
| `clear()` | Remove every page from the store. |
