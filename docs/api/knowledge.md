# Knowledge

Use `Knowledge` to store pages on disk and share them between agents and tasks.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/knowledge.gif" width="600" alt="Agents sharing knowledge" />

```rust
use agentwerk::Knowledge;

let store = Knowledge("./notes")?;
let alice = Agent().knowledge(&store);
let bob = Agent().knowledge(&store);
```

Calling `.knowledge(&store)` registers a `KnowledgeTool` for shared pages. OKF pages live at `./notes/pages/<slug>.md` and appear in `./notes/index.md`. Shared prompts include the first 12,000 index characters by default; agents can read the rest from `index.md`. Pages are always saved in full. Create entries in code:

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
