# Knowledge

Use `Knowledge` to store pages on disk and share them between agents and tasks.

<img src="https://raw.githubusercontent.com/canvascomputing/agentwerk/main/assets/knowledge.gif" width="600" alt="Agents sharing knowledge" />

```python
from agentwerk import Agent, Knowledge

store = Knowledge("./notes")
alice = Agent().knowledge(store)
bob = Agent().knowledge(store)
```

Calling `.knowledge(store)` registers a `KnowledgeTool` for shared pages. OKF pages live at `./notes/pages/<slug>.md` and appear in `./notes/index.md`. Shared prompts include the first 12,000 index characters by default; agents can read the rest from `index.md`. Pages are always saved in full. Create entries in code:

```python
from agentwerk import Page

build_page = Page(
    "build-command",
    "How the project is built.",
    "Run `make` to compile.",
    tags=["build"],
)

store.get_pages().save(build_page)

page = store.get_pages().get_page("build-command")
store.get_pages().remove("build-command")
```

Use these members to inspect and manage the knowledge store.

| Member | Purpose |
| --- | --- |
| `get_index()` | Get the index, which is injected into the agent prompt. |
| `set_index_char_limit(count)` | Limit how much of the index is injected into the prompt. |
| `get_index_char_limit()` | Get the index size limit in force. |
| `get_pages()` | Get the page collection for reading and writing pages. |
| `get_pages().get_all()` | Get every page in the store. |
| `clear()` | Remove every page from the store. |
