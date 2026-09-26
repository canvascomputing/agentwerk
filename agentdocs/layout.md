# Layout

Where code, tests, bindings, examples, and repository guidance live.

## Workspace

**Keep each crate responsible for one layer.**

- `crates/agentwerk/` contains the public Rust library.
- `crates/agentwerk-py/` contains the PyO3 bindings and the pure-Python package surface.
- `crates/use-cases/` contains runnable examples and depends on the libraries, never the reverse.
- Keep `agentwerk-py` outside `default-members` because its extension links against Python.

## Library Root

**Place code under the domain that owns its behavior.**

- `src/lib.rs` declares modules and re-exports the small root API.
- `src/codegrep/` contains the structural matcher used by `GrepTool`.
- `src/event.rs` owns `Event`; `src/persistence.rs` owns shared file primitives and stays crate-private.
- `src/agents/` owns agent configuration, tasks, orchestration, policy, queries, statistics, retries, compaction, and knowledge.
- `src/providers/`, `src/tools/`, and `src/schemas/` own LLM providers, agent actions, and result validation.
- `src/prompts/prompt.rs` owns Werk's single-pass template substitution and AQL expressions. `json_path.rs` parses and evaluates JSON paths over selected tasks, results, or events. `prompts/mod.rs` supplies runtime string values and corrective templates. Agent and Task pass prompts to Werk.

## Agents and Tasks

**Separate orchestration state from one agent's current operation.**

- `agents/agent.rs` configures `Agent`; `agents/condition.rs` configures runtime AQL actions; `agents/tasks/werk.rs` exposes `Werk`.
- `agents/tasks/` owns `Task`, `Reply`, storage transitions, and task errors.
- `agents/loop/` splits execution into the main scheduler, per-agent work, provider requests, compaction, and tool calls.
- `agents/query.rs` owns AQL; `policy.rs`, `stats.rs`, and `retry.rs` own limits, statistics, and retry timing.
- `agents/knowledge.rs` owns `Knowledge`, pages, and the OKF bundle rooted at the directory passed to `Knowledge(path)`.

## Providers

**Keep vendor formats at the edge and shared transport in the center.**

- `providers/provider.rs` defines `ProviderLike`, `Provider`, and the internal `Protocol` boundary.
- `anthropic.rs`, `openai.rs`, `mistral.rs`, and `litellm.rs` adapt vendor authentication and payloads.
- `endpoint.rs` owns HTTP execution; `stream.rs` and `frames.rs` rebuild model responses.
- `types.rs`, `error.rs`, `model.rs`, and `environment.rs` own shared values, failures, model settings, and environment selection.

## Tools and Prompts

**Keep each built-in tool beside its model-facing contract.**

- `tools/tool.rs` owns the public `Tool` builder and internal execution context.
- A built-in tool keeps its Rust implementation, `<name>.tool.md`, and `<name>.schema.json` together.
- `tools/command/` and `tools/task/` use submodules because parsing and completion have separate concerns.
- `prompts/templates.rs` indexes the bundled entries under `prompts/templates/*.md`; `prompt.rs` contains private rendering, and `mod.rs` supplies runtime values.

## Python Bindings

**Mirror Rust concepts without duplicating Rust behavior.**

- `crates/agentwerk-py/src/` has one binding module per exposed domain, including `policy.rs` and `providers.rs`.
- `src/convert.rs` owns Python and JSON conversion helpers; `src/lib.rs` registers the extension surface.
- `python/agentwerk/__init__.py` re-exports `_agentwerk` and owns the pure-Python `@tool` decorator.
- `python/agentwerk/__init__.pyi` declares the Python surface and is checked by `tests/test_parity.py`.

## Tests and Repository Docs

**Put verification next to its layer and keep inventories separate from conventions.**

- Keep Rust unit tests inline under `#[cfg(test)]`; keep live-provider tests under `crates/agentwerk/tests/integration/`.
- Keep Python tests under `crates/agentwerk-py/tests/` and use the `live` marker for provider-dependent cases.
- Keep use-case-specific tests inside their binary modules so the binary test pass reaches them.
- Keep the Rust API reference under `docs/api/` and mirror it for Python under `crates/agentwerk-py/docs/api/`; each README is the index for its language.
- Use `INVENTORY.md` for declaration-level API tracking; use `agentdocs/` only for decisions the code does not state.
- Keep reusable agent skills under `skills/`, hook configuration under `hooks/`, and repository checks under `tools/`.

## Browser Use Cases

**Keep visualization and simulation in the consuming example.**

- `crates/agentwerk-py/examples/pit_stop/` is a uv project using the local Python binding and an example-local Vite/Three.js frontend. Its shared `layout.json` supplies car, staging and equipment geometry; `movement.py` plans explicit crew journeys; `sim_clock.py` pauses shared physical time while agents decide. Seeded setup and authoritative mechanical state live in `simulation.py`. Version 6 records destination facing, jack grip geometry and clearance, and visible parking groups; replay retains earlier formats. The frontend separates recorded motion, persistent equipment, hand articulation, and milestone-derived titles and timers; it bundles its pixel font with its license under `public/licenses/`.
- Keep agent composition, mechanical state, transport, playback, and rendering separate. All authoritative simulation changes originate in validated physical tools; the browser observes events.
- Commit its lockfiles and successful real-agent recording. Keep dependency installations, frontend builds, and capture intermediates untracked.
