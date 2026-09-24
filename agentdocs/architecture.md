# Architecture

The invariants that govern orchestration, tools, providers, events, and durable state.

## Ownership and Concurrency

**A `Werk` owns shared execution state; each `Agent` processes one claimed task at a time.**

- Configure an `Agent`, then drive its private or shared Werk with `start` or a completion method on either type. Agent execution registers its configured copy only once.
- Delegate Agent completion to Werk without forcing a restart; on a shared Werk, Agent queries can select any task and `finish` waits for the entire Werk.
- Let `Werk::bind_agent` move tasks queued on an agent's private Werk into the shared Werk.
- Claim each task once and release Werk locks before `ProviderLike::respond` or a tool handler is awaited.
- Keep one `Werk` as the orchestration boundary; nested Werks are not supported.

## Prompts

**Prompt rendering is a private Werk implementation detail; callers provide strings and shared template values.**

- Delegate Agent setters to Werk. Import missing templates when binding; destination values win.
- Parse double-brace syntax once in `prompts/prompt.rs`; use strict expression resolution for prompts and configurable corrective templates, and infallible named resolution only for bundled text rendered without a Werk. Never scan inserted values.
- Name prompt selectors after Werk's `find_*` methods and allow one layer of template variables inside their AQL arguments as raw source.
- Parse a field or bracket JSON path after the selector call before expanding query variables. Evaluate it through `prompts/json_path.rs` before plain formatting; named template values do not support paths.
- Let Werk snapshot shared templates once when it prepares a role and initial string task, and report failures through `prompt_render_failed` before the first request.
- Read corrective templates, including custom event responses, from the current Werk values at use time. Bind call-specific fields before shared values and keep inserted values literal.
- Freeze the complete rendered system prompt at the task's first request. Reuse its earliest persisted system reply through later turns, retries, continuation, compaction, and reload.
- Record the prepared task message and one frozen system prompt in replies. Keep shared templates as runtime configuration rather than persisted session data; ignore legacy captured template fields when loading tasks.
- Keep later messages literal. AQL selects available results without waiting or creating dependencies.

## Assignment and Identity

**Use labels for assignment and generated IDs for ownership.**

- Match an unlabelled agent only to unlabelled tasks; match a labelled agent only to the same task label.
- Use a unique label when a task must target one agent; agents sharing a label form a pool.
- Let `Agent::get_id` assign `<label>-<n>` or `agent-<n>` and store that ID in `Task::assignee` when claimed.
- Recreate agents in the same label order when resuming a session, because started tasks resume only on the same generated ID.

## Queries and Lifecycle

**Use origin-qualified fields in AQL selections over tasks or events.**

- Keep `Query` non-generic; its namespaced fields infer a private task, event, or joined source at runtime. A joined query evaluates `(Task, Event)` pairs linked through `Event::task_id`, and every finder or lifecycle operation projects those matches to tasks.
- Snapshot task IDs when an event or joined query enters a lifecycle operation. Keep task-only cancellation queries live so they also apply to tasks added later.
- Treat a lone value as `task.label = value`, except that a bare `t-<n>` takes precedence as `task.id = t-<n>`. Keep fields in full expressions origin-qualified.
- Use `Query::new` for runtime input so invalid AQL returns `QueryError`; infallible string conversions may panic.
- Define pending work as unfinished, uncancelled work selected by the query; a task paused for caller input does not keep a completion wait open.
- Keep cancellation scoped to the current run: `start()` clears cancellation without changing `Status`.

## Conditions

**Release condition actions up to their per-run trigger count from live AQL matches.**

- Keep condition definitions and remaining trigger counts in memory; loading a Werk never restores them or makes loaded records eligible.
- Test events against conditions registered by the time condition evaluation begins. A condition added by a synchronous handler sees that handler's event; completed earlier events are never replayed.
- Claim finite trigger capacity before releasing the registry lock, then add unregistered agents before tasks. Recursive matches consume finite capacity; an unlimited condition that reproduces its match can recurse without bound.
- Restore configured trigger counts on each actual start. Never count an unmet condition as pending work.

## Completion

**Preserve one result value across every completion path.**

- Give batch agents `FinishTool`; interactive agents pause and the host ends them with `Werk::set_task_finished`.
- Complete a schema-less batch task from a normal plain-text response, preserving its text verbatim. A host may finish such a task with any JSON value.
- Require every result `Schema` to declare a top-level object, bind its fields directly as the `finish` arguments, and keep schema-bound completion tool-driven.
- Treat `EventTool`'s `task_finished` data as the direct result object; every other published event remains observational.
- Preserve the result value unchanged through result hooks, task files, reloads, events, and `Werk::set_task_finished`.
- Run synchronous result hooks before a finish becomes observable as drained, so a hook can file follow-up work safely. Wake completion waiters after terminal transitions finish, because the event itself precedes the handlers.
- Move `Status` only through task-store transitions; reserve `Status::Failed` for system-driven terminal outcomes.

## Tools and Corrections

**Validate and dispatch every tool call through the registered `Tool`.**

- Resolve the model's exact tool name first, then its lowercase hyphen-to-underscore form with one trailing `_tool` removed.
- Reject an ambiguous folded name instead of choosing one registered tool.
- Compile input rules through `Tool::schema` and validate arguments through `Schema::validate`; do not repeat schema checks inside each tool.
- Keep corrective templates in `prompts/templates/*.md`. Resolve configured keys from Werk's shared templates, fall back to the bundled text, and fail the task through `prompt_render_failed` when a configured expression is invalid.
- Emit `tool_call_repaired` when a name or value is corrected and `tool_call_failed` when the model must recover.

## Events and Hooks

**Route every observation through `Werk::emit_event`.**

- Use `Event.name` as the semantic discriminator for built-in and application events.
- Do not make publication mutate task state; completion through `EventTool` is the explicit exception.
- Append events to `events.jsonl` before handlers run, excluding `text_chunk_received`, and fold policy statistics from the same records.
- Keep synchronous handlers cheap; async hook variants are queued and drained by the completion call.
- Build `on_result` and `on_task` on the ordered `on_event` chain so handlers coexist.
- Let an explicit shared template keyed by a non-terminal `EventTool` event name replace the text returned to the model, binding the event's JSON data before publishing the event.

## Providers and Retries

**Keep vendor protocols behind `ProviderLike` and centralize HTTP behavior in `Endpoint`.**

- Let each concrete provider own an `Endpoint`; do not add another transport abstraction.
- Keep Anthropic and OpenAI message shapes behind the internal `Protocol` trait; reuse the OpenAI shape for Mistral and LiteLLM.
- Decode vendor payloads in each provider and assemble `ModelResponse` through the shared `ResponseBuilder`.
- Apply request retries in the agent request path through `Policy::max_request_retries`, never inside a provider.

## Persistence

**Let each persisted type own its path and encoding.**

- Implement `Persist` for values saved and loaded as a whole; use inherent `append` only for append-only logs such as `Stats` and `Replies`.
- Route whole-file writes through `write_atomic` and log writes through `append_line`.
- Store task metadata, replies, results, tool outputs, events, and knowledge in separate files under the Werk directory.
- Keep automatic session writes best-effort so an I/O failure does not replace an in-memory task outcome.
- Return I/O failures from caller-driven operations such as `Knowledge::get_pages` mutations.

## Knowledge

**Keep `Knowledge` optional, durable, and shared only by explicit handle.**

- Treat the directory passed to `Knowledge(path)` as the OKF v0.1 bundle root and rebuild `index.md` from page frontmatter when opening it.
- Register `KnowledgeTool` only through an explicit `.knowledge(...)` or `.tool(...)` call; constructing an agent must not expose it.
- Inject only the index into the prompt; let `KnowledgeTool` read full pages on demand.
- Read the index once per task so writes become visible on the next task without changing an active prompt prefix.
- Cap injected index characters through `Knowledge::set_index_char_limit` without limiting stored page content.

## Policy and Run Ending

**Let `run_main_loop` announce one terminal `FinishReason` after agents stop.**

- Check `Policy` limits at turn boundaries and emit `policy_violated` before ending the run.
- Keep schema retries per task; treat `compaction_threshold` as a trigger, not a terminal limit.
- Use `FinishReason::Drained` only when no open task remains, so an interactive task can continue across replies.
- Emit `run_finished` before allowing another run to begin on the same Werk.
