//! Mock-driven tests that never hit a live LLM provider.
//! Run with `make test`.

#[path = "unit/context_window_events.rs"]
mod context_window_events;

#[path = "unit/structured_output.rs"]
mod structured_output;
