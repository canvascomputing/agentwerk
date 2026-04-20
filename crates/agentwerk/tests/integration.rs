//! Integration tests that hit a live LLM provider.
//! Run with `make test_integration` (requires provider env vars).

#[path = "integration/common.rs"]
mod common;

#[path = "integration/bash_usage.rs"]
mod bash_usage;
#[path = "integration/file_exploration.rs"]
mod file_exploration;
#[path = "integration/multi_agent_spawn.rs"]
mod multi_agent_spawn;
#[path = "integration/peer_messaging.rs"]
mod peer_messaging;
#[path = "integration/pool.rs"]
mod pool;
#[path = "integration/project_planning.rs"]
mod project_planning;
#[path = "integration/running_agent.rs"]
mod running_agent;
