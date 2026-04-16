use std::future::Future;
use std::pin::Pin;

use crate::error::Result;

use super::output::AgentOutput;
use super::context::RuntimeContext;

/// The agent interface. Obtain instances via [`AgentBuilder::build()`].
///
/// This trait is **sealed** — it cannot be implemented outside this crate
/// because [`RuntimeContext`] is crate-private.
pub trait Agent: Send + Sync {
    fn name(&self) -> &str;
    fn run(
        &self,
        ctx: RuntimeContext,
    ) -> Pin<Box<dyn Future<Output = Result<AgentOutput>> + Send + '_>>;
}
