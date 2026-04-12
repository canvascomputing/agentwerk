use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::Result;

use super::types::{Message, ModelResponse, StreamEvent};
use crate::tools::tool::ToolDefinition;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: u32,
    pub tool_choice: Option<ToolChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Specific { name: String },
}

/// Core LLM provider trait. Object-safe via boxed futures.
pub trait LlmProvider: Send + Sync {
    fn complete(
        &self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>>;

    /// Streaming variant that emits incremental events via callback.
    /// Default implementation falls back to `complete()` and emits `MessageDone`.
    fn complete_streaming(
        &self,
        request: CompletionRequest,
        on_event: Arc<dyn Fn(StreamEvent) + Send + Sync>,
    ) -> Pin<Box<dyn Future<Output = Result<ModelResponse>> + Send + '_>> {
        Box::pin(async move {
            let response = self.complete(request).await?;
            on_event(StreamEvent::MessageDone);
            Ok(response)
        })
    }
}

/// Fire-and-forget HEAD request to warm the TCP+TLS connection pool.
/// Call via `tokio::spawn` at startup before the first real API request.
pub async fn prewarm_connection(client: &reqwest::Client, base_url: &str) {
    let _ = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        client.head(base_url).send(),
    )
    .await;
}
