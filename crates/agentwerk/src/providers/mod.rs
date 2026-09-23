//! Connects agents to Anthropic, OpenAI-compatible APIs, Mistral, and LiteLLM.
//!
//! Request and response types remain available for custom [`ProviderLike`] implementations.

mod anthropic;
mod endpoint;
pub(crate) mod environment;
mod error;
mod frames;
mod litellm;
mod mistral;
pub(crate) mod model;
mod openai;
mod provider;
mod stream;
pub mod types;

pub use anthropic::Anthropic;
pub use error::{ProviderError, ProviderResult, RequestErrorKind};
pub use litellm::LiteLlm;
pub use mistral::Mistral;
pub use model::Model;
pub use openai::OpenAi;
pub use provider::{Provider, ProviderLike};
pub use types::{
    AsUserMessage, ContentBlock, Message, ModelRequest, ModelResponse, ReasoningEffort,
    ResponseStatus, StreamEvent, TokenUsage, ToolDeclineKind,
};

/// Build a model, looking its context window up by name.
#[allow(non_snake_case)]
pub fn Model(name: impl Into<String>) -> Model {
    Model::new(name)
}

/// Wrap anything that implements [`ProviderLike`].
#[allow(non_snake_case)]
pub fn Provider(provider: impl ProviderLike + 'static) -> Provider {
    Provider::new(provider)
}

/// Create an Anthropic endpoint using the API key.
#[allow(non_snake_case)]
pub fn Anthropic(api_key: impl Into<String>) -> Anthropic {
    Anthropic::new(api_key)
}

/// Create an OpenAI endpoint using the API key.
#[allow(non_snake_case)]
pub fn OpenAi(api_key: impl Into<String>) -> OpenAi {
    OpenAi::new(api_key)
}

/// Create a Mistral endpoint using the API key.
#[allow(non_snake_case)]
pub fn Mistral(api_key: impl Into<String>) -> Mistral {
    Mistral::new(api_key)
}

/// Create a LiteLLM endpoint using the API key.
#[allow(non_snake_case)]
pub fn LiteLlm(api_key: impl Into<String>) -> LiteLlm {
    LiteLlm::new(api_key)
}
