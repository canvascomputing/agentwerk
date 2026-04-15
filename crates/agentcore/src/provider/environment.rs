use std::sync::Arc;

use super::{AnthropicProvider, LiteLlmProvider, LlmProvider, MistralProvider, OpenAiProvider};

/// Provider and model detected from environment variables.
pub struct Environment {
    pub provider: Arc<dyn LlmProvider>,
    pub model: String,
}

impl Environment {
    /// Detect an LLM provider from environment variables.
    ///
    /// Detection order:
    ///   1. `LITELLM_API_KEY`  → LiteLLM proxy (URL from `LITELLM_API_URL`, default `http://localhost:4000`)
    ///   2. `MISTRAL_API_KEY`  → Mistral
    ///   3. `ANTHROPIC_API_KEY` → Anthropic
    ///   4. `OPENAI_API_KEY`   → OpenAI
    ///   5. localhost:4000 reachable → LiteLLM fallback (no auth)
    ///
    /// Empty env vars are treated as unset. Panics if no provider can be detected.
    pub fn detect_provider() -> Self {
        let client = reqwest::Client::new();

        if let Some(key) = env("LITELLM_API_KEY") {
            let url = env_or("LITELLM_API_URL", "http://localhost:4000");
            let model = env_or("LITELLM_MODEL", "claude-sonnet-4-20250514");
            return Self {
                provider: Arc::new(LiteLlmProvider::new(key, client).base_url(url)),
                model,
            };
        }

        if let Some(key) = env("MISTRAL_API_KEY") {
            let mut p = MistralProvider::new(key, client);
            if let Some(url) = env("MISTRAL_BASE_URL") {
                p = p.base_url(url);
            }
            let model = env_or("MISTRAL_MODEL", "mistral-medium-2508");
            return Self {
                provider: Arc::new(p),
                model,
            };
        }

        if let Some(key) = env("ANTHROPIC_API_KEY") {
            let mut p = AnthropicProvider::new(key, client);
            if let Some(url) = env("ANTHROPIC_BASE_URL") {
                p = p.base_url(url);
            }
            let model = env_or("ANTHROPIC_MODEL", "claude-sonnet-4-20250514");
            return Self {
                provider: Arc::new(p),
                model,
            };
        }

        if let Some(key) = env("OPENAI_API_KEY") {
            let mut p = OpenAiProvider::new(key, client.clone());
            if let Some(url) = env("OPENAI_BASE_URL") {
                p = p.base_url(url);
            }
            let model = env_or("OPENAI_MODEL", "gpt-4o");
            return Self {
                provider: Arc::new(p),
                model,
            };
        }

        if std::net::TcpStream::connect("127.0.0.1:4000").is_ok() {
            let model = env_or("LITELLM_MODEL", "claude-sonnet-4-20250514");
            return Self {
                provider: Arc::new(
                    LiteLlmProvider::new("", client)
                        .base_url("http://localhost:4000".into()),
                ),
                model,
            };
        }

        panic!(
            "No LLM provider found.\n\
             Set one of: LITELLM_API_KEY, MISTRAL_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY\n\
             Or start a LiteLLM proxy on localhost:4000"
        );
    }
}

/// Read an environment variable, treating empty strings as unset.
fn env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.is_empty())
}

/// Read an environment variable with a fallback default.
fn env_or(name: &str, default: &str) -> String {
    env(name).unwrap_or_else(|| default.into())
}
