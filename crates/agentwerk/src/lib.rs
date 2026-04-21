pub mod agent;
pub mod error;
pub(crate) mod persistence;
pub mod provider;
pub mod tools;
pub(crate) mod util;

pub mod testutil;

// Errors
pub use error::{AgenticError, Result};

// Provider and message types
pub use provider::{
    AnthropicProvider, CompletionRequest, ContentBlock, LiteLlmProvider, Message, MistralProvider,
    Model, ModelLookup, OpenAiProvider, Provider, ProviderError, TokenUsage,
};

// Tool infrastructure and built-in tools
pub use tools::{
    BashTool, EditFileTool, GlobTool, GrepTool, ListDirectoryTool, ReadFileTool, SendMessageTool,
    SpawnAgentTool, TaskTool, Tool, ToolContext, ToolResult, ToolSearchTool, Toolable,
    WebFetchTool, WriteFileTool,
};

// Agent
pub use agent::{
    Agent, AgentEvent, AgentEventKind, AgentHandle, AgentJobId, AgentOutput, AgentOutputFuture,
    AgentPool, AgentPoolStrategy, AgentStatistics, AgentStatus, CompactReason,
    DEFAULT_BEHAVIOR_PROMPT,
};
