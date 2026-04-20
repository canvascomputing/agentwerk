use crate::agent::compact::CompactReason;
use crate::agent::output::AgentStatus;
use crate::provider::types::TokenUsage;

#[derive(Debug, Clone)]
pub struct AgentEvent {
    pub agent_name: String,
    pub kind: AgentEventKind,
}

impl AgentEvent {
    pub(crate) fn new(agent_name: impl Into<String>, kind: AgentEventKind) -> Self {
        Self {
            agent_name: agent_name.into(),
            kind,
        }
    }
}

#[derive(Debug, Clone)]
pub enum AgentEventKind {
    AgentStart {
        description: Option<String>,
    },
    AgentEnd {
        turns: u32,
        status: AgentStatus,
    },
    TurnStart {
        turn: u32,
    },
    TurnEnd {
        turn: u32,
    },
    ToolCallStart {
        tool_name: String,
        call_id: String,
        input: serde_json::Value,
    },
    ToolCallEnd {
        tool_name: String,
        call_id: String,
        output: String,
        is_error: bool,
    },
    TokenUsage {
        model: String,
        usage: TokenUsage,
    },
    ResponseTextChunk {
        content: String,
    },
    RequestStart {
        model: String,
    },
    RequestEnd {
        model: String,
    },
    OutputTruncated {
        turn: u32,
    },
    CompactTriggered {
        turn: u32,
        token_count: u64,
        threshold: u64,
        reason: CompactReason,
    },
    AgentIdle,
    AgentResumed,
}
