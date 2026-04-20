use serde_json::Value;

use crate::error::{AgenticError, Result};

/// Why the agent loop exited.
///
/// Distinct from [`crate::provider::types::ResponseStatus`], which describes
/// what the LLM API reported. `Status` describes the agent-level outcome.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Model chose to stop — responded without tool calls (`EndTurn`).
    Completed,
    /// External cancel signal was set.
    Cancelled,
    /// Configured `max_turns` limit reached.
    TurnLimitReached { limit: u32 },
    /// Configured `max_input_tokens` budget exceeded.
    BudgetExhausted { usage: u64, limit: u64 },
    /// A turn hook callback halted the agent.
    HaltRequested,
}

#[derive(Debug, Clone, Default)]
pub struct Statistics {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub requests: u64,
    pub tool_calls: u64,
    pub turns: u32,
}

#[derive(Debug, Clone)]
pub struct AgentOutput {
    pub response: Option<Value>,
    pub response_raw: String,
    pub statistics: Statistics,
    pub status: Status,
}

impl AgentOutput {
    pub fn empty() -> Self {
        Self {
            response: None,
            response_raw: String::new(),
            statistics: Statistics::default(),
            status: Status::Completed,
        }
    }
}

/// A validated JSON Schema for structured output.
#[derive(Debug, Clone)]
pub(crate) struct OutputSchema {
    pub schema: Value,
}

impl OutputSchema {
    pub(crate) fn new(schema: Value) -> Result<Self> {
        if schema.get("type").and_then(|t| t.as_str()) != Some("object") {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"type\": \"object\"".into(),
            });
        }
        if schema.get("properties").is_none() {
            return Err(AgenticError::SchemaValidation {
                path: String::new(),
                message: "output schema must have \"properties\"".into(),
            });
        }
        Ok(Self { schema })
    }
}

/// Parse the model's terminal text as JSON and validate it against `schema`.
/// Strips an optional outer ```json … ``` (or bare ``` … ```) fence so the
/// most common formatting mistake doesn't force a retry round-trip.
pub(crate) fn parse_and_validate(text: &str, schema: &Value) -> std::result::Result<Value, String> {
    let body = strip_code_fences(text.trim());
    let value: Value = serde_json::from_str(body).map_err(|e| format!("not valid JSON: {e}"))?;
    validate_value(&value, schema).map_err(|e| e.to_string())?;
    Ok(value)
}

fn strip_code_fences(s: &str) -> &str {
    let s = s
        .strip_prefix("```json")
        .or_else(|| s.strip_prefix("```"))
        .unwrap_or(s)
        .trim_start();
    s.strip_suffix("```").unwrap_or(s).trim()
}

/// Recursive walker: validate a JSON value against a schema. Internal to
/// this module — `parse_and_validate` is the only external entry point.
fn validate_value(value: &Value, schema: &Value) -> Result<()> {
    let schema_type = schema.get("type").and_then(|t| t.as_str()).unwrap_or("object");

    match schema_type {
        "object" => validate_object(value, schema),
        "array" => validate_array(value, schema),
        "string" if !value.is_string() => Err(type_error("expected string")),
        "number" if !value.is_number() => Err(type_error("expected number")),
        "integer" if !(value.is_i64() || value.is_u64()) => Err(type_error("expected integer")),
        "boolean" if !value.is_boolean() => Err(type_error("expected boolean")),
        _ => Ok(()),
    }
}

fn type_error(message: &str) -> AgenticError {
    AgenticError::SchemaValidation {
        path: String::new(),
        message: message.into(),
    }
}

fn prepend_path(prefix: &str, error: AgenticError) -> AgenticError {
    match error {
        AgenticError::SchemaValidation { path, message } => AgenticError::SchemaValidation {
            path: if path.is_empty() {
                prefix.to_string()
            } else {
                format!("{prefix}.{path}")
            },
            message,
        },
        other => other,
    }
}

fn validate_object(value: &Value, schema: &Value) -> Result<()> {
    let obj = value
        .as_object()
        .ok_or_else(|| type_error("expected object"))?;

    if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
        for key in required.iter().filter_map(|k| k.as_str()) {
            if !obj.contains_key(key) {
                return Err(AgenticError::SchemaValidation {
                    path: key.into(),
                    message: "missing required field".into(),
                });
            }
        }
    }

    let properties = schema.get("properties").and_then(|p| p.as_object());
    for (key, prop_schema) in properties.into_iter().flatten() {
        if let Some(prop_value) = obj.get(key) {
            validate_value(prop_value, prop_schema).map_err(|e| prepend_path(key, e))?;
        }
    }

    Ok(())
}

fn validate_array(value: &Value, schema: &Value) -> Result<()> {
    let arr = value
        .as_array()
        .ok_or_else(|| type_error("expected array"))?;

    let Some(items_schema) = schema.get("items") else {
        return Ok(());
    };

    for (i, item) in arr.iter().enumerate() {
        let index = format!("[{i}]");
        validate_value(item, items_schema).map_err(|e| prepend_path(&index, e))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "answer": { "type": "integer" } },
            "required": ["answer"]
        })
    }

    #[test]
    fn parses_pure_json() {
        let v = parse_and_validate(r#"{"answer":42}"#, &schema()).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn strips_json_code_fences() {
        let v = parse_and_validate("```json\n{\"answer\":42}\n```", &schema()).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn strips_bare_code_fences() {
        let v = parse_and_validate("```\n{\"answer\":42}\n```", &schema()).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn whitespace_around_fences_ok() {
        let v = parse_and_validate("  ```json\n  {\"answer\":42}\n```  ", &schema()).unwrap();
        assert_eq!(v["answer"], 42);
    }

    #[test]
    fn rejects_invalid_json_with_parser_message() {
        let err = parse_and_validate("the answer is 42", &schema()).unwrap_err();
        assert!(
            err.starts_with("not valid JSON"),
            "expected parse-error prefix, got: {err}"
        );
    }

    #[test]
    fn propagates_validate_value_error_with_path() {
        let err = parse_and_validate(r#"{"answer":"not a number"}"#, &schema()).unwrap_err();
        assert!(err.contains("answer"), "expected field path, got: {err}");
    }

    #[test]
    fn rejects_missing_required_field() {
        let err = parse_and_validate(r#"{}"#, &schema()).unwrap_err();
        assert!(
            err.contains("missing required field"),
            "expected required-field error, got: {err}"
        );
    }
}
