//! Werk-owned prompt rendering.

use std::collections::HashMap;
use std::fmt;

use serde_json::Value;

use super::json_path::JsonPath;
use crate::{Query, Werk};

/// An expression that could not be rendered.
#[derive(Debug, Clone)]
pub(crate) struct RenderError {
    /// Expression that failed, without its surrounding braces.
    pub(crate) expression: String,
    /// Why the expression could not be resolved.
    pub(crate) message: String,
}

impl fmt::Display for RenderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("cannot render {{")?;
        f.write_str(&self.expression)?;
        f.write_str("}}: ")?;
        f.write_str(&self.message)
    }
}

impl std::error::Error for RenderError {}

impl Werk {
    /// Resolve a prompt from runtime values, shared templates, and results.
    ///
    /// Runtime values override shared templates and remain literal after insertion.
    pub(crate) fn render_prompt(
        &self,
        prompt: &str,
        values: &[(&str, String)],
    ) -> Result<String, RenderError> {
        let values: Values<'_> = values
            .iter()
            .map(|(key, value)| (*key, value.as_str()))
            .collect();
        let shared = self.template_values();
        let mut named_value = |name: &str| {
            values
                .get(name)
                .map(|value| (*value).to_string())
                .or_else(|| shared.get(name).cloned())
        };
        render_template(prompt.trim(), |expression| {
            resolve_expression(self, expression, &mut named_value)
        })
    }
}

type Values<'a> = HashMap<&'a str, &'a str>;

const EXPRESSION_OPEN: &str = "{{";
const EXPRESSION_CLOSE: &str = "}}";
const ESCAPED_EXPRESSION_OPEN: &str = "{{{{";
const ESCAPED_EXPRESSION_CLOSE: &str = "}}}}";

/// Render a template's expressions; replacement values are never scanned.
fn render_template(
    template: &str,
    mut resolve: impl FnMut(&str) -> Result<Option<String>, RenderError>,
) -> Result<String, RenderError> {
    let mut output = String::with_capacity(template.len());
    let mut remaining = template;
    while let Some(character) = remaining.chars().next() {
        if let Some(escaped) = remaining.strip_prefix(ESCAPED_EXPRESSION_OPEN) {
            let Some(end) = escaped.find(ESCAPED_EXPRESSION_CLOSE) else {
                return Err(RenderError {
                    expression: escaped.to_string(),
                    message: "unclosed escaped expression".into(),
                });
            };
            output.push_str(EXPRESSION_OPEN);
            output.push_str(&escaped[..end]);
            output.push_str(EXPRESSION_CLOSE);
            remaining = &escaped[end + ESCAPED_EXPRESSION_CLOSE.len()..];
            continue;
        }
        let Some(body) = remaining.strip_prefix(EXPRESSION_OPEN) else {
            output.push(character);
            remaining = &remaining[character.len_utf8()..];
            continue;
        };
        let end = match expression_end(body) {
            Ok(Some(end)) => end,
            Ok(None) => {
                return Err(RenderError {
                    expression: body.to_string(),
                    message: "unclosed expression or quoted value".into(),
                });
            }
            Err(message) => {
                return Err(RenderError {
                    expression: body.to_string(),
                    message: message.into(),
                });
            }
        };
        let expression = &body[..end];
        let expression_end = EXPRESSION_OPEN.len() + end + EXPRESSION_CLOSE.len();
        let literal = &remaining[..expression_end];
        let replacement = resolve(expression)?;
        output.push_str(replacement.as_deref().unwrap_or(literal));
        remaining = &remaining[expression_end..];
    }
    Ok(output)
}

/// Resolve only named values, preserving an unknown or malformed template.
pub(super) fn render_values(
    template: &str,
    mut named_value: impl FnMut(&str) -> Option<String>,
) -> String {
    render_template(template, |expression| Ok(named_value(expression.trim())))
        .unwrap_or_else(|_| template.to_string())
}

fn resolve_expression(
    werk: &Werk,
    expression: &str,
    named_value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<String>, RenderError> {
    let expression = expression.trim();
    resolve_expression_value(werk, expression, named_value).map_err(|message| RenderError {
        expression: expression.to_string(),
        message,
    })
}

fn resolve_expression_value(
    werk: &Werk,
    expression: &str,
    named_value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<String>, String> {
    if let Some(selection) = selection_expression(expression) {
        let value = resolve_selection(werk, selection, named_value)?;
        return Ok(Some(result_text(value)));
    }

    resolve_named_value(expression, named_value)
}

fn resolve_named_value(
    expression: &str,
    named_value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<String>, String> {
    let (name, json_path) = split_json_path(expression);
    let (expanded, had_nested_value) = expand_nested(name, named_value)?;
    if had_nested_value {
        return Err("nested values are only supported inside selection expressions".into());
    }
    let Some(text) = named_value(expanded.trim()) else {
        return Ok(None);
    };
    let Some(json_path) = json_path else {
        return Ok(Some(text));
    };
    let json_path = JsonPath::parse(json_path.trim()).map_err(|error| error.to_string())?;
    let value = serde_json::from_str(&text).unwrap_or(Value::Null);
    Ok(Some(result_text(json_path.evaluate(&value))))
}

fn resolve_selection(
    werk: &Werk,
    selection: SelectionExpression<'_>,
    named_value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Value, String> {
    let (query, _had_nested_value) = expand_nested(selection.query, named_value)?;
    let json_path = selection
        .json_path
        .map(JsonPath::parse)
        .transpose()
        .map_err(|error| error.to_string())?;
    let value = select_value(werk, selection.kind, query.trim())?;
    let Some(json_path) = json_path else {
        return Ok(value);
    };
    Ok(json_path.evaluate(&value))
}

fn expand_nested(
    expression: &str,
    named_value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<(String, bool), String> {
    let mut output = String::with_capacity(expression.len());
    let mut remaining = expression;
    let mut expanded = false;
    while let Some(open) = remaining.find(EXPRESSION_OPEN) {
        output.push_str(&remaining[..open]);
        let body = &remaining[open + EXPRESSION_OPEN.len()..];
        let Some(close) = body.find(EXPRESSION_CLOSE) else {
            return Err("unclosed nested expression".into());
        };
        let name = body[..close].trim();
        if name.is_empty() || name.contains(EXPRESSION_OPEN) || selection_expression(name).is_some()
        {
            return Err("nested expressions must name a template value".into());
        }
        let Some(value) = named_value(name) else {
            return Err(format!("unknown nested template value `{name}`"));
        };
        output.push_str(&value);
        remaining = &body[close + EXPRESSION_CLOSE.len()..];
        expanded = true;
    }
    output.push_str(remaining);
    Ok((output, expanded))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionKind {
    Result,
    Results,
    Task,
    Tasks,
    Event,
    Events,
}

impl SelectionKind {
    fn parse(source: &str) -> Option<Self> {
        Some(match source {
            "result" => Self::Result,
            "results" => Self::Results,
            "task" => Self::Task,
            "tasks" => Self::Tasks,
            "event" => Self::Event,
            "events" => Self::Events,
            _ => return None,
        })
    }

    fn is_plural(self) -> bool {
        matches!(self, Self::Results | Self::Tasks | Self::Events)
    }
}

#[derive(Debug, Clone, Copy)]
struct SelectionExpression<'a> {
    kind: SelectionKind,
    query: &'a str,
    json_path: Option<&'a str>,
}

fn selection_expression(expression: &str) -> Option<SelectionExpression<'_>> {
    let (kind, query) = expression.split_once(':')?;
    let kind = SelectionKind::parse(kind.trim())?;
    let (query, json_path) = split_json_path(query);
    Some(SelectionExpression {
        kind,
        query: query.trim(),
        json_path: json_path.map(str::trim),
    })
}

fn split_json_path(source: &str) -> (&str, Option<&str>) {
    let mut quote = None;
    let mut escaped = false;
    let mut parenthesis_depth = 0usize;
    let mut inside_nested_expression = false;
    let mut byte_offset = 0;
    while byte_offset < source.len() {
        let remaining = &source[byte_offset..];
        if inside_nested_expression {
            if remaining.starts_with(EXPRESSION_CLOSE) {
                inside_nested_expression = false;
                byte_offset += EXPRESSION_CLOSE.len();
                continue;
            }
            let character = remaining.chars().next().expect("remaining is nonempty");
            byte_offset += character.len_utf8();
            continue;
        }
        if remaining.starts_with(EXPRESSION_OPEN) {
            inside_nested_expression = true;
            byte_offset += EXPRESSION_OPEN.len();
            continue;
        }

        let character = remaining.chars().next().expect("remaining is nonempty");
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
            byte_offset += character.len_utf8();
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '(' => parenthesis_depth += 1,
            ')' => parenthesis_depth = parenthesis_depth.saturating_sub(1),
            '|' if parenthesis_depth == 0 && is_json_path_separator(source, byte_offset) => {
                return (
                    &source[..byte_offset],
                    Some(&source[byte_offset + character.len_utf8()..]),
                );
            }
            _ => {}
        }
        byte_offset += character.len_utf8();
    }
    (source, None)
}

fn is_json_path_separator(source: &str, byte_offset: usize) -> bool {
    let before = &source[..byte_offset];
    let after = &source[byte_offset + '|'.len_utf8()..];
    let has_space_before = before.chars().next_back().is_some_and(char::is_whitespace);
    let has_space_after = after.is_empty() || after.chars().next().is_some_and(char::is_whitespace);
    has_space_before && has_space_after
}

/// Nested placeholders may appear in quoted AQL; ordinary braces remain data.
fn expression_end(body: &str) -> Result<Option<usize>, &'static str> {
    let mut brace_depth = 0;
    let mut inside_nested_expression = false;
    let mut quote = None;
    let mut escaped = false;
    let mut byte_offset = 0;
    while byte_offset < body.len() {
        let remaining = &body[byte_offset..];
        if inside_nested_expression {
            if remaining.starts_with(EXPRESSION_OPEN) {
                return Err("nested expressions may only be one level deep");
            }
            if remaining.starts_with(EXPRESSION_CLOSE) {
                inside_nested_expression = false;
                byte_offset += EXPRESSION_CLOSE.len();
                continue;
            }
            byte_offset += remaining
                .chars()
                .next()
                .expect("remaining is nonempty")
                .len_utf8();
            continue;
        }
        if remaining.starts_with(EXPRESSION_OPEN) {
            inside_nested_expression = true;
            byte_offset += EXPRESSION_OPEN.len();
            continue;
        }

        let character = remaining.chars().next().expect("remaining is nonempty");
        if let Some(delimiter) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == delimiter {
                quote = None;
            }
            byte_offset += character.len_utf8();
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '{' => brace_depth += 1,
            '}' if brace_depth > 0 => brace_depth -= 1,
            '}' if remaining.starts_with(EXPRESSION_CLOSE) => return Ok(Some(byte_offset)),
            _ => {}
        }
        byte_offset += character.len_utf8();
    }
    if inside_nested_expression {
        Err("unclosed nested expression")
    } else {
        Ok(None)
    }
}

fn select_value(werk: &Werk, kind: SelectionKind, query: &str) -> Result<Value, String> {
    let query = Query::new(query).map_err(|error| error.to_string())?;
    match kind {
        SelectionKind::Task => serde_json::to_value(werk.find_task(query)),
        SelectionKind::Tasks => serde_json::to_value(werk.find_tasks(query)),
        SelectionKind::Event => serde_json::to_value(werk.find_event(query)),
        SelectionKind::Events => serde_json::to_value(werk.find_events(query)),
        kind => return select_result(werk, kind, query),
    }
    .map_err(|error| format!("cannot serialize selection: {error}"))
}

fn select_result(werk: &Werk, kind: SelectionKind, query: Query) -> Result<Value, String> {
    let mut tasks = werk.result_tasks(query);
    let is_plural = kind.is_plural();
    if !is_plural && tasks.is_empty() {
        return Err("no matching result".into());
    }
    if !is_plural {
        tasks.truncate(1);
    }
    let values = tasks
        .iter()
        .map(|task| {
            task.get_result()
                .cloned()
                .expect("result selector requires a result")
        })
        .collect::<Vec<_>>();
    if is_plural {
        return Ok(Value::Array(values));
    }
    Ok(values
        .into_iter()
        .next()
        .expect("singular selection is nonempty"))
}

fn result_text(value: Value) -> String {
    match value {
        Value::String(text) => text,
        value => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_text_is_trimmed() {
        let werk = Werk::new();
        assert_eq!(
            werk.render_prompt("\n\nYou review code.\n", &[]).unwrap(),
            "You review code."
        );
    }

    use crate::{Event, Task, Werk};

    fn render(werk: &Werk, prompt: impl AsRef<str>) -> Result<String, RenderError> {
        werk.render_prompt(prompt.as_ref(), &[])
    }

    fn session() -> (std::sync::Arc<Werk>, crate::test_util::TempDir) {
        let dir = crate::test_util::TempDir::new().unwrap();
        let werk = Werk::new();
        werk.set_dir(dir.path().to_path_buf()).on_event(|_, _| {});
        (werk, dir)
    }

    #[test]
    fn later_shared_values_replace_previous_bindings() {
        let (werk, _dir) = session();
        werk.set_template("company", "old");
        werk.set_template("company", "Acme");

        assert_eq!(render(&werk, "{{ company }}").unwrap(), "Acme");
    }

    #[test]
    fn shared_values_are_inserted_without_rendering_their_contents() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");
        werk.set_template("data", "{{ company }} {{ result: missing }}");
        assert_eq!(
            render(&werk, "{{ company }}: {{ data }}").unwrap(),
            "Acme: {{ company }} {{ result: missing }}"
        );
    }

    #[test]
    fn runtime_string_values_override_shared_templates_and_stay_literal() {
        let (werk, _dir) = session();
        werk.set_templates([("company", "Shared"), ("topic", "prompts")]);
        let values = [("company", "Local {{ topic }}".to_string())];

        assert_eq!(
            werk.render_prompt("{{ company }}: {{ topic }}", &values)
                .unwrap(),
            "Local {{ topic }}: prompts"
        );
    }

    #[test]
    fn value_rendering_replaces_known_names_once_and_preserves_unknown_names() {
        let values = [("name", "{{ other }}"), ("other", "expanded")];
        let rendered = render_values("{{ name }} {{name}} {{ missing }}", |name| {
            values
                .iter()
                .find_map(|(key, value)| (*key == name).then(|| (*value).to_string()))
        });

        assert_eq!(rendered, "{{ other }} {{ other }} {{ missing }}");
    }

    #[test]
    fn value_rendering_leaves_non_template_braces_literal() {
        let value = |name: &str| (name == "name").then(|| "expanded".to_string());
        let json = r#"{"one":{"two":{"three":{"value":1}}}}"#;

        assert_eq!(render_values("{name}", value), "{name}");
        assert_eq!(render_values(json, value), json);
    }

    #[test]
    fn value_rendering_unescapes_double_brace_expressions() {
        assert_eq!(render_values("{{{{ name }}}}", |_| None), "{{ name }}");
    }

    #[test]
    fn value_rendering_preserves_non_value_expressions() {
        let template = "{{ readable(result: x) }} {{ result: x }} {{ outer {{ name }} }}";

        assert_eq!(render_values(template, |_| None), template);
    }

    #[test]
    fn value_rendering_preserves_the_entire_malformed_template() {
        let value = |name: &str| (name == "name").then(|| "expanded".to_string());
        let malformed = "{{ name }} then {{ missing";

        assert_eq!(render_values(malformed, value), malformed);
    }

    #[test]
    fn prompt_rendering_leaves_non_template_braces_literal() {
        let werk = Werk::new();
        let json = r#"{"one":{"two":{"three":{"value":1}}}}"#;

        assert_eq!(render(&werk, "{company}").unwrap(), "{company}");
        assert_eq!(render(&werk, json).unwrap(), json);
        assert_eq!(
            render(&werk, "standalone }}}} braces").unwrap(),
            "standalone }}}} braces"
        );
    }

    #[test]
    fn prompt_rendering_preserves_unknown_expressions() {
        let werk = Werk::new();

        assert_eq!(render(&werk, "{{ unknown }}").unwrap(), "{{ unknown }}");
    }

    #[test]
    fn prompt_values_ignore_delimiter_whitespace() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");

        assert_eq!(
            render(&werk, "{{company}} | {{ company }} | 日本 {{ company }}").unwrap(),
            "Acme | Acme | 日本 Acme"
        );
    }

    #[test]
    fn template_variable_json_paths_select_json() {
        let (werk, _dir) = session();
        werk.set_template(
            "profile",
            r#"{"company":{"name":"Shared"},"findings":[{"summary":"one"}]}"#,
        );

        assert_eq!(
            render(&werk, "{{ profile | findings[*].summary }}").unwrap(),
            r#"["one"]"#
        );
    }

    #[test]
    fn runtime_variables_override_shared_variables_before_path_selection() {
        let (werk, _dir) = session();
        werk.set_template("profile", r#"{"company":{"name":"Shared"}}"#);
        let runtime = [("profile", r#"{"company":{"name":"Runtime"}}"#.to_string())];

        assert_eq!(
            werk.render_prompt("{{ profile | company.name }}", &runtime)
                .unwrap(),
            "Runtime"
        );
    }

    #[test]
    fn template_variables_without_paths_stay_literal() {
        let (werk, _dir) = session();
        let profile = r#"{"company":{"name":"Acme"}}"#;
        werk.set_template("profile", profile);

        assert_eq!(render(&werk, "{{ profile }}").unwrap(), profile);
    }

    #[test]
    fn unknown_template_variables_with_paths_stay_literal() {
        let werk = Werk::new();

        assert_eq!(
            render(&werk, "{{ missing | company.name }}").unwrap(),
            "{{ missing | company.name }}"
        );
    }

    #[test]
    fn malformed_template_variable_json_renders_null() {
        let (werk, _dir) = session();
        werk.set_template("profile", "not json");

        assert_eq!(
            render(&werk, "{{ profile | company.name }}").unwrap(),
            "null"
        );
    }

    #[test]
    fn only_whitespace_delimited_pipes_start_json_paths() {
        let (werk, _dir) = session();
        werk.set_templates([
            ("profile|company", "compact"),
            ("profile |company", "left only"),
            ("profile| company", "right only"),
        ]);

        for (expression, expected) in [
            ("{{ profile|company }}", "compact"),
            ("{{ profile |company }}", "left only"),
            ("{{ profile| company }}", "right only"),
        ] {
            assert_eq!(render(&werk, expression).unwrap(), expected, "{expression}");
        }
    }

    #[test]
    fn invalid_template_variable_json_paths_report_the_parse_failure() {
        let (werk, _dir) = session();
        werk.set_template("profile", r#"{"items":[]}"#);

        let error = render(&werk, "{{ profile | items[?active] }}").unwrap_err();

        assert_eq!(error.expression, "profile | items[?active]");
        assert_eq!(error.message, "invalid JSON path array index");
    }

    #[test]
    fn four_braces_emit_a_literal_double_brace_expression() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");

        assert_eq!(render(&werk, "{{{{ company }}}}").unwrap(), "{{ company }}");
    }

    #[test]
    fn result_selectors_keep_strings_plain_and_structured_values_compact() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task::labeled("research", "first"));
        let second = werk.add_task(Task::labeled("research", "second"));
        werk.set_task_finished(&first, serde_json::json!("first {{ company }}"))
            .unwrap();
        werk.set_task_finished(&second, serde_json::json!({"answer": 42}))
            .unwrap();
        assert_eq!(
            render(&werk, "{{ result: research }}").unwrap(),
            "first {{ company }}"
        );
        assert_eq!(
            render(&werk, format!("{{{{ result: {second} }}}}")).unwrap(),
            r#"{"answer":42}"#
        );
    }

    #[test]
    fn result_json_paths_render_fields_and_structured_values() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(
            &id,
            serde_json::json!({
                "company": {"name": "Acme"},
                "findings": [{"summary": "one"}],
                "}}": "closed",
            }),
        )
        .unwrap();

        assert_eq!(
            render(&werk, "{{ result: research | company.name }}").unwrap(),
            "Acme"
        );
        assert_eq!(
            render(&werk, "{{ result: research | findings[0] }}").unwrap(),
            r#"{"summary":"one"}"#
        );
        assert_eq!(
            render(&werk, "{{ result: research | company.missing }}").unwrap(),
            "null"
        );
        assert_eq!(
            render(&werk, r#"{{ result: research | "}}" }}"#).unwrap(),
            "closed"
        );
    }

    #[test]
    fn plural_result_json_paths_use_the_selected_array_as_the_root() {
        let (werk, _dir) = session();
        for verdict in ["safe", "review"] {
            let id = werk.add_task(Task::labeled("scan", "go"));
            werk.set_task_finished(&id, serde_json::json!({"verdict": verdict}))
                .unwrap();
        }

        assert_eq!(
            render(&werk, "{{ results: scan | [*].verdict }}").unwrap(),
            r#"["safe","review"]"#
        );
        assert_eq!(
            render(&werk, "{{ results: scan | [0].verdict }}").unwrap(),
            "safe"
        );
        assert_eq!(
            render(&werk, "{{ results: missing | [*].verdict }}").unwrap(),
            "[]"
        );
    }

    #[test]
    fn task_expressions_return_the_first_match_in_query_order() {
        let (werk, _dir) = session();
        werk.add_task(Task::labeled("scan", serde_json::json!({"file": "one"})));
        werk.add_task(Task::labeled("scan", serde_json::json!({"file": "two"})));

        assert_eq!(
            render(&werk, "{{ task: scan ORDER BY task.id DESC | task.file }}",).unwrap(),
            "two"
        );
    }

    #[test]
    fn tasks_expressions_use_the_selected_array_as_the_path_root() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task::labeled("scan", "one"));
        let second = werk.add_task(Task::labeled("scan", "two"));

        assert_eq!(
            render(&werk, "{{ tasks: scan | [*].id }}").unwrap(),
            serde_json::json!([first, second]).to_string()
        );
    }

    #[test]
    fn task_expressions_use_the_current_task_serde_shape() {
        let (werk, _dir) = session();
        werk.add_task(Task::labeled("scan", serde_json::json!({"file": "one"})));

        let serialized: Value =
            serde_json::from_str(&render(&werk, "{{ task: scan }}").unwrap()).unwrap();

        assert_eq!(serialized["task"]["file"], "one");
        assert!(serialized.get("result").is_none());
        assert!(serialized.get("errors").is_none());
        assert!(serialized.get("replies").is_none());
        assert!(serialized.get("cancelled").is_none());
    }

    #[test]
    fn unmatched_task_expressions_render_null_or_an_empty_array() {
        let (werk, _dir) = session();

        assert_eq!(render(&werk, "{{ task: missing }}").unwrap(), "null");
        assert_eq!(render(&werk, "{{ tasks: missing }}").unwrap(), "[]");
    }

    #[test]
    fn event_expressions_return_the_first_match_in_log_order() {
        let (werk, _dir) = session();
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "one"})));
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "two"})));

        assert_eq!(
            render(&werk, "{{ event: event.name = inspection | data.name }}",).unwrap(),
            "one"
        );
    }

    #[test]
    fn events_expressions_use_the_selected_array_as_the_path_root() {
        let (werk, _dir) = session();
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "one"})));
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "two"})));

        assert_eq!(
            render(
                &werk,
                "{{ events: event.name = inspection | [*].data.name }}",
            )
            .unwrap(),
            r#"["one","two"]"#
        );
    }

    #[test]
    fn event_expressions_use_the_current_event_serde_shape() {
        let (werk, _dir) = session();
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "one"})));

        let serialized: Value =
            serde_json::from_str(&render(&werk, "{{ event: event.name = inspection }}").unwrap())
                .unwrap();

        assert_eq!(serialized["name"], "inspection");
        assert_eq!(serialized["data"]["name"], "one");
    }

    #[test]
    fn unmatched_event_expressions_render_null_or_an_empty_array() {
        let (werk, _dir) = session();

        assert_eq!(
            render(&werk, "{{ event: event.name = absent }}").unwrap(),
            "null"
        );
        assert_eq!(
            render(&werk, "{{ events: event.name = absent }}").unwrap(),
            "[]"
        );
    }

    #[test]
    fn removed_template_expressions_stay_literal() {
        let werk = Werk::new();

        for template in [
            "{{ readable(result: research) }}",
            "{{ result_path: research }}",
            "{{ result_paths: research }}",
        ] {
            assert_eq!(render(&werk, template).unwrap(), template);
        }
    }

    #[test]
    fn quoted_aql_pipes_do_not_start_json_paths() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research | notes", "go"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();

        assert_eq!(
            render(
                &werk,
                r#"{{ result: task.label = "research | notes" | answer }}"#,
            )
            .unwrap(),
            "found"
        );
    }

    #[test]
    fn compact_aql_pipes_do_not_start_json_paths() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research|notes", "go"));
        werk.set_task_finished(&id, serde_json::json!("compact"))
            .unwrap();

        assert_eq!(
            render(&werk, "{{ result: research|notes }}").unwrap(),
            "compact"
        );
    }

    #[test]
    fn pipes_inside_query_variable_names_do_not_start_json_paths() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();
        werk.set_template("selection | literal", "research");

        assert_eq!(
            render(&werk, "{{ result: {{ selection | literal }} | answer }}",).unwrap(),
            "found"
        );
    }

    #[test]
    fn malformed_json_paths_report_the_parse_failure() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();

        for (prompt, message) in [
            ("{{ result: research | }}", "JSON path cannot be empty"),
            (
                "{{ result: research | answer || missing }}",
                "unsupported JSON path syntax at byte 6",
            ),
        ] {
            let error = render(&werk, prompt).unwrap_err();
            assert_eq!(error.expression, prompt[2..prompt.len() - 2].trim());
            assert_eq!(error.message, message, "{prompt}");
        }
    }

    #[test]
    fn template_variables_cannot_supply_json_paths() {
        let (werk, _dir) = session();
        werk.set_templates([("path", "answer"), ("profile", r#"{"answer":"found"}"#)]);

        for prompt in [
            "{{ result: research | {{ path }} }}",
            "{{ profile | {{ path }} }}",
            "{{ task: research | {{ path }} }}",
            "{{ event: event.name = task_finished | {{ path }} }}",
        ] {
            let error = render(&werk, prompt).unwrap_err();
            assert_eq!(error.expression, prompt[2..prompt.len() - 2].trim());
            assert_eq!(error.message, "unsupported JSON path syntax at byte 0");
        }
    }

    #[test]
    fn query_variables_cannot_introduce_json_paths() {
        let (werk, _dir) = session();
        werk.set_template("selection", "research | answer");

        let error = render(&werk, "{{ result: {{ selection }} }}").unwrap_err();

        assert_eq!(error.expression, "result: {{ selection }}");
        assert_eq!(error.message, "Unexpected `|` in the query.");
    }

    #[test]
    fn strings_selected_by_json_paths_are_not_rendered_again() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "{{ company }}"}))
            .unwrap();
        werk.set_template("company", "Acme");

        assert_eq!(
            render(&werk, "{{ result: research | answer }}").unwrap(),
            "{{ company }}"
        );
    }

    #[test]
    fn plural_result_selectors_follow_aql_order_and_skip_pending_tasks() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task::labeled("research", "first"));
        let second = werk.add_task(Task::labeled("research", "second"));
        werk.add_task(Task::labeled("research", "pending"));
        werk.set_task_finished(&first, serde_json::json!("first"))
            .unwrap();
        werk.set_task_finished(&second, serde_json::json!("second"))
            .unwrap();

        assert_eq!(
            render(&werk, "{{ results: research ORDER BY task.id DESC }}").unwrap(),
            r#"["second","first"]"#
        );
    }

    #[test]
    fn joined_result_selectors_emit_each_matching_task_once() {
        let (werk, _dir) = session();
        let selected = werk.add_task(Task::labeled("research", "selected"));
        werk.set_task_finished(&selected, serde_json::json!({"answer": 42}))
            .unwrap();
        werk.emit_event(Event::new("selected").task_id(&selected));
        werk.emit_event(Event::new("selected").task_id(&selected));

        assert_eq!(
            render(
                &werk,
                "{{ results: task.label = research AND event.name = selected }}",
            )
            .unwrap(),
            r#"[{"answer":42}]"#
        );
    }

    #[test]
    fn quoted_braces_inside_aql_do_not_end_the_expression() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research}notes", "go"));
        werk.set_task_finished(&id, serde_json::json!({"research": "found"}))
            .unwrap();
        assert_eq!(
            render(&werk, r#"{{ result: task.label = "research}notes" }}"#).unwrap(),
            r#"{"research":"found"}"#
        );
    }

    #[test]
    fn empty_plural_result_selectors_render_empty_arrays() {
        let (werk, _dir) = session();
        assert_eq!(render(&werk, "{{ results: missing }}").unwrap(), "[]");
    }

    #[test]
    fn missing_singular_result_selectors_are_rejected() {
        let (werk, _dir) = session();
        let error = render(&werk, "{{ result: missing }}").unwrap_err();
        assert_eq!(error.message, "no matching result");
    }

    #[test]
    fn malformed_aql_reports_the_query_failure() {
        let (werk, _dir) = session();
        for (prompt, expression, message) in [
            (
                "{{ result: }}",
                "result:",
                "A query cannot be blank. Name an origin-qualified field or a task ID.",
            ),
            (
                "{{ results: task.label = }}",
                "results: task.label =",
                "The query ends in the middle of a term.",
            ),
            (
                "{{ task: task.label = }}",
                "task: task.label =",
                "The query ends in the middle of a term.",
            ),
            (
                "{{ events: event.name = }}",
                "events: event.name =",
                "The query ends in the middle of a term.",
            ),
        ] {
            let error = render(&werk, prompt).unwrap_err();
            assert_eq!(error.expression, expression);
            assert_eq!(error.message, message);
        }
    }

    #[test]
    fn unclosed_expressions_report_the_unclosed_construct() {
        let werk = Werk::new();
        for (prompt, message) in [
            ("{{ result: research", "unclosed expression or quoted value"),
            (
                "{{ result: task.label = \"oops }}",
                "unclosed expression or quoted value",
            ),
        ] {
            let error = render(&werk, prompt).unwrap_err();
            assert_eq!(error.message, message, "{prompt}");
            assert!(error.to_string().starts_with("cannot render {{"));
        }
    }

    #[test]
    fn direct_aql_resolves_but_aql_in_template_values_stays_literal() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!("Use {{ company }}"))
            .unwrap();
        werk.set_template("research", "{{ result: research }}");
        assert_eq!(
            render(&werk, "{{ research }} | {{ result: research }}").unwrap(),
            "{{ result: research }} | Use {{ company }}"
        );
    }

    #[test]
    fn query_variables_expand_before_aql_parsing() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!("found"))
            .unwrap();
        werk.set_template("selection", "research");

        assert_eq!(
            render(&werk, "{{ result: {{ selection }} }}").unwrap(),
            "found"
        );
    }

    #[test]
    fn multiple_query_variables_expand_inside_quoted_aql_values() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task::labeled("research", "go"));
        werk.set_task_finished(&id, serde_json::json!("found"))
            .unwrap();
        werk.set_templates([("field", "task.label"), ("label", "research")]);

        assert_eq!(
            render(&werk, r#"{{ result: {{ field }} = "{{ label }}" }}"#,).unwrap(),
            "found"
        );
    }

    #[test]
    fn nested_replacements_are_not_rendered_again() {
        let (werk, _dir) = session();
        let literal = werk.add_task(Task::labeled("{{ other }}", "go"));
        werk.set_task_finished(&literal, serde_json::json!("literal"))
            .unwrap();
        werk.set_templates([
            ("literal_query", r#"task.label = "{{ other }}""#),
            ("other", "research"),
        ]);

        assert_eq!(
            render(&werk, "{{ result: {{ literal_query }} }}").unwrap(),
            "literal"
        );
    }

    #[test]
    fn invalid_nested_expressions_report_why_they_are_rejected() {
        let (werk, _dir) = session();
        werk.set_templates([("name", "research"), ("outer", "name")]);
        for (prompt, message) in [
            (
                "{{ result: {{ missing }} }}",
                "unknown nested template value `missing`",
            ),
            (
                "{{ result: {{ result: research }} }}",
                "nested expressions must name a template value",
            ),
            (
                "{{ result: {{ outer {{ name }} }} }}",
                "nested expressions may only be one level deep",
            ),
            (
                "{{ prefix {{ name }} }}",
                "nested values are only supported inside selection expressions",
            ),
        ] {
            assert_eq!(
                render(&werk, prompt).unwrap_err().message,
                message,
                "{prompt}"
            );
        }
    }

    #[test]
    fn nested_values_that_produce_invalid_aql_are_rejected() {
        let (werk, _dir) = session();
        werk.set_template("selection", "task.label =");

        let error = render(&werk, "{{ result: {{ selection }} }}").unwrap_err();

        assert_eq!(error.expression, "result: {{ selection }}");
        assert_eq!(error.message, "The query ends in the middle of a term.");
    }
}
