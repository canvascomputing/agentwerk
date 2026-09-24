//! Werk-owned prompt rendering.

use std::collections::HashMap;
use std::sync::{Mutex, Weak};

use serde_json::Value;

use super::json_path::JsonPath;
use crate::{Query, Werk};

pub(crate) struct Prompt {
    werk: Weak<Werk>,
    templates: Mutex<HashMap<String, String>>,
}

impl Prompt {
    pub(crate) fn new(werk: Weak<Werk>) -> Self {
        Self {
            werk,
            templates: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn render<V: AsRef<str>>(&self, text: &str, values: &[(&str, V)]) -> String {
        let werk = self
            .werk
            .upgrade()
            .expect("Prompt cannot outlive its owning Werk");
        let templates = self.templates.lock().unwrap();
        let text = super::templates::name(text)
            .and_then(|name| templates.get(name))
            .map_or(text, String::as_str);
        render_text(&werk, text, values, &templates)
    }

    pub(crate) fn get_template(&self, key: &str) -> Option<String> {
        self.templates.lock().unwrap().get(key).cloned()
    }

    pub(crate) fn set_template(&self, key: String, value: String) {
        self.templates.lock().unwrap().insert(key, value);
    }

    pub(crate) fn inherit_templates(&self, source: &Self) {
        let templates = source.templates.lock().unwrap().clone();
        let mut current = self.templates.lock().unwrap();
        for (key, value) in templates {
            current.entry(key).or_insert(value);
        }
    }
}

fn render_text<V: AsRef<str>>(
    werk: &Werk,
    text: &str,
    runtime_values: &[(&str, V)],
    templates: &HashMap<String, String>,
) -> String {
    let mut value = |name: &str| {
        runtime_values
            .iter()
            .find_map(|(key, value)| (*key == name).then(|| value.as_ref().to_string()))
            .or_else(|| templates.get(name).cloned())
    };
    render_template(text.trim(), |expression, literal| {
        resolve_expression(werk, expression, literal, &mut value)
    })
}

const EXPRESSION_OPEN: &str = "{{";
const EXPRESSION_CLOSE: &str = "}}";
const ESCAPED_EXPRESSION_OPEN: &str = "{{{{";
const ESCAPED_EXPRESSION_CLOSE: &str = "}}}}";

/// Render a template's expressions; replacement values are never scanned.
fn render_template(template: &str, mut value: impl FnMut(&str, &str) -> String) -> String {
    let mut output = String::with_capacity(template.len());
    let mut remaining = template;
    while let Some(character) = remaining.chars().next() {
        if let Some(escaped) = remaining.strip_prefix(ESCAPED_EXPRESSION_OPEN) {
            let Some(end) = escaped.find(ESCAPED_EXPRESSION_CLOSE) else {
                output.push_str(remaining);
                break;
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
            Ok(None) | Err(_) => {
                output.push_str(remaining);
                break;
            }
        };
        let expression = &body[..end];
        let expression_end = EXPRESSION_OPEN.len() + end + EXPRESSION_CLOSE.len();
        let literal = &remaining[..expression_end];
        output.push_str(&value(expression, literal));
        remaining = &remaining[expression_end..];
    }
    output
}

/// Resolve only named values, preserving an unknown or malformed template.
pub(super) fn render_template_values(
    template: &str,
    mut value: impl FnMut(&str) -> Option<String>,
) -> String {
    render_template(template, |expression, literal| {
        value(expression.trim()).unwrap_or_else(|| literal.to_string())
    })
}

fn resolve_expression(
    werk: &Werk,
    expression: &str,
    literal: &str,
    value: &mut impl FnMut(&str) -> Option<String>,
) -> String {
    let expression = expression.trim();
    match selection_expression(expression) {
        Ok(Some(selection)) => match resolve_selection(werk, selection, value) {
            Ok(value) => value.map(result_text).unwrap_or_default(),
            Err(_) => literal.to_string(),
        },
        Err(_) => literal.to_string(),
        Ok(None) if expression.contains(EXPRESSION_OPEN) => literal.to_string(),
        Ok(None) => value(expression).unwrap_or_else(|| literal.to_string()),
    }
}

fn resolve_selection(
    werk: &Werk,
    selection: SelectionExpression<'_>,
    value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<Option<Value>, String> {
    let query = expand_nested(selection.query, value)?;
    let json_path = selection
        .json_path
        .map(JsonPath::parse)
        .transpose()
        .map_err(|error| error.to_string())?;
    let Some(value) = select_value(werk, selection.kind, query.trim())? else {
        return Ok(None);
    };
    let Some(json_path) = json_path else {
        return Ok(Some(value));
    };
    Ok(Some(json_path.evaluate(&value)))
}

fn expand_nested(
    expression: &str,
    value: &mut impl FnMut(&str) -> Option<String>,
) -> Result<String, String> {
    let mut output = String::with_capacity(expression.len());
    let mut remaining = expression;
    while let Some(open) = remaining.find(EXPRESSION_OPEN) {
        output.push_str(&remaining[..open]);
        let body = &remaining[open + EXPRESSION_OPEN.len()..];
        let Some(close) = body.find(EXPRESSION_CLOSE) else {
            return Err("unclosed nested expression".into());
        };
        let name = body[..close].trim();
        if name.is_empty()
            || name.contains(EXPRESSION_OPEN)
            || selection_expression(name)?.is_some()
        {
            return Err("nested expressions must name a template value".into());
        }
        let Some(value) = value(name) else {
            return Err(format!("unknown nested template value `{name}`"));
        };
        output.push_str(&value);
        remaining = &body[close + EXPRESSION_CLOSE.len()..];
    }
    output.push_str(remaining);
    Ok(output)
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
            "find_result" => Self::Result,
            "find_results" => Self::Results,
            "find_task" => Self::Task,
            "find_tasks" => Self::Tasks,
            "find_event" => Self::Event,
            "find_events" => Self::Events,
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

fn selection_expression(expression: &str) -> Result<Option<SelectionExpression<'_>>, String> {
    let Some((name, body)) = expression.split_once('(') else {
        return Ok(None);
    };
    let Some(kind) = SelectionKind::parse(name) else {
        return Ok(None);
    };
    let Some((query, suffix)) = split_selection_call(body) else {
        return Err("unclosed selection call".into());
    };
    let suffix = suffix.trim();
    let json_path = if suffix.is_empty() {
        None
    } else if let Some(path) = suffix.strip_prefix('.') {
        let path = path.trim();
        if path.starts_with('[') {
            return Err("expected a JSON path field after `.`".into());
        }
        Some(path)
    } else if suffix.starts_with('[') {
        Some(suffix)
    } else {
        return Err(
            "expected a JSON path beginning with `.` or `[` after the selection call".into(),
        );
    };
    Ok(Some(SelectionExpression {
        kind,
        query: query.trim(),
        json_path,
    }))
}

fn split_selection_call(source: &str) -> Option<(&str, &str)> {
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
            ')' if parenthesis_depth == 0 => {
                return Some((
                    &source[..byte_offset],
                    &source[byte_offset + character.len_utf8()..],
                ));
            }
            ')' => parenthesis_depth -= 1,
            _ => {}
        }
        byte_offset += character.len_utf8();
    }
    None
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

fn select_value(werk: &Werk, kind: SelectionKind, query: &str) -> Result<Option<Value>, String> {
    let query = Query::new(query).map_err(|error| error.to_string())?;
    match kind {
        SelectionKind::Task => werk.find_task(query).map(serde_json::to_value).transpose(),
        SelectionKind::Tasks => {
            let values = werk.find_tasks(query);
            (!values.is_empty())
                .then(|| serde_json::to_value(values))
                .transpose()
        }
        SelectionKind::Event => werk.find_event(query).map(serde_json::to_value).transpose(),
        SelectionKind::Events => {
            let values = werk.find_events(query);
            (!values.is_empty())
                .then(|| serde_json::to_value(values))
                .transpose()
        }
        kind => return select_result(werk, kind, query),
    }
    .map_err(|error| format!("cannot serialize selection: {error}"))
}

fn select_result(werk: &Werk, kind: SelectionKind, query: Query) -> Result<Option<Value>, String> {
    let mut tasks = werk.result_tasks(query);
    let is_plural = kind.is_plural();
    if tasks.is_empty() {
        return Ok(None);
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
        return Ok(Some(Value::Array(values)));
    }
    Ok(values.into_iter().next())
}

fn result_text(value: Value) -> String {
    let empty = {
        let mut pending = vec![&value];
        loop {
            match pending.pop() {
                Some(Value::Null) => {}
                Some(Value::Array(values)) => pending.extend(values),
                Some(_) => break false,
                None => break true,
            }
        }
    };
    if empty {
        return String::new();
    }
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
            werk.prompt
                .render("\n\nYou review code.\n", &[] as &[(&str, &str)]),
            "You review code."
        );
    }

    use crate::{Event, Task, Werk};

    fn render(werk: &Werk, prompt: impl AsRef<str>) -> String {
        werk.prompt.render(prompt.as_ref(), &[] as &[(&str, &str)])
    }

    fn session() -> (std::sync::Arc<Werk>, crate::test_util::TempDir) {
        let dir = crate::test_util::TempDir::new().unwrap();
        let werk = Werk(dir.path()).unwrap();
        werk.on_event(|_, _| {});
        (werk, dir)
    }

    #[test]
    fn later_shared_values_replace_previous_bindings() {
        let (werk, _dir) = session();
        werk.set_template("company", "old");
        werk.set_template("company", "Acme");

        assert_eq!(render(&werk, "{{ company }}"), "Acme");
    }

    #[test]
    fn bundled_corrective_templates_are_not_implicit_prompt_values() {
        let werk = Werk::new();

        assert_eq!(
            render(&werk, "{{ tool_timed_out }}"),
            "{{ tool_timed_out }}",
        );
    }

    #[test]
    fn bundled_templates_render_their_defaults() {
        let werk = Werk::new();

        assert_eq!(
            werk.prompt.render(
                super::super::templates::EDIT_FILE_OLD_STRING_NOT_FOUND,
                &[("path", "src/lib.rs")],
            ),
            "No `old_string` match in src/lib.rs. Read the file and copy the text exactly, including indentation, because `edit_file` matches byte for byte.",
        );
    }

    #[test]
    fn stable_names_override_bundled_templates() {
        let werk = Werk::new();
        werk.set_template(
            "edit_file_old_string_not_found",
            "Nothing matched in {{ path }}.",
        );

        assert_eq!(
            werk.prompt.render(
                super::super::templates::EDIT_FILE_OLD_STRING_NOT_FOUND,
                &[("path", "src/lib.rs")],
            ),
            "Nothing matched in src/lib.rs.",
        );
    }

    #[test]
    fn plain_text_equal_to_a_shared_value_name_stays_plain_text() {
        let werk = Werk::new();
        werk.set_template("company", "Acme");

        assert_eq!(render(&werk, "company"), "company");
    }

    #[test]
    fn templates_use_runtime_values_shared_values_and_results() {
        let (werk, _dir) = session();
        let research = werk.add_task(Task::new("research").label("research"));
        werk.set_task_finished(&research, serde_json::json!({"answer": 42}))
            .unwrap();
        werk.set_templates([
            ("company", "Acme"),
            (
                "retry",
                "{{ path }} for {{ company }}: {{ find_result(research).answer }}",
            ),
        ]);

        assert_eq!(
            werk.prompt.render(
                &werk.prompt.get_template("retry").unwrap(),
                &[("path", "src/lib.rs")],
            ),
            "src/lib.rs for Acme: 42",
        );
        werk.set_template("path", "shared/path");
        assert_eq!(
            werk.prompt.render(
                &werk.prompt.get_template("retry").unwrap(),
                &[("path", "runtime/path")],
            ),
            "runtime/path for Acme: 42",
        );
    }

    #[test]
    fn template_values_remain_single_pass() {
        let werk = Werk::new();
        werk.set_templates([
            ("retry", "{{ detail }}"),
            ("detail", "{{ find_result(missing) }}"),
        ]);

        assert_eq!(
            werk.prompt.render(
                &werk.prompt.get_template("retry").unwrap(),
                &[] as &[(&str, &str)],
            ),
            "{{ find_result(missing) }}",
        );
    }

    #[test]
    fn invalid_prompt_expressions_stay_literal() {
        let werk = Werk::new();

        for expression in [
            "{{ find_result(task.label =) }}",
            "{{ find_result({{ missing_selection }}) }}",
            "{{ find_result(absent).items[?active] }}",
        ] {
            assert_eq!(render(&werk, expression), expression);
        }
    }

    #[test]
    fn shared_values_are_inserted_without_rendering_their_contents() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");
        werk.set_template("data", "{{ company }} {{ find_result(missing) }}");
        assert_eq!(
            render(&werk, "{{ company }}: {{ data }}"),
            "Acme: {{ company }} {{ find_result(missing) }}"
        );
    }

    #[test]
    fn runtime_string_values_override_shared_templates_and_stay_literal() {
        let (werk, _dir) = session();
        werk.set_templates([("company", "Shared"), ("topic", "prompts")]);
        let values = [("company", "Local {{ topic }}".to_string())];

        assert_eq!(
            werk.prompt.render("{{ company }}: {{ topic }}", &values),
            "Local {{ topic }}: prompts"
        );
    }

    #[test]
    fn value_rendering_replaces_known_names_once_and_preserves_unknown_names() {
        let values = [("name", "{{ other }}"), ("other", "expanded")];
        let rendered = render_template_values("{{ name }} {{name}} {{ missing }}", |name| {
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

        assert_eq!(render_template_values("{name}", value), "{name}");
        assert_eq!(render_template_values(json, value), json);
    }

    #[test]
    fn value_rendering_unescapes_double_brace_expressions() {
        assert_eq!(
            render_template_values("{{{{ name }}}}", |_| None),
            "{{ name }}"
        );
    }

    #[test]
    fn value_rendering_preserves_non_value_expressions() {
        let template = "{{ readable(result: x) }} {{ find_result(x) }} {{ outer {{ name }} }}";

        assert_eq!(render_template_values(template, |_| None), template);
    }

    #[test]
    fn value_rendering_preserves_the_unclosed_remainder() {
        let value = |name: &str| (name == "name").then(|| "expanded".to_string());
        let malformed = "{{ name }} then {{ missing";

        assert_eq!(
            render_template_values(malformed, value),
            "expanded then {{ missing"
        );
    }

    #[test]
    fn prompt_rendering_leaves_non_template_braces_literal() {
        let werk = Werk::new();
        let json = r#"{"one":{"two":{"three":{"value":1}}}}"#;

        assert_eq!(render(&werk, "{company}"), "{company}");
        assert_eq!(render(&werk, json), json);
        assert_eq!(
            render(&werk, "standalone }}}} braces"),
            "standalone }}}} braces"
        );
    }

    #[test]
    fn prompt_rendering_preserves_unknown_expressions() {
        let werk = Werk::new();

        assert_eq!(render(&werk, "{{ unknown }}"), "{{ unknown }}");
    }

    #[test]
    fn prompt_values_ignore_delimiter_whitespace() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");

        assert_eq!(
            render(&werk, "{{company}} | {{ company }} | 日本 {{ company }}"),
            "Acme | Acme | 日本 Acme"
        );
    }

    #[test]
    fn removed_template_value_json_paths_stay_literal() {
        let (werk, _dir) = session();
        werk.set_template("profile", r#"{"company":{"name":"Shared"}}"#);

        assert_eq!(
            render(&werk, "{{ profile | company.name }}"),
            "{{ profile | company.name }}"
        );
    }

    #[test]
    fn template_values_remain_literal_json() {
        let (werk, _dir) = session();
        let profile = r#"{"company":{"name":"Acme"}}"#;
        werk.set_template("profile", profile);

        assert_eq!(render(&werk, "{{ profile }}"), profile);
    }

    #[test]
    fn template_value_names_may_contain_pipes() {
        let (werk, _dir) = session();
        werk.set_templates([
            ("profile|company", "compact"),
            ("profile |company", "left only"),
            ("profile| company", "right only"),
            ("profile | company", "spaced"),
        ]);

        for (expression, expected) in [
            ("{{ profile|company }}", "compact"),
            ("{{ profile |company }}", "left only"),
            ("{{ profile| company }}", "right only"),
            ("{{ profile | company }}", "spaced"),
        ] {
            assert_eq!(render(&werk, expression), expected, "{expression}");
        }
    }

    #[test]
    fn four_braces_emit_a_literal_double_brace_expression() {
        let (werk, _dir) = session();
        werk.set_template("company", "Acme");

        assert_eq!(render(&werk, "{{{{ company }}}}"), "{{ company }}");
    }

    #[test]
    fn result_selectors_keep_strings_plain_and_structured_values_compact() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task("first").label("research"));
        let second = werk.add_task(Task("second").label("research"));
        werk.set_task_finished(&first, serde_json::json!("first {{ company }}"))
            .unwrap();
        werk.set_task_finished(&second, serde_json::json!({"answer": 42}))
            .unwrap();
        assert_eq!(
            render(&werk, "{{ find_result(research) }}"),
            "first {{ company }}"
        );
        assert_eq!(
            render(&werk, format!("{{{{ find_result({second}) }}}}")),
            r#"{"answer":42}"#
        );
    }

    #[test]
    fn result_json_paths_render_fields_and_structured_values() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(
            &id,
            serde_json::json!({
                "company": {"name": "Acme"},
                "findings": [{"summary": "one"}],
                "unusual.name": "quoted",
                "}}": "closed",
            }),
        )
        .unwrap();

        assert_eq!(
            render(&werk, "{{ find_result(research).company.name }}"),
            "Acme"
        );
        assert_eq!(
            render(&werk, "{{ find_result(research).findings[0] }}"),
            r#"{"summary":"one"}"#
        );
        assert_eq!(
            render(&werk, "{{ find_result(research).company.missing }}"),
            ""
        );
        assert_eq!(
            render(&werk, r#"{{ find_result(research)."unusual.name" }}"#),
            "quoted"
        );
        assert_eq!(
            render(&werk, r#"{{ find_result(research)."}}" }}"#),
            "closed"
        );
    }

    #[test]
    fn plural_result_json_paths_use_the_selected_array_as_the_root() {
        let (werk, _dir) = session();
        for verdict in ["safe", "review"] {
            let id = werk.add_task(Task("go").label("scan"));
            werk.set_task_finished(&id, serde_json::json!({"verdict": verdict}))
                .unwrap();
        }

        assert_eq!(
            render(&werk, "{{ find_results(scan)[*].verdict }}"),
            r#"["safe","review"]"#
        );
        assert_eq!(render(&werk, "{{ find_results(scan)[0].verdict }}"), "safe");
        assert_eq!(
            render(&werk, "{{ find_results(scan)[0:1].verdict }}"),
            r#"["safe"]"#
        );
        assert_eq!(render(&werk, "{{ find_results(missing)[*].verdict }}"), "");
    }

    #[test]
    fn selector_json_paths_support_flattening_and_object_wildcards() {
        let (werk, _dir) = session();
        let object = werk.add_task(Task("object").label("object"));
        werk.set_task_finished(&object, serde_json::json!({"answer": "found"}))
            .unwrap();
        for values in [serde_json::json!([1, 2]), serde_json::json!([3])] {
            let id = werk.add_task(Task("array").label("array"));
            werk.set_task_finished(&id, values).unwrap();
        }

        assert_eq!(render(&werk, "{{ find_result(object).* }}"), r#"["found"]"#);
        assert_eq!(render(&werk, "{{ find_results(array)[] }}"), "[1,2,3]");
    }

    #[test]
    fn task_expressions_return_the_first_match_in_query_order() {
        let (werk, _dir) = session();
        werk.add_task(Task(serde_json::json!({"file": "one"})).label("scan"));
        werk.add_task(Task(serde_json::json!({"file": "two"})).label("scan"));

        assert_eq!(
            render(
                &werk,
                "{{ find_task(scan ORDER BY task.id DESC).task.file }}",
            ),
            "two"
        );
    }

    #[test]
    fn tasks_expressions_use_the_selected_array_as_the_path_root() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task("one").label("scan"));
        let second = werk.add_task(Task("two").label("scan"));

        assert_eq!(
            render(&werk, "{{ find_tasks(scan)[*].id }}"),
            serde_json::json!([first, second]).to_string()
        );
    }

    #[test]
    fn task_expressions_use_the_current_task_serde_shape() {
        let (werk, _dir) = session();
        werk.add_task(Task(serde_json::json!({"file": "one"})).label("scan"));

        let serialized: Value =
            serde_json::from_str(&render(&werk, "{{ find_task(scan) }}")).unwrap();

        assert_eq!(serialized["task"]["file"], "one");
        assert!(serialized.get("result").is_none());
        assert!(serialized.get("errors").is_none());
        assert!(serialized.get("replies").is_none());
        assert!(serialized.get("cancelled").is_none());
    }

    #[test]
    fn unmatched_task_expressions_render_nothing() {
        let (werk, _dir) = session();

        assert_eq!(render(&werk, "{{ find_task(missing) }}"), "");
        assert_eq!(render(&werk, "{{ find_tasks(missing) }}"), "");
    }

    #[test]
    fn event_expressions_return_the_first_match_in_log_order() {
        let (werk, _dir) = session();
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "one"})));
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "two"})));

        assert_eq!(
            render(&werk, "{{ find_event(event.name = inspection).data.name }}",),
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
                "{{ find_events(event.name = inspection)[*].data.name }}",
            ),
            r#"["one","two"]"#
        );
    }

    #[test]
    fn event_expressions_use_the_current_event_serde_shape() {
        let (werk, _dir) = session();
        werk.emit_event(Event::new("inspection").data(serde_json::json!({"name": "one"})));

        let serialized: Value =
            serde_json::from_str(&render(&werk, "{{ find_event(event.name = inspection) }}"))
                .unwrap();

        assert_eq!(serialized["name"], "inspection");
        assert_eq!(serialized["data"]["name"], "one");
    }

    #[test]
    fn unmatched_event_expressions_render_nothing() {
        let (werk, _dir) = session();

        assert_eq!(render(&werk, "{{ find_event(event.name = absent) }}"), "");
        assert_eq!(render(&werk, "{{ find_events(event.name = absent) }}"), "");
    }

    #[test]
    fn removed_template_expressions_stay_literal() {
        let werk = Werk::new();

        for template in [
            "{{ readable(result: research) }}",
            "{{ result_path: research }}",
            "{{ result_paths: research }}",
            "{{ result: research }}",
            "{{ results: research }}",
            "{{ task: research }}",
            "{{ tasks: research }}",
            "{{ event: research }}",
            "{{ events: research }}",
        ] {
            assert_eq!(render(&werk, template), template);
        }
    }

    #[test]
    fn quoted_aql_parentheses_do_not_end_selection_calls() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research ) notes"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();

        assert_eq!(
            render(
                &werk,
                r#"{{ find_result(task.label = "research ) notes").answer }}"#,
            ),
            "found"
        );
    }

    #[test]
    fn escaped_quotes_do_not_end_selection_calls() {
        let expression = r#"find_result(task.label = "research \" ) notes").answer"#;
        let selection = selection_expression(expression).unwrap().unwrap();

        assert_eq!(selection.kind, SelectionKind::Result);
        assert_eq!(selection.query, r#"task.label = "research \" ) notes""#);
        assert_eq!(selection.json_path, Some("answer"));
    }

    #[test]
    fn grouped_aql_parentheses_do_not_end_selection_calls() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();

        assert_eq!(
            render(
                &werk,
                "{{ find_result((task.label = research OR task.label = notes)).answer }}",
            ),
            "found"
        );
    }

    #[test]
    fn parentheses_inside_query_variable_names_do_not_end_selection_calls() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();
        werk.set_template("selection ) literal", "research");

        assert_eq!(
            render(&werk, "{{ find_result({{ selection ) literal }}).answer }}",),
            "found"
        );
    }

    #[test]
    fn malformed_json_paths_stay_literal() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "found"}))
            .unwrap();

        for prompt in [
            "{{ find_result(research). }}",
            "{{ find_result(missing). }}",
            "{{ find_result(research).answer || missing }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn malformed_selection_calls_stay_literal() {
        let werk = Werk::new();

        for prompt in [
            "{{ find_result(research }}",
            "{{ find_result(research) answer }}",
            "{{ find_result(research).[0] }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn template_variables_cannot_supply_json_paths() {
        let (werk, _dir) = session();
        werk.set_template("path", "answer");

        for prompt in [
            "{{ find_result(research).{{ path }} }}",
            "{{ find_task(research).{{ path }} }}",
            "{{ find_event(event.name = task_finished).{{ path }} }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn query_variables_cannot_introduce_json_paths() {
        let (werk, _dir) = session();
        werk.set_template("selection", "research | answer");

        let prompt = "{{ find_result({{ selection }}) }}";
        assert_eq!(render(&werk, prompt), prompt);
    }

    #[test]
    fn strings_selected_by_json_paths_are_not_rendered_again() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!({"answer": "{{ company }}"}))
            .unwrap();
        werk.set_template("company", "Acme");

        assert_eq!(
            render(&werk, "{{ find_result(research).answer }}"),
            "{{ company }}"
        );
    }

    #[test]
    fn plural_result_selectors_follow_aql_order_and_skip_pending_tasks() {
        let (werk, _dir) = session();
        let first = werk.add_task(Task("first").label("research"));
        let second = werk.add_task(Task("second").label("research"));
        werk.add_task(Task("pending").label("research"));
        werk.set_task_finished(&first, serde_json::json!("first"))
            .unwrap();
        werk.set_task_finished(&second, serde_json::json!("second"))
            .unwrap();

        assert_eq!(
            render(&werk, "{{ find_results(research ORDER BY task.id DESC) }}",),
            r#"["second","first"]"#
        );
    }

    #[test]
    fn joined_result_selectors_emit_each_matching_task_once() {
        let (werk, _dir) = session();
        let selected = werk.add_task(Task("selected").label("research"));
        werk.set_task_finished(&selected, serde_json::json!({"answer": 42}))
            .unwrap();
        werk.emit_event(Event::new("selected").task_id(&selected));
        werk.emit_event(Event::new("selected").task_id(&selected));

        assert_eq!(
            render(
                &werk,
                "{{ find_results(task.label = research AND event.name = selected) }}",
            ),
            r#"[{"answer":42}]"#
        );
    }

    #[test]
    fn quoted_braces_inside_aql_do_not_end_the_expression() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research}notes"));
        werk.set_task_finished(&id, serde_json::json!({"research": "found"}))
            .unwrap();
        assert_eq!(
            render(&werk, r#"{{ find_result(task.label = "research}notes") }}"#,),
            r#"{"research":"found"}"#
        );
    }

    #[test]
    fn unmatched_result_expressions_render_nothing() {
        let (werk, _dir) = session();

        assert_eq!(render(&werk, "{{ find_result(missing) }}"), "");
        assert_eq!(render(&werk, "{{ find_results(missing) }}"), "");
    }

    #[test]
    fn unmatched_selectors_with_json_paths_render_nothing() {
        let (werk, _dir) = session();

        for expression in [
            "{{ find_result(missing).answer }}",
            "{{ find_results(missing)[*].answer }}",
            "{{ find_task(missing).task.answer }}",
            "{{ find_tasks(missing)[*].task.answer }}",
            "{{ find_event(event.name = missing).data.answer }}",
            "{{ find_events(event.name = missing)[*].data.answer }}",
        ] {
            assert_eq!(render(&werk, expression), "", "{expression}");
        }
    }

    #[test]
    fn unmatched_selectors_disappear_without_changing_surrounding_text() {
        let (werk, _dir) = session();

        for expression in [
            "{{ find_result(missing) }}",
            "{{ find_results(missing) }}",
            "{{ find_task(missing) }}",
            "{{ find_tasks(missing) }}",
            "{{ find_event(event.name = missing) }}",
            "{{ find_events(event.name = missing) }}",
        ] {
            let prompt = format!("before {expression} after");
            assert_eq!(render(&werk, prompt), "before  after", "{expression}");
        }
    }

    #[test]
    fn result_expressions_suppress_nulls_and_empty_arrays() {
        let (werk, _dir) = session();
        let pending = werk.add_task(Task("pending").label("research"));
        let empty = werk.add_task(Task("empty").label("empty"));
        let useful = werk.add_task(Task("useful").label("useful"));

        assert_eq!(render(&werk, "{{ find_result(research) }}"), "");
        assert_eq!(render(&werk, "{{ find_results(research) }}"), "");

        werk.set_task_finished(&pending, Value::Null).unwrap();
        werk.set_task_finished(&empty, serde_json::json!([]))
            .unwrap();
        werk.set_task_finished(&useful, serde_json::json!(["instruction"]))
            .unwrap();

        assert_eq!(render(&werk, "{{ find_result(research) }}"), "");
        assert_eq!(render(&werk, "{{ find_results(research) }}"), "");
        assert_eq!(render(&werk, "{{ find_result(empty) }}"), "");
        assert_eq!(render(&werk, "{{ find_results(empty) }}"), "");
        assert_eq!(
            render(&werk, "{{ find_result(useful) }}"),
            r#"["instruction"]"#
        );
        assert_eq!(
            render(&werk, "{{ find_results(useful) }}"),
            r#"[["instruction"]]"#
        );
    }

    #[test]
    fn malformed_aql_stays_literal() {
        let (werk, _dir) = session();
        for prompt in [
            "{{ find_result() }}",
            "{{ find_results(task.label =) }}",
            "{{ find_task(task.label =) }}",
            "{{ find_events(event.name =) }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn unclosed_expressions_stay_literal() {
        let werk = Werk::new();
        for prompt in [
            "{{ find_result(research)",
            "{{ find_result(task.label = \"oops) }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn direct_aql_resolves_but_aql_in_template_values_stays_literal() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!("Use {{ company }}"))
            .unwrap();
        werk.set_template("research", "{{ find_result(research) }}");
        assert_eq!(
            render(&werk, "{{ research }} | {{ find_result(research) }}"),
            "{{ find_result(research) }} | Use {{ company }}"
        );
    }

    #[test]
    fn query_variables_expand_before_aql_parsing() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!("found"))
            .unwrap();
        werk.set_template("selection", "research");

        assert_eq!(render(&werk, "{{ find_result({{ selection }}) }}"), "found");
    }

    #[test]
    fn multiple_query_variables_expand_inside_quoted_aql_values() {
        let (werk, _dir) = session();
        let id = werk.add_task(Task("go").label("research"));
        werk.set_task_finished(&id, serde_json::json!("found"))
            .unwrap();
        werk.set_templates([("field", "task.label"), ("label", "research")]);

        assert_eq!(
            render(&werk, r#"{{ find_result({{ field }} = "{{ label }}") }}"#,),
            "found"
        );
    }

    #[test]
    fn nested_replacements_are_not_rendered_again() {
        let (werk, _dir) = session();
        let literal = werk.add_task(Task("go").label("{{ other }}"));
        werk.set_task_finished(&literal, serde_json::json!("literal"))
            .unwrap();
        werk.set_templates([
            ("literal_query", r#"task.label = "{{ other }}""#),
            ("other", "research"),
        ]);

        assert_eq!(
            render(&werk, "{{ find_result({{ literal_query }}) }}"),
            "literal"
        );
    }

    #[test]
    fn invalid_nested_expressions_stay_literal() {
        let (werk, _dir) = session();
        werk.set_templates([("name", "research"), ("outer", "name")]);
        for prompt in [
            "{{ find_result({{ missing }}) }}",
            "{{ find_result({{ find_result(research) }}) }}",
            "{{ find_result({{ outer {{ name }} }}) }}",
            "{{ prefix {{ name }} }}",
        ] {
            assert_eq!(render(&werk, prompt), prompt);
        }
    }

    #[test]
    fn nested_values_that_produce_invalid_aql_are_rejected() {
        let (werk, _dir) = session();
        werk.set_template("selection", "task.label =");

        let prompt = "{{ find_result({{ selection }}) }}";
        assert_eq!(render(&werk, prompt), prompt);
    }
}
