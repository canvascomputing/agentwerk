mod command;
mod fetch;
mod files;
mod knowledge;
mod r#loop;
mod registry;
mod schemas;
mod search;
mod task;

pub(crate) use command::{
    COMMAND_ASSIGNMENT_FOUND, COMMAND_CANCELLED, COMMAND_CONTROL_CHARACTER_FOUND,
    COMMAND_FLAG_DENIED, COMMAND_FLAG_NOT_ALLOWED, COMMAND_MISSING, COMMAND_NOT_ALLOWED,
    COMMAND_NOT_STARTED, COMMAND_PATTERN_DENIED, COMMAND_QUOTE_UNTERMINATED,
    COMMAND_SHELL_OPERATOR_FOUND,
};
pub(crate) use fetch::{
    FETCH_BODY_NOT_READ, FETCH_CREDENTIALS_PRESENT, FETCH_HOST_MISSING, FETCH_HOST_NOT_RESOLVABLE,
    FETCH_REDIRECT_LOCATION_MISSING, FETCH_REQUEST_FAILED, FETCH_RESPONSE_TOO_LARGE,
    FETCH_SCHEME_MISSING, FETCH_SCHEME_UNSUPPORTED, FETCH_TOO_LONG, FETCH_TOO_MANY_REDIRECTS,
};
pub(crate) use files::{
    EDIT_FILE_OLD_STRING_NOT_FOUND, EDIT_FILE_OLD_STRING_NOT_UNIQUE, EDIT_FILE_READ_FAILED,
    EDIT_FILE_WRITE_FAILED, LIST_DIRECTORY_FAILED, LIST_DIRECTORY_NOT_FOUND,
    LIST_DIRECTORY_PATH_IS_FILE, PATH_HINT_DIRECTORY_LISTED, PATH_HINT_SUGGESTION,
    PATH_HINT_WORKING_DIRECTORY, READ_FILE_FAILED, READ_FILE_IS_BINARY, READ_FILE_NOT_FOUND,
    READ_FILE_PATH_IS_DIRECTORY, READ_FILE_PATH_IS_DIRECTORY_WITH_ENTRIES, WRITE_FILE_FAILED,
    WRITE_FILE_PARENT_NOT_CREATED,
};
pub(crate) use knowledge::{
    KNOWLEDGE_PAGE_NOT_FOUND, KNOWLEDGE_REMOVE_FAILED, KNOWLEDGE_WRITE_FAILED,
};
pub(crate) use r#loop::{
    ARGUMENTS_EXPECTED, ARGUMENTS_REJECTED, KNOWLEDGE_INDEX_TRUNCATED, NO_TOOL_CALLED,
    REPLY_REJECTED, RESULT_SCHEMA_REQUIRED, SUMMARY_REQUESTED,
};
pub(crate) use registry::{
    NO_TOOLS_REGISTERED, TOOL_NOT_FOUND, TOOL_OUTPUT_EMPTY, TOOL_OUTPUT_OFFLOADED, TOOL_PANICKED,
    TOOL_TIMED_OUT,
};
pub(crate) use schemas::{
    SCHEMA_ANY_OF_UNMATCHED, SCHEMA_ARRAY_TOO_LONG, SCHEMA_ARRAY_TOO_SHORT,
    SCHEMA_CONST_MISMATCHED, SCHEMA_ENUM_MISMATCHED, SCHEMA_FALSE_REJECTED, SCHEMA_HINT_JSON,
    SCHEMA_HINT_QUOTE, SCHEMA_HINT_UNQUOTE, SCHEMA_NOT_MATCHED, SCHEMA_NUMBER_TOO_LARGE,
    SCHEMA_NUMBER_TOO_SMALL, SCHEMA_ONE_OF_AMBIGUOUS, SCHEMA_PATTERN_UNMATCHED,
    SCHEMA_PROPERTY_MISSING, SCHEMA_PROPERTY_UNEXPECTED, SCHEMA_STRING_TOO_LONG,
    SCHEMA_STRING_TOO_SHORT, SCHEMA_TYPE_MISMATCHED,
};
pub(crate) use search::{
    CODE_CONSTRAINT_INCOMPLETE, CODE_CONSTRAINT_METAVARIABLE_UNKNOWN,
    CODE_CONSTRAINT_REGEX_REJECTED, CODE_PATTERN_REJECTED, GREP_CANCELLED, GREP_FAILED,
    GREP_FILE_TYPE_UNKNOWN, GREP_GLOB_REJECTED, GREP_PATTERN_REJECTED,
};
pub(crate) use task::{
    TASK_EDIT_INCOMPLETE, TASK_ID_MISSING, TASK_NOT_ASSIGNED, TASK_NOT_FOUND, TASK_QUERY_INVALID,
    TASK_RESULT_MISSING, TASK_TRANSITION_REJECTED,
};

use super::prompt::render_template_values;

const GROUPS: &[&[(&str, &str)]] = &[
    command::TEMPLATES,
    fetch::TEMPLATES,
    files::TEMPLATES,
    knowledge::TEMPLATES,
    r#loop::TEMPLATES,
    registry::TEMPLATES,
    schemas::TEMPLATES,
    search::TEMPLATES,
    task::TEMPLATES,
];

/// Render a bundled template where no Werk is available.
pub(crate) fn built_in(template: &str, values: &[(&str, &str)]) -> String {
    render_template_values(template, |name| {
        values
            .iter()
            .find_map(|(key, value)| (*key == name).then(|| (*value).to_string()))
    })
}

/// Return the stable configuration name for a bundled template body.
pub(super) fn name(template: &str) -> Option<&'static str> {
    GROUPS
        .iter()
        .flat_map(|group| group.iter())
        .find_map(|(name, default)| (*default == template).then_some(*name))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn bundled_templates_have_unique_names_and_defaults() {
        let mut names = HashSet::new();
        let mut defaults = HashSet::new();
        for (name, default) in GROUPS.iter().flat_map(|group| group.iter()) {
            assert!(names.insert(*name), "duplicate template name: {name}");
            assert!(!default.is_empty(), "empty template: {name}");
            assert!(defaults.insert(*default), "duplicate template body: {name}");
            assert_eq!(super::name(default), Some(*name));
        }
    }

    #[test]
    fn built_in_templates_render_runtime_values() {
        let rendered = built_in(EDIT_FILE_OLD_STRING_NOT_FOUND, &[("path", "src/lib.rs")]);
        assert!(rendered.contains("src/lib.rs"));
        assert!(!rendered.contains("{{ path }}"));
    }
}
