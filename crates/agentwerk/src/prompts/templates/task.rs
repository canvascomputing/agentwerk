//! What task operations say when an ID or result is missing.

pub(crate) const TASK_ID_MISSING: &str = r###"No task was selected. Provide its `id` and retry."###;

pub(crate) const TASK_NOT_ASSIGNED: &str =
    r###"No task is assigned to you. Provide a task `id` and retry."###;

pub(crate) const TASK_NOT_FOUND: &str =
    r###"No task with `id` {{ id }} exists. Use `list` to see the available tasks."###;

pub(crate) const TASK_RESULT_MISSING: &str =
    r###"Task {{ id }} is {{ status }} and has no result yet. Read it again after it finishes."###;

pub(crate) const TASK_QUERY_INVALID: &str = r###"Invalid task query: {{ error }}"###;

pub(crate) const TASK_EDIT_INCOMPLETE: &str =
    r###"An edit requires `task`, `label`, or both. Provide the fields to change and retry."###;

pub(crate) const TASK_TRANSITION_REJECTED: &str = r###"Task transition rejected: {{ error }}"###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("task_id_missing", TASK_ID_MISSING),
    ("task_not_assigned", TASK_NOT_ASSIGNED),
    ("task_not_found", TASK_NOT_FOUND),
    ("task_result_missing", TASK_RESULT_MISSING),
    ("task_query_invalid", TASK_QUERY_INVALID),
    ("task_edit_incomplete", TASK_EDIT_INCOMPLETE),
    ("task_transition_rejected", TASK_TRANSITION_REJECTED),
];
