//! What the command tool says when a line is not one runnable program, or the rules refuse it.

pub(crate) const COMMAND_CANCELLED: &str = r###"Command cancelled: the run is ending."###;

pub(crate) const COMMAND_NOT_STARTED: &str = r###"`{{ program }}` could not be started: {{ error }}. Check the program name before calling again."###;

pub(crate) const COMMAND_MISSING: &str = r###"`command` is empty. Give the one program to run."###;

pub(crate) const COMMAND_SHELL_OPERATOR_FOUND: &str = r###"Command '{{ command }}' holds the shell operator `{{ operator }}`. This tool runs one program directly, with no shell, so make one call per command."###;

pub(crate) const COMMAND_QUOTE_UNTERMINATED: &str = r###"Command '{{ command }}' ends inside a quote or an escape. Close it before calling again."###;

pub(crate) const COMMAND_CONTROL_CHARACTER_FOUND: &str =
    r###"Command '{{ command }}' holds a control character. Remove it before calling again."###;

pub(crate) const COMMAND_ASSIGNMENT_FOUND: &str = r###"Command '{{ command }}' sets an environment variable. This tool runs one program with the environment it was started in, so drop the assignment."###;

pub(crate) const COMMAND_FLAG_DENIED: &str = r###"Command '{{ command }}' carries the denied flag '{{ flag }}'. Call it without that flag."###;

pub(crate) const COMMAND_PATTERN_DENIED: &str = r###"Command '{{ command }}' matches the denied pattern '{{ pattern }}'. Call something this tool permits."###;

pub(crate) const COMMAND_NOT_ALLOWED: &str =
    r###"Command '{{ command }}' is not allowed by tool '{{ tool }}'. {{ allowed }}"###;

pub(crate) const COMMAND_FLAG_NOT_ALLOWED: &str = r###"Command '{{ command }}' carries the flag '{{ flag }}', which tool '{{ tool }}' does not allow. Allowed flags: {{ allowed }}, and no other."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("command_cancelled", COMMAND_CANCELLED),
    ("command_not_started", COMMAND_NOT_STARTED),
    ("command_missing", COMMAND_MISSING),
    ("command_shell_operator_found", COMMAND_SHELL_OPERATOR_FOUND),
    ("command_quote_unterminated", COMMAND_QUOTE_UNTERMINATED),
    (
        "command_control_character_found",
        COMMAND_CONTROL_CHARACTER_FOUND,
    ),
    ("command_assignment_found", COMMAND_ASSIGNMENT_FOUND),
    ("command_flag_denied", COMMAND_FLAG_DENIED),
    ("command_pattern_denied", COMMAND_PATTERN_DENIED),
    ("command_not_allowed", COMMAND_NOT_ALLOWED),
    ("command_flag_not_allowed", COMMAND_FLAG_NOT_ALLOWED),
];
