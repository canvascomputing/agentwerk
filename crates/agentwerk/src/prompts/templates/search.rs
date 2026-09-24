//! What `grep` and the code search say about a pattern, a glob, or a search that did not finish.

pub(crate) const GREP_CANCELLED: &str = r###"Search cancelled: the run is ending."###;

pub(crate) const GREP_FAILED: &str = r###"The search did not run. Narrow `path` and call `grep` once more; an identical retry fails the same way."###;

pub(crate) const GREP_GLOB_REJECTED: &str =
    r###"`glob` is not a valid glob: {{ error }}. Write it as a path pattern such as `**/*.rs`."###;

pub(crate) const GREP_FILE_TYPE_UNKNOWN: &str = r###"No file type named `{{ file_type }}`: {{ error }}. Drop `file_type` and narrow with `glob` instead."###;

pub(crate) const GREP_PATTERN_REJECTED: &str = r###"Search failed: {{ error }}. `pattern` is a regular expression. To find a call or code shape, use `syntax: "code"` (`Name(...)`); otherwise escape the metacharacters."###;

pub(crate) const CODE_PATTERN_REJECTED: &str = r###"`pattern` is not a valid code pattern: {{ error }}. Write it as source, with `$NAME` where the code varies."###;

pub(crate) const CODE_CONSTRAINT_INCOMPLETE: &str = r###"Each entry in `constraints` needs a `metavariable` and a `regex`. Give both, or drop the entry."###;

pub(crate) const CODE_CONSTRAINT_METAVARIABLE_UNKNOWN: &str = r###"`constraints` names ${name}, which `pattern` does not declare. Constrain a metavariable the pattern writes."###;

pub(crate) const CODE_CONSTRAINT_REGEX_REJECTED: &str = r###"The `regex` for ${name} is not a valid regular expression: {{ error }}. Escape the metacharacters and call again."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("grep_cancelled", GREP_CANCELLED),
    ("grep_failed", GREP_FAILED),
    ("grep_glob_rejected", GREP_GLOB_REJECTED),
    ("grep_file_type_unknown", GREP_FILE_TYPE_UNKNOWN),
    ("grep_pattern_rejected", GREP_PATTERN_REJECTED),
    ("code_pattern_rejected", CODE_PATTERN_REJECTED),
    ("code_constraint_incomplete", CODE_CONSTRAINT_INCOMPLETE),
    (
        "code_constraint_metavariable_unknown",
        CODE_CONSTRAINT_METAVARIABLE_UNKNOWN,
    ),
    (
        "code_constraint_regex_rejected",
        CODE_CONSTRAINT_REGEX_REJECTED,
    ),
];
