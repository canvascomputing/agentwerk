//! What the file tools say when a path, a match, or a write does not hold. The `path_hint_*` entries close a not-found message.

pub(crate) const EDIT_FILE_READ_FAILED: &str = r###"{{ path }} could not be read: {{ error }}. Check the path with `list_directory` before editing it."###;

pub(crate) const EDIT_FILE_OLD_STRING_NOT_FOUND: &str = r###"No `old_string` match in {{ path }}. Read the file and copy the text exactly, including indentation, because `edit_file` matches byte for byte."###;

pub(crate) const EDIT_FILE_OLD_STRING_NOT_UNIQUE: &str = r###"`old_string` matches {{ count }} places in {{ path }}. Extend it with the surrounding lines until it is unique, or set `replace_all` to true to change every one."###;

pub(crate) const EDIT_FILE_WRITE_FAILED: &str =
    r###"{{ path }} could not be written: {{ error }}. The file is unchanged."###;

pub(crate) const WRITE_FILE_PARENT_NOT_CREATED: &str = r###"The parent directories of {{ path }} could not be created: {{ error }}. Nothing was written."###;

pub(crate) const WRITE_FILE_FAILED: &str =
    r###"{{ path }} could not be written: {{ error }}. Nothing was written."###;

pub(crate) const READ_FILE_PATH_IS_DIRECTORY: &str =
    r###"'{{ path }}' is a directory, not a file."###;

pub(crate) const READ_FILE_PATH_IS_DIRECTORY_WITH_ENTRIES: &str = r###"'{{ path }}' is a directory, not a file. Read one of its entries by appending the name to the path:
  {{ entries }}"###;

pub(crate) const READ_FILE_IS_BINARY: &str = r###"{{ path }} is a binary file ({{ bytes }} bytes), not text; it cannot be read as source. Judge from the information you already have."###;

pub(crate) const READ_FILE_NOT_FOUND: &str = r###"File does not exist: {{ path }}. {{ hint }}"###;

pub(crate) const READ_FILE_FAILED: &str = r###"{{ path }} could not be read: {{ error }}. Check the path with `list_directory` before retrying."###;

pub(crate) const LIST_DIRECTORY_PATH_IS_FILE: &str =
    r###"Path is not a directory: {{ path }}. Read it with `read_file` instead."###;

pub(crate) const LIST_DIRECTORY_NOT_FOUND: &str =
    r###"Directory does not exist: {{ path }}. {{ hint }}"###;

pub(crate) const LIST_DIRECTORY_FAILED: &str =
    r###"{{ path }} could not be listed: {{ error }}."###;

pub(crate) const PATH_HINT_DIRECTORY_LISTED: &str = r###"'{{ dir }}' contains:
  {{ entries }}"###;

pub(crate) const PATH_HINT_SUGGESTION: &str =
    r###"Note: your current working directory is {{ dir }}. Did you mean {{ suggestion }}?"###;

pub(crate) const PATH_HINT_WORKING_DIRECTORY: &str =
    r###"Note: your current working directory is {{ dir }}."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("edit_file_read_failed", EDIT_FILE_READ_FAILED),
    (
        "edit_file_old_string_not_found",
        EDIT_FILE_OLD_STRING_NOT_FOUND,
    ),
    (
        "edit_file_old_string_not_unique",
        EDIT_FILE_OLD_STRING_NOT_UNIQUE,
    ),
    ("edit_file_write_failed", EDIT_FILE_WRITE_FAILED),
    (
        "write_file_parent_not_created",
        WRITE_FILE_PARENT_NOT_CREATED,
    ),
    ("write_file_failed", WRITE_FILE_FAILED),
    ("read_file_path_is_directory", READ_FILE_PATH_IS_DIRECTORY),
    (
        "read_file_path_is_directory_with_entries",
        READ_FILE_PATH_IS_DIRECTORY_WITH_ENTRIES,
    ),
    ("read_file_is_binary", READ_FILE_IS_BINARY),
    ("read_file_not_found", READ_FILE_NOT_FOUND),
    ("read_file_failed", READ_FILE_FAILED),
    ("list_directory_path_is_file", LIST_DIRECTORY_PATH_IS_FILE),
    ("list_directory_not_found", LIST_DIRECTORY_NOT_FOUND),
    ("list_directory_failed", LIST_DIRECTORY_FAILED),
    ("path_hint_directory_listed", PATH_HINT_DIRECTORY_LISTED),
    ("path_hint_suggestion", PATH_HINT_SUGGESTION),
    ("path_hint_working_directory", PATH_HINT_WORKING_DIRECTORY),
];
