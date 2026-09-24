//! One line per schema violation, kept terse: a rejected value can carry twenty of them in one report. The `schema_hint_*` entries close a type violation.

pub(crate) const SCHEMA_FALSE_REJECTED: &str = r###"value is rejected by `false` schema"###;

pub(crate) const SCHEMA_TYPE_MISMATCHED: &str =
    r###"expected type {{ expected }}, got {{ got }}"###;

pub(crate) const SCHEMA_CONST_MISMATCHED: &str = r###"expected {{ expected }}"###;

pub(crate) const SCHEMA_ENUM_MISMATCHED: &str = r###"value is not in `enum`"###;

pub(crate) const SCHEMA_ANY_OF_UNMATCHED: &str =
    r###"value does not match any of the anyOf schemas"###;

pub(crate) const SCHEMA_ONE_OF_AMBIGUOUS: &str =
    r###"value matches {{ count }} of the oneOf schemas, expected exactly 1"###;

pub(crate) const SCHEMA_NOT_MATCHED: &str = r###"value must not match the `not` schema"###;

pub(crate) const SCHEMA_PROPERTY_MISSING: &str = r###"missing required property `{{ name }}`"###;

pub(crate) const SCHEMA_PROPERTY_UNEXPECTED: &str = r###"unexpected property `{{ name }}`"###;

pub(crate) const SCHEMA_ARRAY_TOO_SHORT: &str =
    r###"array has {{ count }} items, expected at least {{ min }}"###;

pub(crate) const SCHEMA_ARRAY_TOO_LONG: &str =
    r###"array has {{ count }} items, expected at most {{ max }}"###;

pub(crate) const SCHEMA_STRING_TOO_SHORT: &str =
    r###"string length {{ length }} is below minimum {{ min }}"###;

pub(crate) const SCHEMA_STRING_TOO_LONG: &str =
    r###"string length {{ length }} is above maximum {{ max }}"###;

pub(crate) const SCHEMA_PATTERN_UNMATCHED: &str =
    r###"string does not match pattern `{{ pattern }}`"###;

pub(crate) const SCHEMA_NUMBER_TOO_SMALL: &str =
    r###"value {{ value }} is below minimum {{ min }}"###;

pub(crate) const SCHEMA_NUMBER_TOO_LARGE: &str =
    r###"value {{ value }} is above maximum {{ max }}"###;

pub(crate) const SCHEMA_HINT_UNQUOTE: &str = r###"send the value unquoted"###;

pub(crate) const SCHEMA_HINT_JSON: &str = r###"send it as JSON, not as a string"###;

pub(crate) const SCHEMA_HINT_QUOTE: &str = r###"send the value quoted"###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("schema_false_rejected", SCHEMA_FALSE_REJECTED),
    ("schema_type_mismatched", SCHEMA_TYPE_MISMATCHED),
    ("schema_const_mismatched", SCHEMA_CONST_MISMATCHED),
    ("schema_enum_mismatched", SCHEMA_ENUM_MISMATCHED),
    ("schema_any_of_unmatched", SCHEMA_ANY_OF_UNMATCHED),
    ("schema_one_of_ambiguous", SCHEMA_ONE_OF_AMBIGUOUS),
    ("schema_not_matched", SCHEMA_NOT_MATCHED),
    ("schema_property_missing", SCHEMA_PROPERTY_MISSING),
    ("schema_property_unexpected", SCHEMA_PROPERTY_UNEXPECTED),
    ("schema_array_too_short", SCHEMA_ARRAY_TOO_SHORT),
    ("schema_array_too_long", SCHEMA_ARRAY_TOO_LONG),
    ("schema_string_too_short", SCHEMA_STRING_TOO_SHORT),
    ("schema_string_too_long", SCHEMA_STRING_TOO_LONG),
    ("schema_pattern_unmatched", SCHEMA_PATTERN_UNMATCHED),
    ("schema_number_too_small", SCHEMA_NUMBER_TOO_SMALL),
    ("schema_number_too_large", SCHEMA_NUMBER_TOO_LARGE),
    ("schema_hint_unquote", SCHEMA_HINT_UNQUOTE),
    ("schema_hint_json", SCHEMA_HINT_JSON),
    ("schema_hint_quote", SCHEMA_HINT_QUOTE),
];
