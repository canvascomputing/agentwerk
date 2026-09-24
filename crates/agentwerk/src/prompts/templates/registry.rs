//! What the tool registry says about a call it could not dispatch, and what stands in for a result that does not fit.

pub(crate) const TOOL_NOT_FOUND: &str = r###"No tool named `{{ name }}`. Call one of: {{ available }}. A name outside that list never resolves."###;

pub(crate) const NO_TOOLS_REGISTERED: &str =
    r###"No tool named `{{ name }}`. No tools are registered here, so no call resolves."###;

pub(crate) const TOOL_PANICKED: &str = r###"`{{ tool }}` did not finish: it panicked. Its work did not happen; call it again or take another route."###;

pub(crate) const TOOL_TIMED_OUT: &str =
    r###"Tool `{{ tool }}` timed out after {{ milliseconds }}ms."###;

pub(crate) const TOOL_OUTPUT_EMPTY: &str = r###"({{ tool }} completed with no output)"###;

pub(crate) const TOOL_OUTPUT_OFFLOADED: &str = r###"Output too large ({{ size }}). Full output saved to: {{ path }}
Preview (first {{ preview_size }}):
{{ preview }}"###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("tool_not_found", TOOL_NOT_FOUND),
    ("no_tools_registered", NO_TOOLS_REGISTERED),
    ("tool_panicked", TOOL_PANICKED),
    ("tool_timed_out", TOOL_TIMED_OUT),
    ("tool_output_empty", TOOL_OUTPUT_EMPTY),
    ("tool_output_offloaded", TOOL_OUTPUT_OFFLOADED),
];
