//! What agentwerk tells the model between turns: a rejected reply, a summary it must write, knowledge it cannot see in full.

pub(crate) const REPLY_REJECTED: &str = r###"Your previous reply was not accepted.

{{ detail }}

Do not write any text. Your next reply must be a tool call only."###;

pub(crate) const NO_TOOL_CALLED: &str = r###"Your last reply called no tool. Call `finish` with your result when the work is complete, or another tool to continue. A reply with no tool call leaves the task unfinished."###;

pub(crate) const ARGUMENTS_REJECTED: &str = r###"`{{ tool }}` rejected your arguments. Call it again with arguments that match its schema.

{{ violations }}"###;

pub(crate) const ARGUMENTS_EXPECTED: &str = r###"The arguments `{{ tool }}` accepts:
{{ schema }}"###;

pub(crate) const RESULT_SCHEMA_REQUIRED: &str = r###"Call `finish` with a JSON object matching this schema:
{{ schema }}"###;

pub(crate) const SUMMARY_REQUESTED: &str = r###"Respond with plain text only. Do not call any tools: a tool call is rejected and wastes your only turn.

Summarize the conversation above so the agent can continue the same task without losing context. Cover every section that applies:

1. Primary request and intent
2. Key technical concepts
3. Files and code sections examined, modified, or created, with full snippets where they matter
4. Errors encountered and how they were fixed
5. Problem solving and ongoing troubleshooting
6. All non-tool-result messages from the user, verbatim where their wording matters
7. Pending tasks
8. Current work: what was being worked on immediately before this summary
9. Next step: quote directly from the most recent messages so the language stays anchored

Reply with the summary only. Do not call any tools."###;

pub(crate) const KNOWLEDGE_INDEX_TRUNCATED: &str =
    r###"{{ remaining }} more {{ pages }} not listed. Read the full index at {{ path }}."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("reply_rejected", REPLY_REJECTED),
    ("no_tool_called", NO_TOOL_CALLED),
    ("arguments_rejected", ARGUMENTS_REJECTED),
    ("arguments_expected", ARGUMENTS_EXPECTED),
    ("result_schema_required", RESULT_SCHEMA_REQUIRED),
    ("summary_requested", SUMMARY_REQUESTED),
    ("knowledge_index_truncated", KNOWLEDGE_INDEX_TRUNCATED),
];
