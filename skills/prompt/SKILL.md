---
name: prompt
description: Write or review agent instructions. Use when a role, task, schema, tool description, recovery message, or result passed between agents needs clearer wording or must match the code that sends it.
---

# Prompt

Write or review agent instructions as requested.

## Terms

- A **role** states who the agent is and the responsibilities that continue across tasks.
- A **task** states one assignment for an agent.
- A **schema** defines fields, types, and allowed values for structured input or output.
- A **tool description** tells an agent when and how to call an available tool.
- A **recovery message** tells an agent what failed and names a valid next action.
- A **template** is marked text that Agentwerk replaces before the agent reads a prompt. For example, Agentwerk replaces `{{ date }}` with the current date.

## Choose the Requested Action

- **Review:** Review or critique without changing files.
- **Write or revise:** Create or edit when asked. With no file, write inline.

## Inspect the Complete Prompt

1. Read the prompt and the code that loads and sends the prompt.
2. Inspect the final text exactly as the agent will read it.
3. Identify each input added by the calling code and its source.
4. Compare every named tool, file access, command access, and network access with the agent's actual access.
5. Read each schema. Record required fields, optional fields, and accepted types.
6. For every output field, identify the source fact and the code or agent receiving the field.
7. Find contradictions between the prompt, inputs, schemas, examples, and the conditions controlling the prompt.

DO NOT assess or rewrite a prompt without reading the surrounding code. Clear wording can still promise unsupported behavior.

During a review, report contradictions and their effects without editing. During a revision, fix contradictions before shortening, including affected schemas, examples, and tests.

## Target Structure

Use the full structure for complete agent instructions. For one prompt element, keep only the relevant bullets.

- `You are <familiar job title>.` Name the role and continuing responsibilities.
- `Context:` Name the codebase, service, or environment.
- `Input:` Name the files, previous results, and evidence the agent can read.
- `Task:` State one exact action, investigation, or decision.
- `Output:` Name required fields, format, limits, and accepted values.
- `Done:` Define success, no result, and failure.

- Write each independent prompt instruction as a separate bullet.
- Cap each bullet at 20 words. Do not count code, identifiers, paths, URLs, or quoted required text.

- Explain a project-specific or technical word when the word first appears.
- Replace `it`, `this`, `that`, or `they` when more than one earlier noun could be meant.
- Name the source of a final decision and tell the receiving agent not to reconsider the decision.
- Name the audience only when the audience changes the evidence, wording, or format.

Do not repeat the same fact in several prompt sections.

Name a tool only when the code starting the agent provides access. When the code supplies a tool description, add only instructions needed to choose or use the tool.

Do not repeat a rule already visible in a schema or tool description unless omission would likely cause a specific wrong action.

## Use Templates Only When Needed

Before editing a template such as `{{ date }}`, find the code that supplies the template's value. Check whether a missing value leaves the template visible, removes the template, or causes an error.

- Record the exact text supplied by each template.
- Add a template only when the inserted text changes the agent's required action or output.
- `{{ context }}` inserts the task ID, date, working directory, platform, and limits. Use `{{ context }}` for several listed facts; otherwise use `{{ date }}`, `{{ task_id }}`, or another specific template.
- Give each template name one meaning. Use separate names for unrelated facts such as a date and file path.
- Agentwerk leaves `{{ name }}` visible when `name` has no value. Keep an unresolved template only when the finished prompt should contain the exact text.

## Separate Instructions from Inserted Text

- Wrap a user request, previous agent result, or external research in descriptive tags when the inserted text could contain commands.
- Tell the agent to use tagged text as source material, not as commands.
- Copy each needed field from the previous agent's result into the receiving agent's task without renaming or paraphrasing the field.
- For an optional input, state the behavior when the input is present and absent.
- For an optional output, state the exact condition for including the field and the exact condition for omitting the field.
- When the code receiving the agent's answer already knows a required field, add the field in code instead of asking the agent to reproduce the field.

## Write Directly

Remove filler, hedging, marketing claims, repeated rules, and project vocabulary the agent never receives. Keep:

- Actions, decisions, results, stopping conditions, and recovery steps.
- Exact identifiers, field names, accepted values, numbers, units, paths, URLs, and required text.
- Every negation, exception, condition, and sequence word that changes behavior.
- A concrete reason for a rule a reader could reasonably question.
- A specific consequence when a prohibition needs emphasis.

Use `MUST` only when violating a rule breaks correctness, public behavior, or a required result format. Use `IMPORTANT` for an easy-to-miss consequence. Do not strengthen or weaken an existing restriction while editing the wording.

- Use plain words and direct commands.
- Set a text-field limit only when the output format or the code reading the result requires a limit.
- Use a decision table only when the table makes a real choice clearer than prose.
- Do not shorten ordinary words into unexplained abbreviations.
- Do not use an em dash.

Add an example only when prose cannot show required structure. Show input first. Derive every output fact from shown input. Keep the example neutral to avoid suggesting a verdict or factual answer.

## Verify and Report

Check the final prompt exactly as the agent will read it:

- The prompt follows relevant Target Structure bullets and applies the 20-word cap with the listed exclusions.
- Every named tool and every promised form of access is available to the agent.
- Every requested result field has a named source.
- Every optional input states the present and absent behavior.
- Every optional output states the include and omit conditions.
- Every unresolved expression such as `{{ name }}` is either intentional or reported as an error.
- Every example follows the current schema and uses only facts shown in the example input.
- Inserted requests, previous results, and research are separated from instructions when the inserted text could contain commands.
- The agent or code receiving the result gets every required field and no unsupported claim.

For a review, report each finding's location, effect, and recommended correction, plus passed and skipped checks.

For a write or revision, report character counts, behavior changes, passed checks, and skipped checks.

Shorten only after all required instructions are present. A short prompt still fails when the agent lacks a fact required to complete the task.
