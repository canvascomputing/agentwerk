---
name: prompt
description: Write or review agent roles, tasks, tool descriptions, result schemas, and recovery messages. Use when their wording is unclear or may disagree with the code that sends them.
---

# Prompt

## Choose the Requested Action

- **Review:** Critique without changing files.
- **Write or revise:** Create or edit when asked. With no file, write inline.

## Check What the Agent Receives

1. For text sent by an application, read the code that chooses and sends it. Without a caller, flag only missing facts the agent needs to act.
2. Reconstruct the role, task, or tool description as the agent receives it. Find the source of each value added by code.
3. Check available tools, file access, commands, and network access. For each result field, find what the agent can read to fill it and who reads the answer.
4. Compare the text with its examples, schema, and sending code when available. Fix conflicts before cutting words.

Do not claim an agent has access to something you have not checked. Clear wording can still promise tools or files the agent cannot use.

During a review, report conflicts without editing. When a revision changes required fields, allowed values, or behavior, update its schema, examples, and tests.

For a tool description, explain when to call it and what it returns. If there is a schema, put field shapes there. A recovery message should name the failure and a valid next action.

## Separate the Role from the Task

Keep roles and tasks separate when the application sends them separately. Do not add a `Task:` section to a lasting role just to fill a template.

Open a role with the agent's job, the project or service it works in, where the work happens, and what it remains responsible for. Add only rules that apply across tasks. Check these facts in the code or user request.

State the current action, any target, available files or data, expected result, and when to stop. If failure changes the answer, say what to do when blocked. Keep changing locations and values in the task.

For example, a pit stop task can supply `step=loosen`, `target=front-left`, the car's state, and a map of locations. The application sends the role separately:

```text
Role:
You are a wheel-gun operator on a simulated F1 pit crew. You work in the pit box alongside other crew members. You loosen old wheels and secure replacements when assigned.

Task:
Use the supplied observation and destination map to loosen the old wheel at front-left. Finish at stage:gunner:front-left with the work complete. Leave the work position available for the next worker.
```

The labels show the difference. The pit stop application sends the task as JSON.

- Explain a project-specific word when the agent first needs it.
- Replace `it`, `this`, `that`, or `they` when more than one earlier noun could be meant.
- When one agent passes a final decision to another, name who made it and say it is final.
- Name who reads the answer only when that changes what the agent must include or how it writes.

In a role, name only tools the agent has. Do not repeat rules already in their tool descriptions or result schema unless the agent would likely make a specific mistake without them.

## Use Templates Only When Needed

When an application renders `{{ date }}` or another placeholder, find the code that supplies its value. Check what happens when the value is missing.

- Check the exact value that replaces each placeholder.
- Add a placeholder only when its value changes what the agent must do or return.
- `{{ context }}` adds the task ID, date, working directory, platform, and configured limits when present. For one value, use a specific name such as `{{ task_id }}`.
- Give each placeholder one meaning. Use different names for a date and a file path.
- Agentwerk leaves `{{ name }}` visible when `name` has no value. Keep it only when the agent should see that exact text.

## Mark Copied Requests and Results

- Mark where a quoted user request, earlier agent answer, or research begins and ends when it could contain commands. Say whether to follow its directions or use it only as evidence.
- Copy needed fields from an earlier agent's answer without changing their names or values.
- For a value that may be absent, say what the agent does when it is present and when it is missing.
- For a result field that may be omitted, say exactly when to include and omit it.
- If receiving code already knows a field, add it in code rather than asking the agent to repeat it.

## Write Directly

Write short, natural sentences. Do not join separate rules with semicolons or enforce a word cap. Cut filler, hedging, marketing claims, repetition, and project terms the agent never sees. Keep:

- What the agent must do, decide, return, check, and do when blocked.
- Exact identifiers, field names, accepted values, numbers, units, paths, URLs, and required text.
- Words such as "not", "only", "before", and "unless" when removing them would change a rule.
- A reason for a surprising rule.
- The consequence when a warning needs emphasis.

Use `MUST` only for required actions or result formats. Use `IMPORTANT` for an easy-to-miss consequence. Do not weaken or strengthen existing restrictions.

- Set a text-field limit only when the output format or the code reading the result requires a limit.
- Use a decision table only when the table makes a real choice clearer than prose.
- Do not shorten ordinary words into unexplained abbreviations.
- Do not use an em dash.

Add an example only when it makes a required structure clearer. Show the facts given to the agent before its answer. Do not put facts in the answer that the example never supplied.

## Verify and Report

Check the role, task, and tool descriptions as the agent will read them:

- A new agent can tell who it is, where it works, and what it must do now and across later tasks.
- Every named tool and promised form of access is available. The agent can find the facts needed for each result field.
- The agent knows what to do when an optional value is missing and when to include an optional result field.
- Every unresolved `{{ name }}` is intentional or reported. Examples follow the result schema and use only facts shown with them.
- Quoted requests and research have clear boundaries and a stated purpose. The recipient gets every required field without unsupported claims.
- Each sentence names an action, fact, or condition. No sentence joins separate rules with a semicolon.

For a review, report each finding's location, effect, and correction, plus passed and skipped checks.

For a write or revision, report what changed and which checks passed or were skipped. Give character counts only when a length limit matters.
