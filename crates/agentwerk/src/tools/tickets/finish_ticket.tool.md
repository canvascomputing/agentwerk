---
name: finish_ticket
read_only: false
---

Record your final answer and mark the current ticket `Finished`. This MUST be the last action of every response that completes the work: text with no `finish_ticket` call leaves the ticket `InProgress`, and the loop re-assigns it and asks you to retry. `Finished` is terminal: it cannot be re-opened or amended.

- Operates on your current ticket; there is no `key` parameter. Call exactly once, when the work is done: never mid-task to "check in".
- `result` is a JSON value of its natural type (string, object, array, number, boolean); omit it when there is nothing to report. A stringified object (`"{\"x\": 1}"` instead of `{"x": 1}`) saves an escaped string, not the object.
- When the ticket carries a `schema`, `result` must validate against it; failures count toward `max_schema_retries`, so read the violation and fix the shape rather than retrying blindly.

## When NOT to use

- Create or edit other tickets: use `manage_tickets_tool`; this finishes the current ticket only.

## Schema

```json
{
  "type": "object",
  "properties": {
    "result": {
      "description": "The agent's final answer as a JSON value: string, object, array, number, or boolean. Pass structured payloads as JSON values, not as JSON-encoded strings. Omit this field when there is nothing to report. When the ticket has a `schema`, the value is validated against it. Examples: `\"forwarded 3 findings\"`, `{\"status\": \"ok\", \"items\": [...]}`, `[1, 2, 3]`, `42`."
    }
  }
}
```
