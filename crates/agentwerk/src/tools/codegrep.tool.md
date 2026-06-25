---
name: codegrep_tool
read_only: true
---

Search file contents by structural code pattern: metavariables (`$NAME`), balanced brackets, and ellipses (`...`, `....`). Example: `fn $NAME(...)` matches any function and captures the name. Each match prints `<path>:<line>:<col>: <matched_text>` plus `[$NAME=value]` for captures (newlines shown as `\n`, long matches truncated); skips `.git`, `target`, `node_modules`, `vendor`.

- Use when the match must ignore identifier names, whitespace, or argument shape; use `grep_tool` for a byte-for-byte literal. Scope with `glob` (`*.rs`, `*.ts`) and pair with `read_file_tool` to inspect a hit.
- Patterns are structural, NOT regex: anchors (`^`/`$`), escapes (`\(`), and classes (`\d`, `[a-z]`, `?`, `*`) are not syntax. Write the literal: `eval(`, not `eval\(`.
- `$NAME` captures one token, `$...NAME` a span; reusing a name back-references it exactly. `...` spans tokens at one bracket level and stops at newlines; `....` crosses them.

## When NOT to use

- Match a literal substring: use `grep_tool`.
- Find files by name: use `glob_tool`.
- Regex features (`\d`, `[a-z]`, `?`, `*`): not supported; an empty `pattern` is an error, not match-all.

## Schema

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Codegrep pattern: literal text, balanced brackets, metavariables (`$NAME` for one word, `$...NAME` for a span), and ellipses (`...` for a token sequence, `....` for one that crosses newlines). Must be non-empty."
    },
    "path": {
      "type": "string",
      "description": "Directory or file to search under (default: `.`)."
    },
    "glob": {
      "type": "string",
      "description": "File-name filter (e.g. `*.rs`, `*.{ts,tsx}`). Applied before content search."
    },
    "mode": {
      "type": "string",
      "description": "`multiline` (default) treats newlines as whitespace, like a formatter would. `singleline` makes `...` stop at newlines so a pattern stays within one line; use `....` to span lines."
    },
    "caseless": {
      "type": "boolean",
      "description": "Match identifiers without regard to letter case (default: false)."
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of matches to return (default: 100)."
    }
  },
  "required": [
    "pattern"
  ]
}
```
