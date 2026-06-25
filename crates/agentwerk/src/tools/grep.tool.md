---
name: grep_tool
read_only: true
---

Search file contents for a literal substring (not a regex) under the working directory. Skips `.git`, `target`, `node_modules`, `vendor`. ALWAYS use this for content search; never run `grep`/`rg` via `bash_tool`.

- Scope with `glob` (`*.rs`, `*.{ts,tsx}`) to search fewer files.
- `output_mode`: `files` (default) lists matching paths; `content` gives `<path>:<line>:<col>: <line>` (context lines omit the col); `count` gives `<path>: <count>`.
- In `content` mode long lines truncate to ~200 bytes around the match; read full context with `read_file_tool` and `col_offset`/`col_limit`.
- Results cap at `max_results` (default 100); narrow `pattern` or raise the cap rather than re-running.

## When NOT to use

- Find files by name, not contents: use `glob_tool`.
- Open-ended searches needing multiple rounds: delegate to `agent_tool`.

## Schema

```json
{
  "type": "object",
  "properties": {
    "pattern": {
      "type": "string",
      "description": "Substring to search for. Literal match; not a regex."
    },
    "path": {
      "type": "string",
      "description": "Directory or file to search under (default: `.`)."
    },
    "glob": {
      "type": "string",
      "description": "File-name filter (e.g. `*.rs`, `*.{ts,tsx}`). Applied before content search."
    },
    "output_mode": {
      "type": "string",
      "description": "What to return: `files` (default) \u2014 distinct paths that contain the match; `content` \u2014 matching lines with file path, line number, and surrounding context; `count` \u2014 match count per file."
    },
    "context_lines": {
      "type": "integer",
      "description": "Lines of context before and after each match (default: 0). Only used when `output_mode` is `content`."
    },
    "case_insensitive": {
      "type": "boolean",
      "description": "Match without regard to case (default: false)."
    },
    "max_results": {
      "type": "integer",
      "description": "Maximum number of results to return (default: 100). Applies to whichever unit `output_mode` produces."
    }
  },
  "required": [
    "pattern"
  ]
}
```
