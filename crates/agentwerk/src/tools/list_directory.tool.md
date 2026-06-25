---
name: list_directory_tool
read_only: true
---

List a directory's entries to survey an unfamiliar layout. Output is one entry per line, sorted alphabetically: `<name>  file  <size_bytes>` or `<name>  dir`; in recursive mode `<name>` is relative to `path`. The path resolves against the working directory.

## When NOT to use

- Find files by pattern across the tree: use `glob_tool`.
- Search file contents: use `grep_tool`.
- Read one file: use `read_file_tool`.

## Schema

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Directory to list (default: `.`)."
    },
    "recursive": {
      "type": "boolean",
      "description": "Walk subdirectories and list every entry beneath `path` (default: false). Use sparingly \u2014 prefer `glob_tool` for large trees."
    }
  }
}
```
