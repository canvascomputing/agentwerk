//! The agent's eyes on the filesystem. Lets a model read a file it did not receive in the prompt.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

use serde_json::Value;

use crate::providers::ProviderResult;

use super::tool::{ToolContext, ToolLike, ToolResult};
use super::tool_file::ToolFile;

/// Read a file with optional line offset and limit. Returns line-numbered
/// text so the model can reference specific lines in subsequent edits.
/// Read-only.
///
/// # Examples
///
/// ```
/// use agentwerk::Agent;
/// use agentwerk::tools::ReadFileTool;
///
/// Agent::new().tool(ReadFileTool);
/// ```
pub struct ReadFileTool;

fn tool_file() -> &'static ToolFile {
    static FILE: OnceLock<ToolFile> = OnceLock::new();
    FILE.get_or_init(|| ToolFile::parse(include_str!("read_file.tool.md")))
}

fn description() -> &'static str {
    static DESC: OnceLock<String> = OnceLock::new();
    DESC.get_or_init(|| tool_file().render_markdown())
}

impl ToolLike for ReadFileTool {
    fn name(&self) -> &str {
        &tool_file().name
    }

    fn description(&self) -> &str {
        description()
    }

    fn input_schema(&self) -> Value {
        tool_file().input_schema.clone()
    }

    fn is_read_only(&self) -> bool {
        tool_file().read_only
    }

    fn call<'a>(
        &'a self,
        input: Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let path = match input["path"].as_str() {
                Some(p) => p,
                None => {
                    return Ok(ToolResult::error("Missing required parameter: path"));
                }
            };

            let resolved = ctx.dir.join(path);

            let content = match std::fs::read_to_string(&resolved) {
                Ok(c) => c,
                Err(e) => {
                    return Ok(ToolResult::error(format!("Failed to read file: {e}")));
                }
            };

            let lines: Vec<&str> = content.lines().collect();

            let offset = input["offset"].as_u64().unwrap_or(1).max(1) as usize;
            let limit = input["limit"]
                .as_u64()
                .map(|l| l as usize)
                .unwrap_or(lines.len().saturating_sub(offset - 1));

            let col_offset = input["col_offset"].as_u64().map(|c| c.max(1) as usize);
            let col_limit = input["col_limit"].as_u64().map(|c| c as usize);

            let start = (offset - 1).min(lines.len());
            let end = (start + limit).min(lines.len());

            let mut result = String::new();
            for (i, line) in lines[start..end].iter().enumerate() {
                let line_num = start + i + 1;
                if !result.is_empty() {
                    result.push('\n');
                }
                match col_offset {
                    Some(col) => {
                        let byte_start = snap_to_char_boundary(line, (col - 1).min(line.len()));
                        let byte_end = match col_limit {
                            Some(cl) => {
                                snap_to_char_boundary(line, (byte_start + cl).min(line.len()))
                            }
                            None => line.len(),
                        };
                        let slice = &line[byte_start..byte_end];
                        let display_col = byte_start + 1;
                        result.push_str(&format!("{line_num}:{display_col}\t{slice}"));
                    }
                    None => {
                        result.push_str(&format!("{line_num}\t{line}"));
                    }
                }
            }

            Ok(ToolResult::success(result))
        })
    }
}

fn snap_to_char_boundary(s: &str, pos: usize) -> usize {
    let pos = pos.min(s.len());
    let mut p = pos;
    while p < s.len() && !s.is_char_boundary(p) {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn test_ctx(dir: &std::path::Path) -> ToolContext {
        ToolContext::new(PathBuf::from(dir))
    }

    #[tokio::test]
    async fn read_file_cases() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\ndelta\n").unwrap();

        struct Case {
            name: &'static str,
            input: Value,
            expect_error: bool,
            expect_contains: &'static str,
        }

        let cases = vec![
            Case {
                name: "full file",
                input: serde_json::json!({ "path": "test.txt" }),
                expect_error: false,
                expect_contains: "1\talpha\n2\tbeta\n3\tgamma\n4\tdelta",
            },
            Case {
                name: "offset and limit",
                input: serde_json::json!({ "path": "test.txt", "offset": 2, "limit": 2 }),
                expect_error: false,
                expect_contains: "2\tbeta\n3\tgamma",
            },
            Case {
                name: "nonexistent file",
                input: serde_json::json!({ "path": "no_such_file.txt" }),
                expect_error: true,
                expect_contains: "Failed to read file",
            },
            Case {
                name: "col_offset slices from byte position",
                input: serde_json::json!({ "path": "test.txt", "offset": 1, "limit": 1, "col_offset": 3 }),
                expect_error: false,
                expect_contains: "1:3\tpha",
            },
            Case {
                name: "col_offset with col_limit bounds the slice",
                input: serde_json::json!({ "path": "test.txt", "offset": 2, "limit": 1, "col_offset": 2, "col_limit": 3 }),
                expect_error: false,
                expect_contains: "2:2\teta",
            },
            Case {
                name: "col_offset beyond line returns empty",
                input: serde_json::json!({ "path": "test.txt", "offset": 1, "limit": 1, "col_offset": 100 }),
                expect_error: false,
                expect_contains: "1:6\t",
            },
            Case {
                name: "col_limit past end of line clamps to EOL",
                input: serde_json::json!({ "path": "test.txt", "offset": 2, "limit": 1, "col_offset": 2, "col_limit": 100 }),
                expect_error: false,
                expect_contains: "2:2\teta",
            },
        ];

        let tool = ReadFileTool;
        let ctx = test_ctx(dir.path());

        for case in cases {
            let result = tool.call(case.input, &ctx).await.unwrap();
            let is_error = matches!(result, ToolResult::Error(_));
            let (ToolResult::Success(content)
            | ToolResult::Error(content)
            | ToolResult::SchemaError(content)) = &result;
            assert_eq!(
                is_error, case.expect_error,
                "case '{}': expected is_error={}, got is_error={}",
                case.name, case.expect_error, is_error
            );
            assert!(
                content.contains(case.expect_contains),
                "case '{}': expected content to contain {:?}, got {:?}",
                case.name,
                case.expect_contains,
                content
            );
        }
    }

    #[tokio::test]
    async fn read_past_eof_returns_empty() {
        let dir = crate::test_util::TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "alpha\nbeta\n").unwrap();

        let result = ReadFileTool
            .call(
                serde_json::json!({ "path": "test.txt", "offset": 100 }),
                &test_ctx(dir.path()),
            )
            .await
            .unwrap();

        let ToolResult::Success(content) = &result else {
            panic!("offset past EOF should succeed with an empty slice, got {result:?}");
        };
        assert_eq!(content, "");
    }

    #[tokio::test]
    async fn col_offset_snaps_to_char_boundary() {
        let dir = crate::test_util::TempDir::new().unwrap();
        // 'é' is two bytes; col_offset 5 lands on its second byte.
        std::fs::write(dir.path().join("test.txt"), "caféx\n").unwrap();

        let result = ReadFileTool
            .call(
                serde_json::json!({ "path": "test.txt", "col_offset": 5 }),
                &test_ctx(dir.path()),
            )
            .await
            .unwrap();

        let ToolResult::Success(content) = &result else {
            panic!("slicing mid-codepoint should succeed, got {result:?}");
        };
        // The slice starts at the next char boundary instead of splitting 'é'.
        assert_eq!(content, "1:6\tx");
    }
}
