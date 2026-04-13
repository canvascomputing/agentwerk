use std::future::Future;
use std::pin::Pin;

use serde_json::Value;

use crate::error::Result;
use crate::tools::tool::{Tool, ToolContext, ToolResult};

pub struct WebFetchTool;

const MAX_URL_LENGTH: usize = 2000;
const MAX_RESPONSE_BYTES: usize = 10 * 1024 * 1024; // 10 MB
const DEFAULT_MAX_LENGTH: usize = 100_000;
const FETCH_TIMEOUT_SECS: u64 = 30;

const DESCRIPTION: &str = "\
Fetch the contents of a URL and return it as text.

- Automatically upgrades HTTP to HTTPS.
- HTML pages are converted to readable plain text.
- Non-HTML content (JSON, plain text, etc.) is returned as-is.
- Response is truncated to max_length characters (default: 100,000).
- Use the prompt parameter to indicate what you're looking for, so you can focus on the relevant parts of the response.
- Will fail for authenticated or private URLs. Check if a specialized tool provides authenticated access first.";

impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        DESCRIPTION
    }

    fn input_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch"
                },
                "prompt": {
                    "type": "string",
                    "description": "What to extract or focus on from the page content"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Max response length in characters (default: 100000)"
                }
            },
            "required": ["url"]
        })
    }

    fn is_read_only(&self) -> bool {
        true
    }

    fn call<'a>(
        &'a self,
        input: Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<ToolResult>> + Send + 'a>> {
        Box::pin(async move {
            let url = match input["url"].as_str() {
                Some(u) => u,
                None => return Ok(ToolResult::error("Missing required parameter: url")),
            };

            let prompt = input["prompt"].as_str().unwrap_or("");
            let max_length = input["max_length"]
                .as_u64()
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_MAX_LENGTH);

            let validated = match validate_url(url) {
                Ok(u) => u,
                Err(msg) => return Ok(ToolResult::error(msg)),
            };

            let client = reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(FETCH_TIMEOUT_SECS))
                .redirect(reqwest::redirect::Policy::limited(10))
                .build()
                .map_err(|e| crate::error::AgenticError::Tool(e.to_string()))?;

            let response = match client
                .get(validated.as_str())
                .header("Accept", "text/html, text/plain, */*")
                .header("User-Agent", "agentcore/web_fetch")
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => return Ok(ToolResult::error(format!("Fetch failed: {e}"))),
            };

            let status = response.status();
            let content_type = response
                .headers()
                .get("content-type")
                .and_then(|v| v.to_str().ok())
                .unwrap_or("")
                .to_string();

            let bytes = match response.bytes().await {
                Ok(b) => {
                    if b.len() > MAX_RESPONSE_BYTES {
                        return Ok(ToolResult::error(format!(
                            "Response too large: {} bytes (max {})",
                            b.len(),
                            MAX_RESPONSE_BYTES
                        )));
                    }
                    b
                }
                Err(e) => return Ok(ToolResult::error(format!("Failed to read response: {e}"))),
            };

            let raw_text = String::from_utf8_lossy(&bytes);

            let text = if content_type.contains("text/html") {
                strip_html(&raw_text)
            } else {
                raw_text.into_owned()
            };

            let mut output = String::new();
            if !prompt.is_empty() {
                output.push_str(&format!("Prompt: {prompt}\n\n"));
            }
            output.push_str(&format!(
                "URL: {url}\nStatus: {status}\nContent-Type: {content_type}\nSize: {} bytes\n\n",
                bytes.len()
            ));

            let remaining = max_length.saturating_sub(output.len());
            if text.len() > remaining {
                output.push_str(&text[..remaining]);
                output.push_str("\n\n[Content truncated...]");
            } else {
                output.push_str(&text);
            }

            Ok(ToolResult::success(output))
        })
    }
}

fn validate_url(url: &str) -> std::result::Result<url::Url, String> {
    if url.len() > MAX_URL_LENGTH {
        return Err(format!("URL too long: {} chars (max {MAX_URL_LENGTH})", url.len()));
    }

    let mut parsed = url::Url::parse(url).map_err(|e| format!("Invalid URL: {e}"))?;

    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err("URLs with embedded credentials are not allowed".into());
    }

    match parsed.host_str() {
        Some(host) => {
            if host.split('.').count() < 2 {
                return Err("URL must have a publicly resolvable hostname".into());
            }
        }
        None => return Err("URL must have a hostname".into()),
    }

    // Upgrade http to https
    if parsed.scheme() == "http" {
        parsed.set_scheme("https").ok();
    }

    if parsed.scheme() != "https" {
        return Err(format!("Unsupported scheme: {}", parsed.scheme()));
    }

    Ok(parsed)
}

fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut chars = html.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '<' {
            // Check for script/style opening tags
            let rest: String = chars.clone().take(10).collect();
            let rest_lower = rest.to_lowercase();
            if rest_lower.starts_with("script") || rest_lower.starts_with("style") {
                in_script = true;
            }
            if rest_lower.starts_with("/script") || rest_lower.starts_with("/style") {
                in_script = false;
            }
            in_tag = true;
            continue;
        }
        if ch == '>' {
            in_tag = false;
            continue;
        }
        if in_tag || in_script {
            continue;
        }
        if ch == '&' {
            let entity = decode_entity(&mut chars);
            result.push_str(&entity);
            continue;
        }
        result.push(ch);
    }

    collapse_whitespace(&result)
}

fn decode_entity(chars: &mut std::iter::Peekable<std::str::Chars>) -> String {
    let mut entity = String::new();
    for _ in 0..10 {
        match chars.peek() {
            Some(&';') => {
                chars.next();
                return resolve_entity(&entity);
            }
            Some(&c) if c.is_alphanumeric() || c == '#' => {
                entity.push(c);
                chars.next();
            }
            _ => break,
        }
    }
    // Not a valid entity, return as-is
    format!("&{entity}")
}

fn resolve_entity(entity: &str) -> String {
    match entity {
        "amp" => "&".into(),
        "lt" => "<".into(),
        "gt" => ">".into(),
        "quot" => "\"".into(),
        "apos" => "'".into(),
        "nbsp" => " ".into(),
        s if s.starts_with("#x") || s.starts_with("#X") => {
            u32::from_str_radix(&s[2..], 16)
                .ok()
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("&{entity};"))
        }
        s if s.starts_with('#') => {
            s[1..]
                .parse::<u32>()
                .ok()
                .and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| format!("&{entity};"))
        }
        _ => format!("&{entity};"),
    }
}

fn collapse_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut blank_lines = 0;

    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_lines += 1;
            if blank_lines <= 2 {
                result.push('\n');
            }
        } else {
            blank_lines = 0;
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(trimmed);
        }
    }

    result.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_url_valid_https() {
        let result = validate_url("https://example.com/page");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().as_str(), "https://example.com/page");
    }

    #[test]
    fn validate_url_upgrades_http() {
        let result = validate_url("http://example.com/page");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().scheme(), "https");
    }

    #[test]
    fn validate_url_rejects_credentials() {
        let result = validate_url("https://user:pass@example.com");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("credentials"));
    }

    #[test]
    fn validate_url_rejects_single_label_host() {
        let result = validate_url("https://localhost/page");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("publicly resolvable"));
    }

    #[test]
    fn validate_url_rejects_too_long() {
        let long = format!("https://example.com/{}", "a".repeat(MAX_URL_LENGTH));
        let result = validate_url(&long);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too long"));
    }

    #[test]
    fn validate_url_rejects_ftp() {
        let result = validate_url("ftp://example.com/file");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unsupported scheme"));
    }

    #[test]
    fn strip_html_basic() {
        let html = "<html><body><h1>Hello</h1><p>World</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn strip_html_entities() {
        let html = "Tom &amp; Jerry &lt;3 &gt; &quot;cats&quot;";
        let text = strip_html(html);
        assert_eq!(text, "Tom & Jerry <3 > \"cats\"");
    }

    #[test]
    fn strip_html_numeric_entities() {
        let html = "&#65;&#x42;";
        let text = strip_html(html);
        assert_eq!(text, "AB");
    }

    #[test]
    fn strip_html_removes_script() {
        let html = "<p>Before</p><script>alert('xss')</script><p>After</p>";
        let text = strip_html(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn strip_html_removes_style() {
        let html = "<style>body { color: red; }</style><p>Content</p>";
        let text = strip_html(html);
        assert!(text.contains("Content"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn collapse_whitespace_limits_blank_lines() {
        let text = "line1\n\n\n\n\nline2";
        let result = collapse_whitespace(text);
        assert_eq!(result, "line1\n\nline2");
    }
}
