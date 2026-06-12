//! File-backed knowledge store that persists across tickets and runs.
//! Pages are markdown files addressed by slug; a compact index is
//! injected into the system prompt. The model-facing `ManageKnowledgeTool`
//! is a thin wrapper in `tools::knowledge` that holds an
//! `Arc<Knowledge>` from this module.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::persistence::{write_atomic, Persist};

const INDEX_FILE: &str = "index.md";
const PAGES_DIR: &str = "pages";
const DEFAULT_INDEX_CHAR_LIMIT: usize = 4000;
const LEGACY_MEMORY_FILE: &str = "memory.jsonl";
const MIGRATED_SUFFIX: &str = ".migrated";

/// One entry in the in-memory index.
#[derive(Clone, Debug)]
struct IndexEntry {
    slug: String,
    summary: String,
}

/// Returned by [`Pages::save`] and [`Pages::remove`] on success so the tool
/// layer can show the model how much of the index budget is consumed.
#[derive(Debug)]
pub struct KnowledgeOutcome {
    pub message: &'static str,
    pub index_chars_used: usize,
    pub index_char_limit: usize,
    pub pages: usize,
}

/// Durable memory the agent curates and shares across tickets and
/// other agents. Written to disk and curated by the agent through
/// `ManageKnowledgeTool`. Pages are individual markdown files under
/// `<dir>/pages/<slug>.md`; the index at `<dir>/index.md` carries
/// one-line summaries injected into the system prompt.
///
/// ```no_run
/// use agentwerk::{Agent, Knowledge};
///
/// # fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let store = Knowledge::load("./.agentwerk")?;
/// let alice = Agent::new().knowledge(&store);
/// let bob = Agent::new().knowledge(&store);
/// # let _ = (alice, bob);
/// # Ok(())
/// # }
/// ```
pub struct Knowledge {
    knowledge_dir: PathBuf,
    index: Mutex<Vec<IndexEntry>>,
    write_lock: Mutex<()>,
    index_char_limit: AtomicUsize,
}

impl Knowledge {
    /// Open or create a knowledge store rooted at `knowledge_dir`.
    /// Creates `<dir>/pages/` if missing. Reads `index.md` if present;
    /// if absent but pages exist, rebuilds from page H1 headings.
    /// If `memory.jsonl` exists and `index.md` does not, migrates
    /// all entries into `pages/legacy-notes.md`.
    pub fn load(knowledge_dir: impl Into<PathBuf>) -> io::Result<Arc<Self>> {
        let knowledge_dir = knowledge_dir.into();
        fs::create_dir_all(knowledge_dir.join(PAGES_DIR))?;

        let index_path = knowledge_dir.join(INDEX_FILE);
        let legacy_path = knowledge_dir.join(LEGACY_MEMORY_FILE);

        let index = if index_path.exists() {
            parse_index_file(&index_path)?
        } else if legacy_path.exists() {
            // Migration: convert memory.jsonl to knowledge format.
            migrate_memory_jsonl(&knowledge_dir)?
        } else {
            // Rebuild from page H1 headings if pages exist but no index.
            rebuild_index_from_pages(&knowledge_dir)?
        };

        Ok(Arc::new(Self {
            knowledge_dir,
            index: Mutex::new(index),
            write_lock: Mutex::new(()),
            index_char_limit: AtomicUsize::new(DEFAULT_INDEX_CHAR_LIMIT),
        }))
    }

    /// Override the rendered-index char budget. Default is 4000. Page
    /// bodies are never capped; only the bullet list injected into the
    /// system prompt is bounded. Chain after `load` before binding the
    /// store to any agent:
    /// `Knowledge::load(dir)?.index_char_limit(12_000)`.
    pub fn index_char_limit(self: Arc<Self>, n: usize) -> Arc<Self> {
        self.index_char_limit.store(n, Ordering::Relaxed);
        self
    }

    /// Rendered index content for the system prompt. Returns the body
    /// of `index.md` (the bullet list), or an empty string when no
    /// pages exist.
    pub fn index(&self) -> String {
        let index = self.index.lock().unwrap();
        render_index(&index)
    }

    /// Sub-handle for the page collection. Save, load, and remove
    /// pages through `knowledge.pages()` so the verb pair stays bare
    /// `save` / `load` and the bootstrap `Knowledge::load(dir)` does
    /// not collide with per-slug loads.
    pub fn pages(&self) -> Pages<'_> {
        Pages { inner: self }
    }

    /// Remove every page file and the index.
    pub fn clear(&self) -> Result<(), String> {
        let _w = self.write_lock.lock().unwrap();

        let pages_dir = self.knowledge_dir.join(PAGES_DIR);
        if pages_dir.exists() {
            for entry in fs::read_dir(&pages_dir).map_err(|e| e.to_string())? {
                let entry = entry.map_err(|e| e.to_string())?;
                if entry.path().extension().is_some_and(|ext| ext == "md") {
                    fs::remove_file(entry.path()).map_err(|e| e.to_string())?;
                }
            }
        }

        let index_path = self.knowledge_dir.join(INDEX_FILE);
        if index_path.exists() {
            fs::remove_file(&index_path).map_err(|e| e.to_string())?;
        }

        *self.index.lock().unwrap() = Vec::new();
        Ok(())
    }
}

/// One knowledge page: the file shape stored under `<dir>/pages/<slug>.md`.
/// `summary` is mirrored into the page frontmatter so `Page::load`
/// recovers it without consulting the index file.
#[derive(Debug, Clone)]
pub struct Page {
    pub slug: String,
    pub summary: String,
    pub content: String,
    pub tags: Vec<String>,
}

impl Persist for Page {
    type Key = String;

    fn save(&self, dir: &Path) -> io::Result<()> {
        let body = render_page(&self.summary, &self.content, &self.tags);
        write_atomic(&page_path(dir, &self.slug), body.as_bytes())
    }

    fn load(dir: &Path, slug: &Self::Key) -> io::Result<Self> {
        let raw = fs::read_to_string(page_path(dir, slug))?;
        let (summary, tags, content) = parse_page(&raw);
        Ok(Page {
            slug: slug.clone(),
            summary,
            content,
            tags,
        })
    }
}

/// Sub-handle returned by [`Knowledge::pages`]. Hosts the verb-bare
/// `save` / `load` / `remove` over the page collection while keeping
/// the index updated and the char budget enforced.
pub struct Pages<'a> {
    inner: &'a Knowledge,
}

impl Pages<'_> {
    /// Upsert `page` and its index entry. Creates or overwrites the
    /// page file and refreshes the index. Returns the outcome with
    /// the new char usage so the tool layer can show progress.
    pub fn save(&self, page: Page) -> Result<KnowledgeOutcome, String> {
        let slug = normalize_slug(&page.slug)?;
        let summary = page.summary.trim();
        if summary.is_empty() {
            return Err("Summary must not be empty".into());
        }
        if page.content.trim().is_empty() {
            return Err("Content must not be empty".into());
        }

        let _w = self.inner.write_lock.lock().unwrap();
        let mut index = self.inner.index.lock().unwrap().clone();

        let entry = IndexEntry {
            slug: slug.clone(),
            summary: summary.to_string(),
        };
        if let Some(pos) = index.iter().position(|e| e.slug == slug) {
            index[pos] = entry;
        } else {
            index.push(entry);
        }

        let limit = self.inner.index_char_limit.load(Ordering::Relaxed);
        let rendered = render_index(&index);
        if rendered.len() > limit {
            return Err(format!(
                "Index at {}/{} chars. This write would push it to {} chars. \
                 Consolidate or remove pages first.",
                render_index(&self.inner.index.lock().unwrap()).len(),
                limit,
                rendered.len(),
            ));
        }

        let normalized = Page {
            slug,
            summary: summary.to_string(),
            content: page.content,
            tags: page.tags,
        };
        normalized
            .save(&self.inner.knowledge_dir)
            .map_err(|e| format!("Failed to write page: {e}"))?;

        let index_body = render_index_file(&index);
        write_atomic(
            &self.inner.knowledge_dir.join(INDEX_FILE),
            index_body.as_bytes(),
        )
        .map_err(|e| format!("Failed to write index: {e}"))?;

        let chars_used = rendered.len();
        let page_count = index.len();
        *self.inner.index.lock().unwrap() = index;
        Ok(KnowledgeOutcome {
            message: "page written",
            index_chars_used: chars_used,
            index_char_limit: limit,
            pages: page_count,
        })
    }

    /// Read the page at `slug`. Returns the full [`Page`] struct.
    pub fn load(&self, slug: &str) -> Result<Page, String> {
        let slug = normalize_slug(slug)?;
        if !page_path(&self.inner.knowledge_dir, &slug).exists() {
            return Err(format!("Page `{slug}` not found"));
        }
        <Page as Persist>::load(&self.inner.knowledge_dir, &slug)
            .map_err(|e| format!("Failed to read page `{slug}`: {e}"))
    }

    /// Delete the page file at `slug` and its index entry.
    pub fn remove(&self, slug: &str) -> Result<KnowledgeOutcome, String> {
        let slug = normalize_slug(slug)?;
        let _w = self.inner.write_lock.lock().unwrap();
        let mut index = self.inner.index.lock().unwrap().clone();

        let pos = index
            .iter()
            .position(|e| e.slug == slug)
            .ok_or_else(|| format!("Page `{slug}` not found in index"))?;
        index.remove(pos);

        let page_path = page_path(&self.inner.knowledge_dir, &slug);
        if page_path.exists() {
            fs::remove_file(&page_path).map_err(|e| format!("Failed to remove page file: {e}"))?;
        }

        let index_body = render_index_file(&index);
        write_atomic(
            &self.inner.knowledge_dir.join(INDEX_FILE),
            index_body.as_bytes(),
        )
        .map_err(|e| format!("Failed to write index: {e}"))?;

        let chars_used = render_index(&index).len();
        let page_count = index.len();
        *self.inner.index.lock().unwrap() = index;
        Ok(KnowledgeOutcome {
            message: "page removed",
            index_chars_used: chars_used,
            index_char_limit: self.inner.index_char_limit.load(Ordering::Relaxed),
            pages: page_count,
        })
    }
}

fn page_path(knowledge_dir: &Path, slug: &str) -> PathBuf {
    knowledge_dir.join(PAGES_DIR).join(format!("{slug}.md"))
}

// ---- slug validation ----

/// Normalize an arbitrary string into a valid slug: lowercase
/// `[a-z0-9-]`, no leading/trailing hyphens, max 60 chars.
/// Slashes, dots, underscores, spaces, and other non-alphanumeric
/// characters are replaced with hyphens; consecutive hyphens are
/// collapsed; file extensions are stripped.
pub(crate) fn normalize_slug(raw: &str) -> Result<String, String> {
    let slug: String = raw
        .to_ascii_lowercase()
        .chars()
        .map(|c| {
            if c.is_ascii_lowercase() || c.is_ascii_digit() {
                c
            } else {
                '-'
            }
        })
        .collect();

    // Collapse runs of hyphens, trim leading/trailing hyphens.
    let mut collapsed = String::with_capacity(slug.len());
    let mut prev_hyphen = true; // true → skip leading hyphens
    for c in slug.chars() {
        if c == '-' {
            if !prev_hyphen {
                collapsed.push('-');
            }
            prev_hyphen = true;
        } else {
            collapsed.push(c);
            prev_hyphen = false;
        }
    }
    // Trim trailing hyphen.
    while collapsed.ends_with('-') {
        collapsed.pop();
    }

    if collapsed.is_empty() {
        return Err("Slug must not be empty".into());
    }

    // Truncate to 60 chars on a hyphen boundary when possible.
    if collapsed.len() > 60 {
        collapsed.truncate(60);
        if let Some(last_hyphen) = collapsed.rfind('-') {
            collapsed.truncate(last_hyphen);
        }
        while collapsed.ends_with('-') {
            collapsed.pop();
        }
    }

    if collapsed.is_empty() {
        return Err("Slug must not be empty".into());
    }

    Ok(collapsed)
}

// ---- frontmatter ----

fn render_page(summary: &str, content: &str, tags: &[String]) -> String {
    let now = format_iso8601_now();
    let summary_line = format!("\nsummary: {summary}");
    let tags_str = if tags.is_empty() {
        String::new()
    } else {
        format!(
            "\ntags: [{}]",
            tags.iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    format!("---\nupdated: {now}{summary_line}{tags_str}\n---\n{content}\n")
}

/// Parse a page file into `(summary, tags, content)`. Legacy pages
/// without a `summary` line in their frontmatter come back with an
/// empty summary; the index file still holds the canonical text.
fn parse_page(raw: &str) -> (String, Vec<String>, String) {
    let trimmed = raw.trim_start();
    if !trimmed.starts_with("---") {
        return (String::new(), Vec::new(), raw.to_string());
    }
    let after_first = &trimmed[3..];
    let Some(end) = after_first.find("\n---") else {
        return (String::new(), Vec::new(), raw.to_string());
    };
    let frontmatter = &after_first[..end];
    let rest = &after_first[end + 4..];
    let content = rest.strip_prefix('\n').unwrap_or(rest).to_string();

    let mut summary = String::new();
    let mut tags = Vec::new();
    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix("summary:") {
            summary = value.trim().to_string();
        } else if let Some(value) = line.strip_prefix("tags:") {
            let body = value.trim().trim_start_matches('[').trim_end_matches(']');
            tags = body
                .split(',')
                .map(|t| t.trim().to_string())
                .filter(|t| !t.is_empty())
                .collect();
        }
    }
    (summary, tags, content)
}

fn strip_frontmatter(raw: &str) -> String {
    parse_page(raw).2
}

// ---- index rendering / parsing ----

fn render_index(entries: &[IndexEntry]) -> String {
    entries
        .iter()
        .map(|e| format!("- **{}** — {}", e.slug, e.summary))
        .collect::<Vec<_>>()
        .join("\n")
}

fn render_index_file(entries: &[IndexEntry]) -> String {
    let body = render_index(entries);
    if body.is_empty() {
        String::new()
    } else {
        format!("{body}\n")
    }
}

fn parse_index_file(path: &Path) -> io::Result<Vec<IndexEntry>> {
    let raw = fs::read_to_string(path)?;
    Ok(parse_index_lines(&raw))
}

fn parse_index_lines(raw: &str) -> Vec<IndexEntry> {
    raw.lines()
        .filter_map(|line| {
            let line = line.trim();
            // Expected format: `- **slug** — summary`
            let rest = line.strip_prefix("- **")?;
            let (slug, rest) = rest.split_once("**")?;
            let summary = rest
                .strip_prefix(" — ")
                .or_else(|| rest.strip_prefix(" - "))?;
            if slug.is_empty() || summary.is_empty() {
                return None;
            }
            Some(IndexEntry {
                slug: slug.to_string(),
                summary: summary.trim().to_string(),
            })
        })
        .collect()
}

fn rebuild_index_from_pages(knowledge_dir: &Path) -> io::Result<Vec<IndexEntry>> {
    let pages_dir = knowledge_dir.join(PAGES_DIR);
    if !pages_dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for entry in fs::read_dir(&pages_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "md") {
            let slug = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            if slug.is_empty() {
                continue;
            }
            let raw = fs::read_to_string(&path)?;
            let body = strip_frontmatter(&raw);
            let summary = extract_h1_summary(&body);
            entries.push(IndexEntry { slug, summary });
        }
    }
    entries.sort_by(|a, b| a.slug.cmp(&b.slug));

    // Write the rebuilt index to disk.
    if !entries.is_empty() {
        let index_body = render_index_file(&entries);
        write_atomic(&knowledge_dir.join(INDEX_FILE), index_body.as_bytes())?;
    }
    Ok(entries)
}

fn extract_h1_summary(body: &str) -> String {
    for line in body.lines() {
        let trimmed = line.trim();
        if let Some(heading) = trimmed.strip_prefix("# ") {
            return heading.trim().to_string();
        }
    }
    "(no summary)".to_string()
}

// ---- migration from memory.jsonl ----

#[derive(serde::Deserialize)]
struct LegacyMemoryRecord {
    content: String,
    #[allow(dead_code)]
    added_at: u64,
}

fn migrate_memory_jsonl(knowledge_dir: &Path) -> io::Result<Vec<IndexEntry>> {
    let legacy_path = knowledge_dir.join(LEGACY_MEMORY_FILE);
    let raw = fs::read_to_string(&legacy_path)?;

    let mut contents: Vec<String> = Vec::new();
    for line in raw.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(record) = serde_json::from_str::<LegacyMemoryRecord>(trimmed) {
            if !record.content.is_empty() {
                contents.push(record.content);
            }
        }
    }

    let entries = if contents.is_empty() {
        Vec::new()
    } else {
        let page_content = contents
            .iter()
            .map(|c| format!("- {c}"))
            .collect::<Vec<_>>()
            .join("\n");
        let page_body = format!("# Legacy Notes\n\n{page_content}");
        let slug = "legacy-notes";
        let summary = format!("Migrated from memory.jsonl ({} entries)", contents.len());

        let page_file = render_page(&summary, &page_body, &[]);
        let page_path = knowledge_dir.join(PAGES_DIR).join(format!("{slug}.md"));
        write_atomic(&page_path, page_file.as_bytes())?;

        let entry = IndexEntry {
            slug: slug.to_string(),
            summary,
        };
        vec![entry]
    };

    // Write the index.
    let index_body = render_index_file(&entries);
    write_atomic(&knowledge_dir.join(INDEX_FILE), index_body.as_bytes())?;

    // Rename the legacy file.
    let migrated_path = knowledge_dir.join(format!("{LEGACY_MEMORY_FILE}{MIGRATED_SUFFIX}"));
    fs::rename(&legacy_path, &migrated_path)?;

    Ok(entries)
}

// ---- timestamp ----

/// ISO 8601 UTC timestamp without external dependencies. Uses the same
/// civil-from-days algorithm as `prompts::format_current_date`, extended
/// with hours, minutes, and seconds.
fn format_iso8601_now() -> String {
    let epoch_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let secs_of_day = epoch_secs % 86400;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    let days = epoch_secs / 86400;
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if month <= 2 { year + 1 } else { year };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh_store() -> (Arc<Knowledge>, crate::test_util::TempDir) {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::load(dir.path()).unwrap();
        (store, dir)
    }

    fn save_page(
        store: &Knowledge,
        slug: &str,
        summary: &str,
        content: &str,
        tags: &[&str],
    ) -> KnowledgeOutcome {
        let page = Page {
            slug: slug.to_string(),
            summary: summary.to_string(),
            content: content.to_string(),
            tags: tags.iter().map(|s| s.to_string()).collect(),
        };
        store.pages().save(page).unwrap()
    }

    #[test]
    fn load_creates_pages_directory() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let nested = dir.path().join("not-yet-there");
        let _ = Knowledge::load(&nested).unwrap();
        assert!(nested.join(PAGES_DIR).exists());
    }

    #[test]
    fn load_with_no_existing_files_starts_empty() {
        let (store, _dir) = fresh_store();
        assert!(store.index().is_empty());
    }

    #[test]
    fn save_page_creates_file_and_index() {
        let (store, dir) = fresh_store();
        save_page(
            &store,
            "deploy-config",
            "Staging on port 8080",
            "# Deploy\n\nPort 8080.",
            &[],
        );
        assert!(dir.path().join(PAGES_DIR).join("deploy-config.md").exists());
        assert!(dir.path().join(INDEX_FILE).exists());
        let idx = store.index();
        assert!(idx.contains("deploy-config"));
        assert!(idx.contains("Staging on port 8080"));
    }

    #[test]
    fn save_page_upserts_existing_entry() {
        let (store, _dir) = fresh_store();
        save_page(&store, "config", "v1", "# Config\n\nVersion 1.", &[]);
        save_page(&store, "config", "v2", "# Config\n\nVersion 2.", &[]);
        let idx = store.index();
        assert!(idx.contains("v2"));
        assert!(!idx.contains("v1"));
        // Only one line in the index (the upserted entry).
        assert_eq!(idx.lines().count(), 1);
    }

    #[test]
    fn load_page_returns_body_without_frontmatter() {
        let (store, _dir) = fresh_store();
        save_page(
            &store,
            "test",
            "A test page",
            "# Test\n\nHello world.",
            &["tag1"],
        );
        let body = store.pages().load("test").map(|p| p.content).unwrap();
        assert!(body.contains("# Test"));
        assert!(body.contains("Hello world."));
        assert!(!body.contains("---"));
        assert!(!body.contains("updated:"));
        assert!(!body.contains("tags:"));
    }

    #[test]
    fn load_page_not_found() {
        let (store, _dir) = fresh_store();
        let err = store.pages().load("nonexistent").unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn remove_page_clears_file_and_index_entry() {
        let (store, dir) = fresh_store();
        save_page(&store, "temp", "Temporary", "# Temp\n\nWill delete.", &[]);
        assert!(dir.path().join(PAGES_DIR).join("temp.md").exists());
        store.pages().remove("temp").unwrap();
        assert!(!dir.path().join(PAGES_DIR).join("temp.md").exists());
        assert!(store.index().is_empty());
    }

    #[test]
    fn remove_page_errors_when_not_in_index() {
        let (store, _dir) = fresh_store();
        let err = store.pages().remove("nonexistent").unwrap_err();
        assert!(err.contains("not found"));
    }

    #[test]
    fn clear_removes_all_pages_and_index() {
        let (store, dir) = fresh_store();
        save_page(&store, "a", "Page A", "# A", &[]);
        save_page(&store, "b", "Page B", "# B", &[]);
        store.clear().unwrap();
        assert!(store.index().is_empty());
        assert!(!dir.path().join(INDEX_FILE).exists());
        // Pages directory exists but is empty.
        assert!(dir.path().join(PAGES_DIR).exists());
        let page_count = fs::read_dir(dir.path().join(PAGES_DIR)).unwrap().count();
        assert_eq!(page_count, 0);
    }

    #[test]
    fn normalize_slug_rejects_empty() {
        assert!(normalize_slug("").is_err());
    }

    #[test]
    fn normalize_slug_truncates_long_input() {
        let long = "a".repeat(80);
        let slug = normalize_slug(&long).unwrap();
        assert!(slug.len() <= 60);
    }

    #[test]
    fn normalize_slug_lowercases() {
        assert_eq!(normalize_slug("Deploy").unwrap(), "deploy");
        assert_eq!(normalize_slug("DEPLOY-Config").unwrap(), "deploy-config");
    }

    #[test]
    fn normalize_slug_strips_leading_trailing_hyphens() {
        assert_eq!(normalize_slug("-deploy").unwrap(), "deploy");
        assert_eq!(normalize_slug("deploy-").unwrap(), "deploy");
        assert_eq!(normalize_slug("--deploy--").unwrap(), "deploy");
    }

    #[test]
    fn normalize_slug_converts_spaces_and_underscores() {
        assert_eq!(normalize_slug("deploy config").unwrap(), "deploy-config");
        assert_eq!(normalize_slug("deploy_config").unwrap(), "deploy-config");
    }

    #[test]
    fn normalize_slug_converts_file_paths() {
        assert_eq!(
            normalize_slug("src/malware/bad_file.py").unwrap(),
            "src-malware-bad-file-py"
        );
    }

    #[test]
    fn normalize_slug_collapses_consecutive_hyphens() {
        assert_eq!(normalize_slug("a---b").unwrap(), "a-b");
        assert_eq!(normalize_slug("a/./b").unwrap(), "a-b");
    }

    #[test]
    fn normalize_slug_passes_through_valid() {
        assert_eq!(normalize_slug("deploy-config").unwrap(), "deploy-config");
        assert_eq!(normalize_slug("a").unwrap(), "a");
        assert_eq!(normalize_slug("config-v2").unwrap(), "config-v2");
        assert_eq!(normalize_slug("a1b2c3").unwrap(), "a1b2c3");
    }

    #[test]
    fn frontmatter_is_stripped_correctly() {
        let raw = "---\nupdated: 2026-01-01T00:00:00Z\ntags: [a, b]\n---\n# Title\n\nBody.";
        let body = strip_frontmatter(raw);
        assert_eq!(body, "# Title\n\nBody.");
    }

    #[test]
    fn strip_frontmatter_no_frontmatter() {
        let raw = "# Title\n\nBody.";
        assert_eq!(strip_frontmatter(raw), raw);
    }

    #[test]
    fn index_char_limit_rejects_oversized_write() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let store = Knowledge::load(dir.path()).unwrap();
        // Write a very long summary to push the index past the limit.
        let long_summary = "x".repeat(DEFAULT_INDEX_CHAR_LIMIT + 1);
        let err = store
            .pages()
            .save(Page {
                slug: "big".into(),
                summary: long_summary.clone(),
                content: "# Big".into(),
                tags: vec![],
            })
            .unwrap_err();
        assert!(err.contains("chars"), "{err}");
    }

    #[test]
    fn index_char_limit_can_be_lowered_after_open() {
        let (store, _dir) = fresh_store();
        let store = store.index_char_limit(80);

        // 80-char budget rejects what the default 4000-char budget would accept.
        let long_summary = "x".repeat(200);
        let err = store
            .pages()
            .save(Page {
                slug: "big".into(),
                summary: long_summary.clone(),
                content: "# Big".into(),
                tags: vec![],
            })
            .unwrap_err();
        assert!(err.contains("chars"), "{err}");

        // Outcome reports the custom limit.
        let out = save_page(&store, "small", "ok", "# Small", &[]);
        assert_eq!(out.index_char_limit, 80);
    }

    #[test]
    fn outcome_reports_usage() {
        let (store, _dir) = fresh_store();
        let out = save_page(&store, "test", "A test", "# Test\n\nContent.", &[]);
        assert_eq!(out.message, "page written");
        assert_eq!(out.pages, 1);
        assert_eq!(out.index_char_limit, DEFAULT_INDEX_CHAR_LIMIT);
        assert!(out.index_chars_used > 0);
    }

    #[test]
    fn writes_through_one_arc_clone_are_visible_through_another() {
        let (store, _dir) = fresh_store();
        let other = Arc::clone(&store);
        save_page(
            &store,
            "shared",
            "Shared note",
            "# Shared\n\nShared content.",
            &[],
        );
        assert!(other.index().contains("shared"));
    }

    #[test]
    fn entries_survive_drop_and_reopen() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let s1 = Knowledge::load(dir.path()).unwrap();
        save_page(
            &s1,
            "durable",
            "Survives restart",
            "# Durable\n\nPersisted.",
            &[],
        );
        drop(s1);
        let s2 = Knowledge::load(dir.path()).unwrap();
        assert!(s2.index().contains("durable"));
        assert!(s2.index().contains("Survives restart"));
    }

    #[test]
    fn rebuild_index_from_pages_when_index_missing() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let pages_dir = dir.path().join(PAGES_DIR);
        fs::create_dir_all(&pages_dir).unwrap();
        fs::write(
            pages_dir.join("my-page.md"),
            "---\nupdated: 2026-01-01T00:00:00Z\n---\n# My Page\n\nContent here.",
        )
        .unwrap();
        // Open without an existing index.md.
        let store = Knowledge::load(dir.path()).unwrap();
        let idx = store.index();
        assert!(idx.contains("my-page"));
        assert!(idx.contains("My Page"));
    }

    #[test]
    fn migration_from_memory_jsonl() {
        let dir = crate::test_util::TempDir::new().unwrap();
        let legacy = dir.path().join(LEGACY_MEMORY_FILE);
        fs::write(
            &legacy,
            "{\"content\":\"fact one\",\"added_at\":1}\n\
             {\"content\":\"fact two\",\"added_at\":2}\n",
        )
        .unwrap();
        let store = Knowledge::load(dir.path()).unwrap();
        let idx = store.index();
        assert!(idx.contains("legacy-notes"));
        assert!(idx.contains("2 entries"));
        // The legacy file should be renamed.
        assert!(!legacy.exists());
        assert!(dir
            .path()
            .join(format!("{LEGACY_MEMORY_FILE}{MIGRATED_SUFFIX}"))
            .exists());
        // The page should exist.
        assert!(dir.path().join(PAGES_DIR).join("legacy-notes.md").exists());
        let body = store
            .pages()
            .load("legacy-notes")
            .map(|p| p.content)
            .unwrap();
        assert!(body.contains("fact one"));
        assert!(body.contains("fact two"));
    }

    #[test]
    fn save_page_with_tags() {
        let (store, dir) = fresh_store();
        save_page(
            &store,
            "tagged",
            "A tagged page",
            "# Tagged\n\nWith tags.",
            &["config", "deploy"],
        );
        let raw = fs::read_to_string(dir.path().join(PAGES_DIR).join("tagged.md")).unwrap();
        assert!(raw.contains("tags: [config, deploy]"));
    }

    #[test]
    fn parse_index_lines_handles_em_dash_and_hyphen() {
        let input = "- **slug-one** — Summary one\n- **slug-two** - Summary two";
        let entries = parse_index_lines(input);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].slug, "slug-one");
        assert_eq!(entries[0].summary, "Summary one");
        assert_eq!(entries[1].slug, "slug-two");
        assert_eq!(entries[1].summary, "Summary two");
    }

    #[test]
    fn write_page_rejects_empty_summary() {
        let (store, _dir) = fresh_store();
        let err = store
            .pages()
            .save(Page {
                slug: "test".into(),
                summary: ("").to_string(),
                content: "content".into(),
                tags: vec![],
            })
            .unwrap_err();
        assert!(err.contains("Summary"));
    }

    #[test]
    fn write_page_rejects_empty_content() {
        let (store, _dir) = fresh_store();
        let err = store
            .pages()
            .save(Page {
                slug: "test".into(),
                summary: ("summary").to_string(),
                content: "".into(),
                tags: vec![],
            })
            .unwrap_err();
        assert!(err.contains("Content"));
    }

    #[test]
    fn remove_page_returns_outcome() {
        let (store, _dir) = fresh_store();
        save_page(&store, "temp", "Temp", "# Temp", &[]);
        let out = store.pages().remove("temp").unwrap();
        assert_eq!(out.message, "page removed");
        assert_eq!(out.pages, 0);
    }
}
