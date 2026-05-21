//! Persistence contracts used by every value that reads or writes a
//! file in agentwerk. `Persist` covers whole-value state files;
//! `Append` covers jsonl append-only logs. Each implementer encodes
//! its own file location, so the wrong file cannot be reached through
//! the wrong type.

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) trait Persist: Sized {
    type Key;
    fn save(&self, dir: &Path) -> io::Result<()>;
    fn load(dir: &Path, key: &Self::Key) -> io::Result<Self>;
}

pub(crate) trait Append {
    type Record;
    fn append(dir: &Path, record: &Self::Record) -> io::Result<()>;
}

pub(crate) struct Results;

impl Append for Results {
    type Record = serde_json::Value;
    fn append(dir: &Path, record: &Self::Record) -> io::Result<()> {
        let line = serde_json::to_string(record).map_err(io::Error::other)?;
        append_line(&dir.join("results.jsonl"), &line)
    }
}

pub(crate) struct TicketEvents;

impl Append for TicketEvents {
    type Record = serde_json::Value;
    fn append(dir: &Path, record: &Self::Record) -> io::Result<()> {
        let line = serde_json::to_string(record).map_err(io::Error::other)?;
        append_line(&dir.join("tickets.jsonl"), &line)
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "out".to_string());
    let temp = parent.join(format!(".{file_name}.tmp.{pid}.{counter}"));
    let result = (|| -> io::Result<()> {
        let mut f = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&temp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

pub(crate) fn append_line(path: &Path, line: &str) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    file.write_all(line.as_bytes())?;
    file.write_all(b"\n")
}

/// Newest `ticket.<ts>.json` file in `dir`, or `None` when the directory
/// is absent or empty.
pub(crate) fn latest_path(dir: &Path) -> Option<PathBuf> {
    fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter_map(|e| {
            let name = e.file_name();
            let ts = parse_filename_ts(name.to_string_lossy().as_ref())?;
            Some((ts, e.path()))
        })
        .max_by_key(|(ts, _)| *ts)
        .map(|(_, path)| path)
}

pub(crate) fn parse_filename_ts(name: &str) -> Option<u64> {
    name.strip_prefix("ticket.")?
        .strip_suffix(".json")?
        .parse()
        .ok()
}

/// Relative path of a tool's output file under a tickets dir:
/// `tickets/<key>/outputs/<id>.txt`. Callers join with the tickets dir
/// to write; storing the relative form keeps comment transcripts portable.
pub(crate) fn output_path(key: &str, id: &str) -> PathBuf {
    PathBuf::from("tickets")
        .join(key)
        .join("outputs")
        .join(format!("{id}.txt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::TempDir;

    #[test]
    fn write_atomic_creates_parent_and_writes_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nested").join("file.txt");
        write_atomic(&path, b"hello").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"hello");
    }

    #[test]
    fn write_atomic_overwrites_existing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("f.txt");
        write_atomic(&path, b"v1").unwrap();
        write_atomic(&path, b"v2").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"v2");
    }

    #[test]
    fn append_line_creates_file_and_appends_newline() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("log.jsonl");
        append_line(&path, "first").unwrap();
        append_line(&path, "second").unwrap();
        assert_eq!(fs::read_to_string(&path).unwrap(), "first\nsecond\n");
    }

    #[test]
    fn latest_path_picks_highest_timestamp() {
        let dir = TempDir::new().unwrap();
        let d = dir.path();
        fs::write(d.join("ticket.10.json"), "").unwrap();
        fs::write(d.join("ticket.30.json"), "").unwrap();
        fs::write(d.join("ticket.20.json"), "").unwrap();
        let picked = latest_path(d).unwrap();
        assert_eq!(picked.file_name().unwrap(), "ticket.30.json");
    }

    #[test]
    fn latest_path_ignores_unrelated_files() {
        let dir = TempDir::new().unwrap();
        let d = dir.path();
        fs::write(d.join("ticket.1.json"), "").unwrap();
        fs::write(d.join("README.md"), "").unwrap();
        let picked = latest_path(d).unwrap();
        assert_eq!(picked.file_name().unwrap(), "ticket.1.json");
    }

    #[test]
    fn parse_filename_ts_accepts_well_formed() {
        assert_eq!(parse_filename_ts("ticket.123.json"), Some(123));
    }

    #[test]
    fn parse_filename_ts_rejects_malformed() {
        assert_eq!(parse_filename_ts("ticket.abc.json"), None);
        assert_eq!(parse_filename_ts("note.123.json"), None);
        assert_eq!(parse_filename_ts("ticket.123.txt"), None);
    }

    #[test]
    fn results_appends_a_record_to_results_jsonl() {
        let dir = TempDir::new().unwrap();
        Results::append(dir.path(), &serde_json::json!({"k": 1})).unwrap();
        Results::append(dir.path(), &serde_json::json!({"k": 2})).unwrap();
        let body = fs::read_to_string(dir.path().join("results.jsonl")).unwrap();
        assert_eq!(body, "{\"k\":1}\n{\"k\":2}\n");
    }

    #[test]
    fn ticket_events_appends_to_tickets_jsonl() {
        let dir = TempDir::new().unwrap();
        TicketEvents::append(dir.path(), &serde_json::json!({"event": "created"})).unwrap();
        let body = fs::read_to_string(dir.path().join("tickets.jsonl")).unwrap();
        assert_eq!(body, "{\"event\":\"created\"}\n");
    }
}
