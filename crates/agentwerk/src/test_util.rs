//! Temporary directory that deletes itself on drop. In-house replacement for
//! the `tempfile` dev-dependency, scoped to crate tests.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Owns a fresh directory under `std::env::temp_dir()` and removes it on drop.
pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new() -> io::Result<Self> {
        let base = std::env::temp_dir();
        for _ in 0..16 {
            let candidate = base.join(format!("agentwerk-{}-{}", std::process::id(), unique()));
            match std::fs::create_dir(&candidate) {
                Ok(()) => return Ok(Self { path: candidate }),
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(e),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not allocate a unique temp directory",
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

fn unique() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos.wrapping_add(COUNTER.fetch_add(1, Ordering::Relaxed))
}
