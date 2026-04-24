//! Cross-process advisory file lock. Serialises writers that share one on-disk store across processes.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::time::Duration;

use crate::persistence::error::{PersistenceError, PersistenceResult as Result};

const MAX_LOCK_RETRIES: u32 = 30;
const MIN_BACKOFF: Duration = Duration::from_millis(5);
const MAX_BACKOFF: Duration = Duration::from_millis(100);

/// Run `f` while holding an exclusive advisory lock on `lock_path`.
/// Retries with exponential backoff up to `MAX_LOCK_RETRIES`, then fails.
pub(crate) fn with_file_lock<T>(lock_path: &Path, f: impl FnOnce() -> Result<T>) -> Result<T> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(lock_path)?;

    let mut backoff = MIN_BACKOFF;
    for _ in 0..MAX_LOCK_RETRIES {
        if try_lock_exclusive(&file)? {
            // flock releases when `file` drops on return: panic-safe, no explicit unlock.
            return f();
        }
        std::thread::sleep(backoff);
        backoff = (backoff * 2).min(MAX_BACKOFF);
    }

    Err(PersistenceError::LockFailed {
        attempts: MAX_LOCK_RETRIES,
    })
}

#[cfg(unix)]
fn try_lock_exclusive(file: &File) -> Result<bool> {
    use std::os::unix::io::AsRawFd;
    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        return Ok(true);
    }
    let err = std::io::Error::last_os_error();
    if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
        return Ok(false);
    }
    Err(PersistenceError::IoFailed(err))
}

#[cfg(not(unix))]
fn try_lock_exclusive(_file: &File) -> Result<bool> {
    Ok(true)
}
