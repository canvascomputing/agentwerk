//! What the knowledge tool says when a page cannot be read, saved, or removed.

pub(crate) const KNOWLEDGE_PAGE_NOT_FOUND: &str = r###"No page found for `{{ slug }}`. The `list` action shows every page that exists: an unlisted slug cannot be read."###;

pub(crate) const KNOWLEDGE_WRITE_FAILED: &str =
    r###"The page could not be saved: {{ error }}. Nothing was written."###;

pub(crate) const KNOWLEDGE_REMOVE_FAILED: &str =
    r###"The page could not be removed: {{ error }}. It is still there."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("knowledge_page_not_found", KNOWLEDGE_PAGE_NOT_FOUND),
    ("knowledge_write_failed", KNOWLEDGE_WRITE_FAILED),
    ("knowledge_remove_failed", KNOWLEDGE_REMOVE_FAILED),
];
