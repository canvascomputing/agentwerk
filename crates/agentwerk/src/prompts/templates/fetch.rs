//! What `fetch` says about an address it will not fetch or could not read.

pub(crate) const FETCH_TOO_LONG: &str = r###"`url` is {{ length }} characters, over the {{ limit }} character limit. Fetch one page rather than a long generated address."###;

pub(crate) const FETCH_SCHEME_MISSING: &str =
    r###"`url` names no scheme. Write the full address, starting with https://."###;

pub(crate) const FETCH_SCHEME_UNSUPPORTED: &str =
    r###"Scheme `{{ scheme }}` cannot be fetched. Use http or https."###;

pub(crate) const FETCH_CREDENTIALS_PRESENT: &str = r###"`url` carries embedded credentials, which are never sent. Remove the part before the @."###;

pub(crate) const FETCH_HOST_MISSING: &str =
    r###"`url` names no host. Write the full address, such as https://example.com/page."###;

pub(crate) const FETCH_HOST_NOT_RESOLVABLE: &str =
    r###"`{{ host }}` is not a publicly resolvable host name. Fetch a public address instead."###;

pub(crate) const FETCH_TOO_MANY_REDIRECTS: &str = r###"The address redirected more than {{ limit }} times and was not followed further. Fetch the final address directly."###;

pub(crate) const FETCH_REQUEST_FAILED: &str =
    r###"The request failed: {{ error }}. Check the address, or fetch a different one."###;

pub(crate) const FETCH_BODY_NOT_READ: &str = r###"The response body could not be read: {{ error }}. Fetch the address again, or another one."###;

pub(crate) const FETCH_RESPONSE_TOO_LARGE: &str = r###"The response is {{ bytes }} bytes, over the {{ limit }} byte limit, and was not read. Fetch a smaller page."###;

pub(crate) const FETCH_REDIRECT_LOCATION_MISSING: &str = r###"The address redirected without a Location header, so there is nowhere to follow. Fetch the final address directly."###;

pub(super) const TEMPLATES: &[(&str, &str)] = &[
    ("fetch_too_long", FETCH_TOO_LONG),
    ("fetch_scheme_missing", FETCH_SCHEME_MISSING),
    ("fetch_scheme_unsupported", FETCH_SCHEME_UNSUPPORTED),
    ("fetch_credentials_present", FETCH_CREDENTIALS_PRESENT),
    ("fetch_host_missing", FETCH_HOST_MISSING),
    ("fetch_host_not_resolvable", FETCH_HOST_NOT_RESOLVABLE),
    ("fetch_too_many_redirects", FETCH_TOO_MANY_REDIRECTS),
    ("fetch_request_failed", FETCH_REQUEST_FAILED),
    ("fetch_body_not_read", FETCH_BODY_NOT_READ),
    ("fetch_response_too_large", FETCH_RESPONSE_TOO_LARGE),
    (
        "fetch_redirect_location_missing",
        FETCH_REDIRECT_LOCATION_MISSING,
    ),
];
