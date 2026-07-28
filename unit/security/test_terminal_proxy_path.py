r"""Regression: the terminal proxy path sanitizer must reject backslashes.

open-webui 0.11.0 fix `b40b6fd69` (#27198): `_sanitize_proxy_path` normalized the
path with `posixpath.normpath`, which only splits on `/`. A segment like
`a/..\..\etc` therefore survives as a single opaque component and passes the
`..`-prefix check, but an upstream that treats `\` as a separator resolves it and
escapes the base: directory traversal smuggled past the sanitizer. The fix
rejects any decoded path containing a backslash outright.

The rejection must come *after* the repeated-unquote loop, otherwise `%5c`
(an encoded backslash) still slips through.

Discriminates: passes on v0.11.0, fails on v0.10.2 (backslash paths accepted).
"""

import pytest

pytestmark = pytest.mark.regression


@pytest.mark.parametrize(
    "path",
    [
        r"a/..\..\etc/passwd",
        r"..\..\etc/passwd",
        r"safe\segment",
        "a/%5c..%5cetc",  # encoded backslash, must be caught after decoding
        "a/%255c..%255cetc",  # doubly encoded
    ],
)
def test_backslash_paths_are_rejected(terminals_router_module, path):
    assert terminals_router_module._sanitize_proxy_path(path) is None, (
        f"{path!r} was accepted; an upstream treating '\\' as a separator would "
        "resolve it out of the base directory (#27198)"
    )


@pytest.mark.parametrize(
    "path, expected",
    [
        ("files/report.pdf", "files/report.pdf"),
        ("/files/report.pdf", "files/report.pdf"),
        ("files/", "files/"),
        ("files/./report.pdf", "files/report.pdf"),
        ("files/sub/../report.pdf", "files/report.pdf"),
    ],
)
def test_ordinary_paths_still_pass_through(terminals_router_module, path, expected):
    """Sanity: the fix must not start refusing legitimate proxy paths."""
    assert terminals_router_module._sanitize_proxy_path(path) == expected


@pytest.mark.parametrize("path", ["../etc/passwd", "..", ".", "%2e%2e/etc", "%252e%252e/etc"])
def test_forward_slash_traversal_still_rejected(terminals_router_module, path):
    """Sanity: the pre-existing traversal guards are intact."""
    assert terminals_router_module._sanitize_proxy_path(path) is None
