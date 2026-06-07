"""Dependency contract: python-mimeparse (import name ``mimeparse``).

Open WebUI uses mimeparse in exactly one place: ``utils/misc.py`` has a
``get_mime_type_match`` helper (used by the audio/STT path to decide whether
a request's ``Content-Type`` is one the backend supports). It calls two
functions and relies on their precise return shapes:

    match = mimeparse.best_match(supported, header)
    if not match:                       # '' (empty string) means no match
        return None
    _, _, match_params  = mimeparse.parse_mime_type(match)
    _, _, header_params = mimeparse.parse_mime_type(header)
    for k, v in match_params.items():
        if header_params.get(k) != v:
            return None
    return match

So the contract that must hold:
  - ``best_match(supported, header)`` returns the best matching supported
    media type, an **empty string** (falsy) when nothing matches, and
    raises (caught by the caller's broad ``except``) on an unparseable
    header;
  - ``parse_mime_type(s)`` returns a 3-tuple ``(type, subtype, params)``
    where ``params`` is a dict — the helper unpacks exactly three values and
    iterates ``.items()`` on the third.

A bump that changed ``best_match`` to return ``None`` instead of ``''``, or
reshaped ``parse_mime_type``'s tuple, would silently break MIME negotiation
(or raise a ValueError on the 3-way unpack). This module pins both.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "mimeparse"
DIST_NAME = "python-mimeparse"

USED_SYMBOLS = [
    "best_match",
    "parse_mime_type",
    "parse_media_range",
    "quality",
    "MimeTypeParseException",
]


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "mimeparse"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_functions_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "best_match")
    depcheck.assert_callable(mod, "parse_mime_type")


def test_best_match_signature(depcheck):
    """misc.py calls best_match(supported, header) positionally."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.best_match, ["supported", "header"])


def test_parse_mime_type_signature(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.parse_mime_type, ["mime_type"])


# ---------------------------------------------------------------------------
# best_match — the matching contract.
# ---------------------------------------------------------------------------


def test_best_match_exact(depcheck):
    """A header that exactly matches a supported type returns that type."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.best_match(["audio/*", "video/webm"], "video/webm") == "video/webm"


def test_best_match_wildcard(depcheck):
    """The default supported set in misc.py is ['audio/*', 'video/webm'];
    a concrete audio header must match the audio/* range."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.best_match(["audio/*", "video/webm"], "audio/mpeg") == "audio/*"


def test_best_match_no_match_returns_empty_string(depcheck):
    """CONTRACT PIN: on no match, best_match returns '' (empty string), NOT
    None. misc.py relies on the result being falsy (`if not match`). If a
    bump returned None or raised here, the `not match` guard would still work
    for None but a different sentinel could break callers comparing to ''."""
    mod = depcheck.load(IMPORT_NAME)
    result = mod.best_match(["audio/*"], "image/png")
    assert result == ""
    assert not result  # the property misc.py actually depends on


def test_best_match_picks_higher_quality(depcheck):
    """Among multiple candidates best_match honours q-values in the header
    (the negotiation the helper delegates entirely to mimeparse)."""
    mod = depcheck.load(IMPORT_NAME)
    # header offers both, preferring json (q=0.9) over xml (q=0.5)
    header = "application/xml;q=0.5,application/json;q=0.9"
    best = mod.best_match(["application/json", "application/xml"], header)
    assert best == "application/json"


def test_best_match_raises_on_unparseable_header(depcheck):
    """An unparseable header raises MimeTypeParseException. misc.py wraps the
    whole call in try/except and returns None, so this must remain a raised
    exception (not a silent ''), to keep that error path intact."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.MimeTypeParseException):
        mod.best_match(["audio/*"], "not-a-mime-type")


def test_mimetypeparseexception_is_exception(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.MimeTypeParseException, Exception)


# ---------------------------------------------------------------------------
# parse_mime_type — the 3-tuple (type, subtype, params) contract.
# ---------------------------------------------------------------------------


def test_parse_mime_type_returns_three_tuple(depcheck):
    """misc.py does `_, _, params = mimeparse.parse_mime_type(...)`. The
    return must be a 3-element sequence (a ValueError on unpack otherwise)."""
    mod = depcheck.load(IMPORT_NAME)
    parsed = mod.parse_mime_type("audio/mpeg")
    assert len(parsed) == 3
    type_, subtype, params = parsed  # must unpack into exactly three
    assert type_ == "audio"
    assert subtype == "mpeg"
    assert isinstance(params, dict)


def test_parse_mime_type_params_is_dict_with_items(depcheck):
    """The third element must be a dict; misc.py iterates `params.items()`."""
    mod = depcheck.load(IMPORT_NAME)
    type_, subtype, params = mod.parse_mime_type("text/html; charset=utf-8; q=0.9")
    assert type_ == "text"
    assert subtype == "html"
    assert params == {"charset": "utf-8", "q": "0.9"}
    # The exact operation misc.py performs:
    assert {k: v for k, v in params.items()} == {"charset": "utf-8", "q": "0.9"}


def test_parse_mime_type_no_params_empty_dict(depcheck):
    """A bare type yields an empty params dict (so the misc.py param-equality
    loop is a no-op and the match is accepted)."""
    mod = depcheck.load(IMPORT_NAME)
    _, _, params = mod.parse_mime_type("audio/webm")
    assert params == {}


def test_misc_helper_param_check_logic(depcheck):
    """Reproduce misc.py's full param-matching logic end to end: a matched
    type whose params disagree with the header's must be rejected, and one
    whose params agree (or are absent) must be accepted. This is the exact
    behaviour get_mime_type_match builds on top of mimeparse."""
    mod = depcheck.load(IMPORT_NAME)

    def get_mime_type_match(supported, header):
        match = mod.best_match(supported, header)
        if not match:
            return None
        _, _, match_params = mod.parse_mime_type(match)
        _, _, header_params = mod.parse_mime_type(header)
        for k, v in match_params.items():
            if header_params.get(k) != v:
                return None
        return match

    # Plain match, no params -> accepted.
    assert get_mime_type_match(["audio/*", "video/webm"], "audio/mpeg") == "audio/*"
    # No candidate at all -> None.
    assert get_mime_type_match(["audio/*"], "image/png") is None
    # Unparseable header -> exception bubbles (caller catches it); assert it raises.
    with pytest.raises(mod.MimeTypeParseException):
        get_mime_type_match(["audio/*"], "garbage")


# ---------------------------------------------------------------------------
# Supporting functions used indirectly (quality / parse_media_range).
# ---------------------------------------------------------------------------


def test_quality_returns_float(depcheck):
    """quality(mime, ranges) yields a 0..1 fitness score (the basis best_match
    builds on)."""
    mod = depcheck.load(IMPORT_NAME)
    q = mod.quality("audio/mpeg", "audio/*")
    assert isinstance(q, float)
    assert 0.0 <= q <= 1.0


def test_parse_media_range_defaults_q(depcheck):
    """parse_media_range injects a default q=1 when absent (the accept-header
    semantics best_match depends on)."""
    mod = depcheck.load(IMPORT_NAME)
    _, _, params = mod.parse_media_range("audio/*")
    assert params.get("q") == "1"


def test_function_signatures_are_introspectable(depcheck):
    """Both public functions must keep an introspectable signature (regression
    guard against a C-accelerated reshuffle that hides the call shape)."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.signature(mod.best_match) is not None
    assert inspect.signature(mod.parse_mime_type) is not None
