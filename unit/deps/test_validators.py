"""Dependency contract: validators (import name ``validators``).

The Open WebUI backend uses ``validators.url`` as the first gate on
user-supplied URLs in the web-retrieval / RAG fetch path — a
SECURITY-relevant code path (it feeds the SSRF defenses). There are two
call sites with two *different* idioms that both depend on the exact return
contract of ``validators.url``:

  - ``retrieval/web/utils.py`` (``validate_url``):
        ``if isinstance(validators.url(url), validators.ValidationError):
            raise ValueError(...)``
    i.e. it relies on an *invalid* URL returning a ``ValidationError``
    *instance* (not raising, not returning ``False``).
  - ``retrieval/web/main.py``:
        ``if not validators.url(url): ...``
    i.e. it relies on an invalid URL being *falsy*.

For both to be correct simultaneously, ``validators.url`` must:
  - return the literal ``True`` for a valid URL (truthy);
  - return a ``ValidationError`` *instance* for an invalid URL, and that
    instance must be *falsy* (so ``not result`` is True).

This module pins exactly that dual contract, plus a broad battery of
valid/invalid URLs, all OFFLINE and deterministic.

It also pins two load-bearing *negative* facts the backend's own code
compensates for, so a silent change in validators is caught:
  - ``validators.url`` accepts non-HTTP schemes (e.g. ``ftp://``) by
    default — which is why ``validate_url`` has its OWN explicit
    http/https scheme check afterwards;
  - ``validators.url`` accepts private/loopback IPs (e.g.
    ``http://127.0.0.1``) by default — which is why the backend does its
    OWN private-IP resolution check. ``validators.url`` is NOT, by itself,
    an SSRF filter; if a future version started rejecting these, the
    behaviour of the fetch path would change and this test flags it.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "validators"
DIST_NAME = "validators"

USED_SYMBOLS = ["url", "ValidationError"]

# URLs validators.url accepts (returns True). Kept conservative: only forms
# the library is documented to accept and the backend may legitimately fetch.
VALID_URLS = [
    "https://example.com",
    "http://example.com",
    "https://example.com/path",
    "http://example.com/path?query=1&x=2",
    "https://example.com/path?q=1#fragment",
    "https://example.com:8080/x",
    "https://sub.domain.example.co.uk/a/b/c",
    "https://user:pass@example.com/secure",
]

# Clearly-invalid strings validators.url rejects (returns a ValidationError).
INVALID_URLS = [
    "",
    "not a url",
    "example.com",  # no scheme
    "//example.com",  # scheme-relative
    "http://",  # no host
    "javascript:alert(1)",  # not a URL host form
    "http://exa mple.com",  # space in host
    "http:///nohost",
]


def _url(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "url")


def _ValidationError(depcheck):
    return depcheck.resolve(depcheck.load(IMPORT_NAME), "ValidationError")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "validators"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_url_is_callable(depcheck):
    depcheck.assert_callable(depcheck.load(IMPORT_NAME), "url")


def test_validation_error_is_exception_class(depcheck):
    """validators.ValidationError must be a class (the isinstance check in
    validate_url depends on it being a usable second arg to isinstance)."""
    import inspect

    ve = _ValidationError(depcheck)
    assert inspect.isclass(ve)
    assert issubclass(ve, Exception)


# --------------------------------------------------------------------------- #
# The dual return contract (the crux both call sites depend on)
# --------------------------------------------------------------------------- #


def test_valid_url_returns_true(depcheck):
    """A valid URL returns the literal True (truthy) — main.py's
    `if not validators.url(url)` must be False for good URLs."""
    url = _url(depcheck)
    result = url("https://example.com")
    assert result is True


def test_invalid_url_returns_validation_error_instance(depcheck):
    """An invalid URL returns a ValidationError *instance* — utils.py's
    `isinstance(validators.url(url), validators.ValidationError)` must be
    True for bad URLs (it must NOT raise, and must NOT return bare False)."""
    url = _url(depcheck)
    ve = _ValidationError(depcheck)
    result = url("not a url")
    assert isinstance(result, ve), (
        f"expected a ValidationError instance for an invalid URL, got "
        f"{result!r} ({type(result)!r}) — utils.py's isinstance gate breaks"
    )


def test_invalid_url_result_is_falsy(depcheck):
    """The returned ValidationError must be falsy — main.py's
    `if not validators.url(url)` must be True for bad URLs. (validators
    overrides __bool__ on ValidationError to return False.)"""
    url = _url(depcheck)
    result = url("not a url")
    assert not result, "ValidationError result is unexpectedly truthy"


def test_invalid_url_does_not_raise(depcheck):
    """validators.url returns (does not raise) on invalid input — both call
    sites assume a return value, not an exception."""
    url = _url(depcheck)
    # Should simply return a falsy ValidationError, no exception.
    result = url("http://exa mple.com")
    assert not result


# --------------------------------------------------------------------------- #
# Behavioural battery: many valid / invalid URLs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("good", VALID_URLS)
def test_valid_urls_accepted(depcheck, good):
    url = _url(depcheck)
    result = url(good)
    assert result is True, f"expected {good!r} to validate, got {result!r}"


@pytest.mark.parametrize("bad", INVALID_URLS)
def test_invalid_urls_rejected(depcheck, bad):
    url = _url(depcheck)
    ve = _ValidationError(depcheck)
    result = url(bad)
    # Must be a falsy ValidationError — satisfies BOTH call-site idioms.
    assert isinstance(result, ve), f"{bad!r} unexpectedly validated ({result!r})"
    assert not result


# --------------------------------------------------------------------------- #
# Load-bearing NEGATIVE facts the backend compensates for
# --------------------------------------------------------------------------- #


def test_url_accepts_non_http_scheme_by_default(depcheck):
    """validators.url accepts ftp:// (and other schemes) by default. This is
    why validate_url adds its OWN `scheme in ['http','https']` check. If
    validators ever started rejecting ftp here, the comment in utils.py would
    be stale and the defense layering would change — flag it."""
    url = _url(depcheck)
    result = url("ftp://example.com")
    assert result is True, (
        "validators.url no longer accepts ftp:// — the backend's separate "
        "http/https scheme check in validate_url assumes validators does NOT "
        "restrict scheme; re-verify the SSRF gate if this changed"
    )


def test_url_accepts_private_ip_by_default(depcheck):
    """validators.url accepts http://127.0.0.1 by default (private=None).
    validators.url is NOT an SSRF filter on its own — the backend's
    private-IP resolution check is what blocks loopback/internal targets.
    Pin this so a silent change to validators' default IP handling is
    noticed (it would shift where the SSRF boundary actually lives)."""
    url = _url(depcheck)
    result = url("http://127.0.0.1")
    assert result is True, (
        "validators.url now rejects private IPs by default — the SSRF "
        "boundary moved; the backend relied on its own IP check, re-verify"
    )


def test_url_private_flag_can_reject_private_ip(depcheck):
    """The library DOES expose private= to reject private hosts; pin that the
    knob exists and works (even though the backend currently does the IP
    check itself rather than using this flag)."""
    url = _url(depcheck)
    result = url("http://127.0.0.1", private=False)
    # With private=False, a private IP must be rejected.
    assert not result


def test_localhost_without_tld_rejected(depcheck):
    """http://localhost has no TLD and validators rejects it by default — pin
    this so the behaviour the fetch path observes for bare-hostname inputs is
    stable."""
    url = _url(depcheck)
    result = url("http://localhost")
    assert not result


# --------------------------------------------------------------------------- #
# Replicate the two call-site predicates end-to-end
# --------------------------------------------------------------------------- #


def test_utils_isinstance_predicate(depcheck):
    """Reproduce utils.validate_url's gate exactly:
    isinstance(validators.url(url), validators.ValidationError) is True for a
    bad URL and False for a good one."""
    url = _url(depcheck)
    ve = _ValidationError(depcheck)
    assert isinstance(url("not a url"), ve) is True
    assert isinstance(url("https://example.com"), ve) is False


def test_main_not_predicate(depcheck):
    """Reproduce main.py's gate exactly: `not validators.url(url)` is True for
    a bad URL and False for a good one."""
    url = _url(depcheck)
    assert (not url("not a url")) is True
    assert (not url("https://example.com")) is False
