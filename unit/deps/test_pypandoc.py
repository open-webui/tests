"""Dependency contract: pypandoc.

pypandoc is the Python wrapper around the ``pandoc`` document converter,
pinned in ``backend/requirements.txt``. In Open WebUI it is reached
*transitively*: the document RAG ingestion path (``retrieval/loaders``)
uses langchain's Unstructured loaders, which shell out to pandoc via
pypandoc to convert formats like ``.epub`` / ``.rtf`` / ``.odt``. The
backend code only references it indirectly, and specifically handles its
failure mode: ``routers/retrieval.py`` catches the conversion error and
maps it to a user-facing message:

    if 'No pandoc was found' in str(e):
        ... raise HTTPException(detail=ERROR_MESSAGES.PANDOC_NOT_INSTALLED)

IMPORTANT — two-part dependency: pypandoc (the Python package) is separate
from the ``pandoc`` *binary* it drives. The package can be importable
while the binary is absent (no pandoc on PATH) — in which case any
conversion raises ``OSError('No pandoc was found: ...')``, the exact
string the backend matches. This module:

  * pins the pypandoc API surface + keyword arguments unconditionally
    (importing the package needs no binary);
  * pins that the "No pandoc was found" substring contract still holds
    when the binary is missing (so the backend's error mapping keeps
    working);
  * runs real conversions ONLY when the pandoc binary is available, and
    skips cleanly otherwise.

Everything is offline (local subprocess at most, no network).

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pypandoc"
DIST_NAME = "pypandoc"

# The pypandoc API surface the Unstructured/langchain loader path uses.
USED_SYMBOLS = [
    "convert_text",  # convert an in-memory string
    "convert_file",  # convert a file on disk (loader path)
    "get_pandoc_version",  # availability probe
    "get_pandoc_path",
    "ensure_pandoc_installed",
]

# The substring routers/retrieval.py matches on to detect a missing binary.
PANDOC_MISSING_SUBSTRING = "No pandoc was found"


def _pandoc_available(mod) -> bool:
    try:
        mod.get_pandoc_version()
        return True
    except Exception:
        return False


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pypandoc"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """convert_text/convert_file + the availability helpers must remain on the
    package (these are what the loader stack and the backend's error handling
    rely on)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_convert_functions_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.convert_text)
    assert callable(mod.convert_file)


def test_convert_text_signature(depcheck):
    """convert_text(source, to, format=, extra_args=, ...). The langchain path
    relies on the (source, to, format=) shape; pin those parameter names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.convert_text, ["source", "to", "format"])


def test_convert_file_signature(depcheck):
    """convert_file(source_file, to, format=, ...). The loaders pass a file path
    + target format; pin those parameter names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.convert_file, ["source_file", "to", "format"])


def test_missing_binary_error_substring_contract(depcheck):
    """CRITICAL backend contract: when the pandoc binary is absent, attempting a
    conversion must raise an error whose str() contains 'No pandoc was found',
    because routers/retrieval.py keys its PANDOC_NOT_INSTALLED handling on that
    exact substring. Only assert this when the binary really is missing (the
    condition the backend handles); skip if pandoc is installed."""
    mod = depcheck.load(IMPORT_NAME)
    if _pandoc_available(mod):
        pytest.skip("pandoc binary present; the missing-binary path is not exercised")
    with pytest.raises(Exception) as excinfo:
        mod.convert_text("# x", "html", format="md")
    assert PANDOC_MISSING_SUBSTRING in str(excinfo.value), (
        "pypandoc's missing-binary error no longer contains "
        f"{PANDOC_MISSING_SUBSTRING!r} — routers/retrieval.py's "
        "PANDOC_NOT_INSTALLED mapping would silently break"
    )


def test_ensure_pandoc_installed_is_callable(depcheck):
    """ensure_pandoc_installed() is the documented way to fetch/verify the
    binary. It must remain callable (we do NOT call it — it can download the
    pandoc binary, i.e. network)."""
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.ensure_pandoc_installed)


def test_convert_text_roundtrip_when_pandoc_present(depcheck):
    """When the pandoc binary IS available, convert markdown -> html offline and
    assert the conversion actually ran (this is the real loader behaviour).
    Skips cleanly when the binary is absent."""
    mod = depcheck.load(IMPORT_NAME)
    if not _pandoc_available(mod):
        pytest.skip("pandoc binary not on PATH; conversion path covered by surface tests")
    html = mod.convert_text("# Heading\n\nbody text", "html", format="md")
    assert "<h1" in html
    assert "body text" in html


def test_convert_text_html_to_plain_when_pandoc_present(depcheck):
    """A second direction (html -> plain) to confirm format routing works, when
    the binary is present."""
    mod = depcheck.load(IMPORT_NAME)
    if not _pandoc_available(mod):
        pytest.skip("pandoc binary not on PATH; conversion path covered by surface tests")
    out = mod.convert_text("<p>hello <b>world</b></p>", "plain", format="html")
    assert "hello" in out
    assert "world" in out


def test_get_pandoc_version_returns_str_when_present(depcheck):
    """get_pandoc_version() is the availability probe; when the binary exists it
    must return a version string (the loaders/diagnostics depend on it)."""
    mod = depcheck.load(IMPORT_NAME)
    if not _pandoc_available(mod):
        pytest.skip("pandoc binary not on PATH")
    version = mod.get_pandoc_version()
    assert isinstance(version, str)
    assert version  # non-empty
