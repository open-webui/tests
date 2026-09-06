"""Dependency contract: ftfy (import name ``ftfy``).

ftfy ("fixes text for you") repairs mojibake — text that was decoded with the
wrong codec, producing garbage like ``donâ€™t`` for ``don't``. Open WebUI runs
every loaded document through it in ``retrieval/loaders/main.py``:

    return [
        Document(
            page_content=ftfy.fix_text(doc.page_content, unescape_html=False),
            metadata=doc.metadata,
        )
        for doc in docs
    ]

This is the cleanup step after the encoding-detection fallback chain (which can
land on latin-1, leaving Windows-1252 mojibake that ftfy then repairs). The
backend uses exactly one entry point: ``ftfy.fix_text(str, unescape_html=False)
-> str``. The keyword argument keeps ftfy's default entity decoding off, so a
document's literal ``&amp;`` survives extraction instead of being rewritten.

This module pins that contract so an ftfy bump that renamed ``fix_text``,
changed its signature, or regressed the core repair behaviour fails loudly
here instead of as silently-garbled RAG document text. Pattern mirrors
test_requests.py: a symbol/signature check plus offline behavioural contracts
asserting the properties the loader relies on — clean text passes through
unchanged, classic Windows-1252/UTF-8 mojibake is repaired, the result is
always a ``str``, and the operation is idempotent (re-fixing already-fixed
text is a no-op). Pure CPU, deterministic, no network.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "ftfy"
DIST_NAME = "ftfy"


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`ftfy` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "ftfy"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature (API surface)
# ---------------------------------------------------------------------------


def test_fix_text_exists_and_callable(depcheck):
    """loaders/main.py: ftfy.fix_text(doc.page_content, unescape_html=False).
    The function must exist and be callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "fix_text")


def test_fix_text_accepts_text_arg(depcheck):
    """fix_text is called positionally with the page content string; the `text`
    parameter must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.fix_text, ["text"])


# ---------------------------------------------------------------------------
# Behavioural: the fix_text contract the loader relies on
# ---------------------------------------------------------------------------


def test_behaviour_returns_str(depcheck):
    """fix_text must return a str — its output goes straight into
    Document(page_content=...), which downstream code treats as text."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.fix_text("hello world")
    assert isinstance(out, str)


def test_behaviour_clean_text_passthrough(depcheck):
    """Well-formed text must pass through unchanged — the common case (most
    loaded documents are already valid UTF-8 and must not be mangled)."""
    mod = depcheck.load(IMPORT_NAME)
    clean = "The quick brown fox jumps over the lazy dog. 1234567890!?"
    assert mod.fix_text(clean) == clean


def test_behaviour_empty_string(depcheck):
    """Empty page content (e.g. a blank page) must round-trip to '' without
    error — loaders can yield empty Documents."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.fix_text("") == ""


def test_behaviour_repairs_windows1252_mojibake(depcheck):
    """The canonical reason ftfy is in the loader: Windows-1252 punctuation
    mis-decoded as UTF-8/latin-1 (â€™ for ', â€œ/â€ for smart quotes). fix_text
    must repair it back to the intended characters."""
    mod = depcheck.load(IMPORT_NAME)
    broken = "The Mona Lisa doesnâ€™t have eyebrows."
    fixed = mod.fix_text(broken)
    # The mojibake apostrophe sequence must be gone and a real apostrophe present.
    assert "â€™" not in fixed, f"mojibake not repaired: {fixed!r}"
    assert "doesn" in fixed and "t have eyebrows" in fixed
    # An apostrophe-like character (straight or curly) must remain between them.
    assert "doesn't" in fixed or "doesn’t" in fixed


def test_behaviour_repairs_mojibake_smart_quotes(depcheck):
    """Double smart-quote mojibake (â€œ ... â€) is another frequent loader
    artifact; fix_text must clean both ends."""
    mod = depcheck.load(IMPORT_NAME)
    broken = "He said â€œhelloâ€ to everyone."
    fixed = mod.fix_text(broken)
    assert "â€" not in fixed, f"smart-quote mojibake not repaired: {fixed!r}"
    assert "hello" in fixed


def test_behaviour_idempotent(depcheck):
    """Re-fixing already-fixed text must be a no-op. The loader could plausibly
    process re-ingested content; a non-idempotent fix would corrupt it. Pin
    fix_text(fix_text(x)) == fix_text(x)."""
    mod = depcheck.load(IMPORT_NAME)
    for sample in (
        "plain ascii",
        "The Mona Lisa doesnâ€™t have eyebrows.",
        "café résumé naïve",
        "He said â€œhelloâ€.",
    ):
        once = mod.fix_text(sample)
        twice = mod.fix_text(once)
        assert twice == once, f"fix_text not idempotent for {sample!r}"


def test_behaviour_preserves_valid_unicode(depcheck):
    """Legitimate (correctly-encoded) non-ASCII text must NOT be 'repaired' away.
    fix_text should leave valid accented/CJK text intact so real multilingual
    documents survive the loader."""
    mod = depcheck.load(IMPORT_NAME)
    valid = "café — naïve — 日本語 — Ωμέγα"
    fixed = mod.fix_text(valid)
    # The meaningful tokens must survive (ftfy may normalise the dash, so check
    # the words rather than exact equality of punctuation).
    for token in ("café", "naïve", "日本語", "Ωμέγα"):
        assert token in fixed, f"valid unicode token {token!r} lost: {fixed!r}"


def test_behaviour_unescape_html_off_keeps_entities(depcheck):
    """The loader passes unescape_html=False so a document's own HTML entities
    survive extraction. Pin both that ftfy still accepts the keyword and that it
    honours it: by default fix_text decodes entities on every line before the
    first literal '<', which rewrites the stored document text."""
    mod = depcheck.load(IMPORT_NAME)
    escaped = "if (a &amp;&amp; b) return a &gt; b;"
    assert mod.fix_text(escaped, unescape_html=False) == escaped
    assert mod.fix_text(escaped) != escaped


def test_behaviour_strips_control_chars(depcheck):
    """ftfy removes stray control characters (e.g. a NUL) that can sneak in from
    binary-ish source files; pin that a NUL is cleaned so downstream text
    handling/storage doesn't choke."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.fix_text("good\x00text")
    assert "\x00" not in out
    assert "good" in out and "text" in out
