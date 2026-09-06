"""Dependency contract: Markdown (PyPI ``Markdown``, import name ``markdown``).

Open WebUI converts Markdown to HTML with this library in ``env.py`` (startup):
``import markdown`` then ``html_content = markdown.markdown(changelog_content)``.
The bundled CHANGELOG.md is rendered to HTML and then parsed with BeautifulSoup
to build the in-app "what's new" structure. A render failure here happens at
import time. The PDF export that also rendered through it is gone with
``utils/pdf_generator.py`` (508de2077).

The call site uses the plain ``markdown.markdown(text)`` form (no extensions),
so the load-bearing contract is: ``markdown.markdown`` exists and turns standard
Markdown (headings, bold/italic, lists, links, code, code fences, blockquotes)
into the corresponding HTML. This module pins that surface and exercises the
conversions for real. The ``pymdownx.*`` checks below skip when
pymdown-extensions is not installed; it is no longer a pinned dependency.
A `Markdown` bump that removed/renamed ``markdown.markdown`` or broke a core
conversion fails loudly here instead of at changelog-render time.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks,
plus offline render contracts (pure CPU, no network). Uses the `depcheck`
fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "markdown"
DIST_NAME = "Markdown"

# Top-level names the backend (and the standard API) resolves on `markdown`.
USED_SYMBOLS = ["markdown", "Markdown"]

# pymdownx extensions the codebase references / pins (pymdown-extensions). These
# must remain *loadable as markdown extensions* even though the live call sites
# pass no extensions today — the PDF path is written to opt into them.
PYMDOWNX_EXTENSIONS = ["pymdownx.extra", "pymdownx.superfences", "pymdownx.highlight"]


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "markdown"


def test_version_reported(depcheck):
    """The installed distribution version (PyPI name ``Markdown``) must be
    resolvable so bump tooling and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """``markdown.markdown`` (the function) and ``markdown.Markdown`` (the class)
    must both exist — the function is what both call sites use."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_markdown_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "markdown")


def test_markdown_signature(depcheck):
    """Both call sites do ``markdown.markdown(text)`` / ``markdown(text,
    extensions=[...])`` — first positional ``text``, plus **kwargs for
    extensions. Pin the first parameter name."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.markdown, ["text"])


# --------------------------------------------------------------------------- #
# Core rendering — the conversions the changelog + chat content rely on
# --------------------------------------------------------------------------- #
def test_render_returns_str(depcheck):
    """``markdown.markdown`` returns an HTML string (env.py feeds the result
    straight into BeautifulSoup)."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("hello")
    assert isinstance(out, str)


def test_render_heading(depcheck):
    """``# Heading`` -> ``<h1>Heading</h1>``. CHANGELOG.md is heading-heavy; the
    in-app changelog parser depends on headings becoming <hN> tags."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("# Heading")
    assert "<h1>" in out and "Heading" in out and "</h1>" in out


def test_render_heading_levels(depcheck):
    """Multiple heading levels map to <h1>../<h3>.. — the changelog uses several
    levels to structure releases."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("# A\n\n## B\n\n### C")
    assert "<h1>" in out
    assert "<h2>" in out
    assert "<h3>" in out


def test_render_bold_and_italic(depcheck):
    """``**bold**`` -> ``<strong>``, ``*italic*`` -> ``<em>``. Inline emphasis
    must survive into the PDF/HTML."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("This is **bold** and *italic*.")
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out


def test_render_paragraph(depcheck):
    """Plain text becomes a ``<p>`` paragraph."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("Just a sentence.")
    assert "<p>" in out and "Just a sentence." in out and "</p>" in out


def test_render_unordered_list(depcheck):
    """``- item`` lines render to ``<ul><li>...`` — changelog bullet points."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("- one\n- two\n- three")
    assert "<ul>" in out
    assert out.count("<li>") == 3
    assert "</ul>" in out


def test_render_ordered_list(depcheck):
    """``1. item`` renders to ``<ol><li>...``."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("1. first\n2. second")
    assert "<ol>" in out
    assert out.count("<li>") == 2


def test_render_inline_code(depcheck):
    """Backtick spans -> ``<code>``. Chat content and changelogs use inline
    code for identifiers."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("call `do_thing()` now")
    assert "<code>do_thing()</code>" in out


def test_render_fenced_code_block(depcheck):
    """An indented code block becomes ``<pre><code>...``. (The base library
    handles indented blocks without extensions; fenced ``` blocks need the
    'fenced_code' extension, covered separately.)"""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("    x = 1\n    y = 2")
    assert "<pre>" in out and "<code>" in out


def test_render_link(depcheck):
    """``[text](url)`` -> an anchor tag with the href — changelog links."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("[Open WebUI](https://example.com)")
    assert '<a href="https://example.com">Open WebUI</a>' in out


def test_render_blockquote(depcheck):
    """``> quoted`` -> ``<blockquote>``."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("> a quote")
    assert "<blockquote>" in out and "a quote" in out


def test_render_empty_string(depcheck):
    """Rendering empty input yields an empty (or whitespace) HTML string — no
    crash on an empty CHANGELOG / empty message."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.markdown("") == ""


def test_render_changelog_like_document(depcheck):
    """End-to-end shape: a realistic CHANGELOG fragment (heading + version +
    bullet list) must render to HTML containing the headings and list items the
    in-app changelog parser then walks with BeautifulSoup."""
    mod = depcheck.load(IMPORT_NAME)
    doc = "# Changelog\n\n## [1.2.3]\n\n### Added\n\n- New feature A\n- New feature B\n"
    out = mod.markdown(doc)
    assert "<h1>" in out
    assert "<h2>" in out
    assert "<h3>" in out
    assert out.count("<li>") == 2


# --------------------------------------------------------------------------- #
# Extensions — fenced_code (core) + the pymdownx.* extensions the codebase pins
# --------------------------------------------------------------------------- #
def test_fenced_code_extension(depcheck):
    """A ``` fenced block with the built-in 'fenced_code' extension renders to
    ``<pre><code>``. Pin that core extensions still load via the
    ``extensions=[...]`` kwarg the PDF path uses."""
    mod = depcheck.load(IMPORT_NAME)
    out = mod.markdown("```\nprint('hi')\n```", extensions=["fenced_code"])
    assert "<pre>" in out and "<code>" in out and "print" in out


def test_tables_extension(depcheck):
    """The 'tables' extension (pulled in by pymdownx.extra) turns pipe tables
    into ``<table>``. Pin it loads and renders a table."""
    mod = depcheck.load(IMPORT_NAME)
    table = "| a | b |\n| - | - |\n| 1 | 2 |"
    out = mod.markdown(table, extensions=["tables"])
    assert "<table>" in out and "<td>" in out


def test_pymdownx_extensions_loadable(depcheck):
    """The codebase references ``pymdownx.extra`` (PDF path) from the pinned
    ``pymdown-extensions`` package. Pin that those extensions load through the
    ``extensions=[...]`` machinery without raising — a missing/renamed extension
    would surface as a BuildError at render time."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load(IMPORT_NAME)
    pymdownx = depcheck.try_load("pymdownx")
    if pymdownx is None:
        pytest.skip("pymdown-extensions not installed; core render is covered above")
    for ext in PYMDOWNX_EXTENSIONS:
        # Rendering with the extension enabled must not raise.
        out = mod.markdown("# probe\n\ntext", extensions=[ext])
        assert isinstance(out, str)


def test_pymdownx_extra_renders_document(depcheck):
    """The exact extension the PDF generator's comment references,
    ``pymdownx.extra``, must render a representative document (heading + bold +
    list) to HTML. This is the intended chat-content -> HTML path for export."""
    depcheck.load(IMPORT_NAME)
    mod = depcheck.load(IMPORT_NAME)
    pymdownx = depcheck.try_load("pymdownx")
    if pymdownx is None:
        pytest.skip("pymdown-extensions not installed")
    out = mod.markdown(
        "# Title\n\nSome **bold** text.\n\n- a\n- b",
        extensions=["pymdownx.extra"],
    )
    assert "<h1>" in out
    assert "<strong>bold</strong>" in out
    assert out.count("<li>") == 2


# --------------------------------------------------------------------------- #
# markdown.Markdown class — reusable converter (reset/convert)
# --------------------------------------------------------------------------- #
def test_markdown_class_convert_and_reset(depcheck):
    """``markdown.Markdown`` is the reusable converter behind the function:
    ``md.convert(text)`` then ``md.reset()`` to reuse it. Pin that surface and a
    convert round-trip (a server that renders many messages may reuse one
    instance)."""
    mod = depcheck.load(IMPORT_NAME)
    md = mod.Markdown()
    assert callable(getattr(md, "convert", None))
    assert callable(getattr(md, "reset", None))
    out = md.convert("# Reused")
    assert "<h1>" in out and "Reused" in out
    # reset() returns the instance and clears state for the next document.
    md.reset()
    out2 = md.convert("## Again")
    assert "<h2>" in out2
