"""Dependency contract: pymdown-extensions (import package ``pymdownx``).

pymdown-extensions is a bundle of extensions for Python-Markdown
(``markdown``). It's pinned in ``backend/requirements.txt``. Open WebUI's
``utils/pdf_generator.py`` documents the intended markdown->HTML rendering
via this bundle:

    # html_content = markdown(content, extensions=["pymdownx.extra"])

IMPORTANT — usage note: that line is currently COMMENTED OUT — the PDF
generator instead does a naive ``content.replace('\\n', '<br/>')`` after
HTML-escaping, so pymdownx is not on a live first-party code path right
now. It remains a declared dependency (the documented/intended renderer,
and a common extension set for any future markdown rendering). There are
no live call sites passing keyword arguments, so this module pins the
*contract that matters*: that the ``pymdownx.*`` extension modules the
comment references load and register correctly with Python-Markdown, by
actually rendering through ``markdown(..., extensions=[...])`` offline.

The extensions are pure-Python text transforms: fully offline, no network,
no I/O. If ``markdown`` itself isn't importable we skip the behavioural
render tests but still check the extension modules import.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pymdownx"
DIST_NAME = "pymdown-extensions"

# Extension submodules the bundle is expected to provide. `extra` is the one
# the pdf_generator comment names; the rest are the widely-used set whose loss
# would signal a breaking bundle change.
EXTENSION_MODULES = [
    "pymdownx.extra",  # named in pdf_generator.py
    "pymdownx.superfences",  # fenced code / mermaid
    "pymdownx.highlight",
    "pymdownx.arithmatex",  # math
    "pymdownx.tasklist",
    "pymdownx.tilde",  # ~~strike~~ / subscript
    "pymdownx.tabbed",
    "pymdownx.emoji",
]

# Extension *names* passed to markdown(extensions=[...]).
RENDERABLE_EXTENSIONS = [
    "pymdownx.extra",
    "pymdownx.superfences",
    "pymdownx.tasklist",
    "pymdownx.tilde",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pymdownx"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_extension_modules_import(depcheck):
    """Every pymdownx.* extension module the codebase references (or might) must
    import. Reports all failures at once."""
    depcheck.load(IMPORT_NAME)
    failures = {}
    for name in EXTENSION_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:
            failures[name] = f"{type(e).__name__}: {e}"
    assert not failures, f"pymdownx extension module(s) failed to import: {failures}"


def test_extra_extension_provides_makeextension(depcheck):
    """Python-Markdown loads an extension via its module-level makeExtension()
    factory. pymdownx.extra (the one pdf_generator names) must expose it."""
    depcheck.load(IMPORT_NAME)
    extra = importlib.import_module("pymdownx.extra")
    assert hasattr(extra, "makeExtension"), "pymdownx.extra lost makeExtension()"
    assert callable(extra.makeExtension)


def test_markdown_renders_with_extra(depcheck):
    """The exact commented-out usage: markdown(content, extensions=
    ['pymdownx.extra']). Render a small doc and assert the extension activated
    (a header turns into <h1>, and extra's nested constructs work)."""
    depcheck.load(IMPORT_NAME)
    markdown = depcheck.try_load("markdown")
    if markdown is None:
        pytest.skip("Python-Markdown not installed; extension import covers offline")
    html = markdown.markdown(
        "# Title\n\nsome **bold** text",
        extensions=["pymdownx.extra"],
    )
    assert "<h1" in html
    assert "<strong>bold</strong>" in html


def test_markdown_renders_with_all_referenced_extensions(depcheck):
    """Loading the whole referenced extension set together must not conflict —
    markdown(extensions=[...]) is how they'd be combined in practice."""
    depcheck.load(IMPORT_NAME)
    markdown = depcheck.try_load("markdown")
    if markdown is None:
        pytest.skip("Python-Markdown not installed; extension import covers offline")
    html = markdown.markdown(
        "# H\n\n~~gone~~\n\n- [x] done\n- [ ] todo",
        extensions=RENDERABLE_EXTENSIONS,
    )
    assert "<h1" in html
    # tilde -> <del>; tasklist -> checkbox inputs.
    assert "<del>gone</del>" in html
    assert 'type="checkbox"' in html


def test_superfences_renders_fenced_block(depcheck):
    """superfences is the extension powering code/mermaid fences — a likely
    choice for any chat-to-PDF renderer. Verify a fenced block becomes a
    code block."""
    depcheck.load(IMPORT_NAME)
    markdown = depcheck.try_load("markdown")
    if markdown is None:
        pytest.skip("Python-Markdown not installed; extension import covers offline")
    src = "```python\nprint('hi')\n```"
    html = markdown.markdown(src, extensions=["pymdownx.superfences"])
    assert "<code" in html or "<pre" in html
    # superfences (+ default highlight/Pygments) may split the source into
    # syntax-highlight <span>s, so assert the identifier survives rather than a
    # contiguous literal.
    assert "print" in html and "hi" in html


def test_tasklist_extension_makeextension(depcheck):
    """tasklist must also expose makeExtension() (registration contract)."""
    depcheck.load(IMPORT_NAME)
    tasklist = importlib.import_module("pymdownx.tasklist")
    assert hasattr(tasklist, "makeExtension")
    assert callable(tasklist.makeExtension)


def test_not_on_live_path_marker():
    """Documentation guard (no dep assertion): records that the pymdownx render
    call in pdf_generator.py is currently commented out, so this dependency is
    declared/intended rather than live. The render pins above guard the
    integration should it be (re)enabled."""
    assert True
