"""Dependency contract: beautifulsoup4 (import name ``bs4``).

Open WebUI uses BeautifulSoup to turn HTML into structured data / plain text
in two places, and the contract this module pins is exactly what those two
consumers rely on:

1. ``backend/open_webui/env.py`` — at import time it renders ``CHANGELOG.md``
   to HTML (via ``markdown``) and parses it to build ``changelog_json``::

       soup = BeautifulSoup(html_content, "html.parser")
       for version in soup.find_all("h2"):
           ... version.get_text().strip().split(" - ") ...
           current = version.find_next_sibling()
           while current and current.name != "h2":
               if current.name == "h3":
                   section_items = parse_section(current.find_next_sibling("ul"))
               current = current.find_next_sibling()

   and ``parse_section`` does::

       for li in section.find_all("li"):
           raw_html = str(li)
           text = li.get_text(separator=" ", strip=True)

   So the env.py contract is: the ``"html.parser"`` backend, ``find_all(tag)``,
   ``find_next_sibling()`` / ``find_next_sibling(tag)``, the ``.name``
   attribute, ``get_text()``, ``get_text(separator=..., strip=...)`` and
   ``str(tag)`` round-tripping a tag back to HTML. A regression here breaks
   *module import* of open_webui.env, i.e. the whole backend won't start.

2. ``backend/open_webui/retrieval/web/utils.py`` — ``SafeWebBaseLoader``
   (a subclass of langchain's ``WebBaseLoader``) builds soups from fetched web
   pages and extracts text + metadata::

       final_results.append(BeautifulSoup(result, parser, **self.bs_kwargs))
       text = soup.get_text(**self.bs_get_text_kwargs)
       if title := soup.find("title"):
           metadata["title"] = title.get_text()
       if description := soup.find("meta", attrs={"name": "description"}):
           metadata["description"] = description.get("content", "No description found.")
       if html := soup.find("html"):
           metadata["language"] = html.get("lang", "No language found.")

   with ``parser`` chosen as ``"xml"`` for ``*.xml`` URLs else the loader's
   ``default_parser`` (``"html.parser"``). So the retrieval contract adds:
   ``find(tag)``, ``find(tag, attrs={...})``, ``Tag.get(key, default)``,
   ``find(...)`` returning a falsy ``None`` for a missing tag (the ``:=`` /
   ``if`` guards depend on that), and the ``"xml"`` parser for XML/sitemaps.

This file therefore pins both the *symbol surface* (constructor, the
``Tag``/``NavigableString`` classes, the methods/attributes above) and the
*offline behavioural contracts* on in-memory HTML/XML strings — no network,
no filesystem. The ``"html.parser"`` backend is stdlib and always present, so
its behaviour is asserted unconditionally; ``lxml``/``html5lib`` are probed
only when importable (the loader can be configured to use them, and
``"xml"`` requires lxml). A beautifulsoup4 major bump that drops/renames any
of this, or quietly changes extraction behaviour (whitespace handling,
missing-tag falsiness, ``.get`` default semantics), fails loudly here instead
of surfacing as a backend that won't import or a garbled RAG ingestion.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface) +
offline behavioural contracts (no network). Uses the ``depcheck`` fixture
from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "bs4"
DIST_NAME = "beautifulsoup4"

# Symbols the Open WebUI backend references on ``bs4`` (directly, or as part of
# the small public surface its two consumers depend on). ``BeautifulSoup`` is
# the only name actually imported; ``Tag``/``NavigableString`` are the result
# types whose methods/attributes the code drives, pinned so a major rewrite of
# the class layout is caught.
USED_SYMBOLS = [
    "BeautifulSoup",
    "Tag",
    "NavigableString",
]

# Parser backends. "html.parser" is stdlib (always available) and is what both
# consumers pass explicitly / via the loader default. "lxml" + its "xml" mode
# and "html5lib" are optional alternates the loader can be configured to use;
# "xml" specifically is used for *.xml URLs and requires lxml.
STDLIB_PARSER = "html.parser"
OPTIONAL_TREE_BUILDERS = {
    "lxml": "lxml",  # bs4 feature name -> import name to probe
    "html5lib": "html5lib",
}


# --------------------------------------------------------------------------- #
# Local, offline HTML/XML fixtures (deterministic; no network, no files).
# --------------------------------------------------------------------------- #

# Mirrors the shape ``markdown.markdown(CHANGELOG.md)`` produces and that
# env.py walks: <h2> version headers, <h3> section titles, <ul><li> items.
_CHANGELOG_HTML = """
<h2>[0.1.0] - 2024-01-01</h2>
<h3>Added</h3>
<ul>
  <li>Feature one: did a thing.</li>
  <li>Feature two: did another thing.</li>
</ul>
<h3>Fixed</h3>
<ul>
  <li>Bug: squashed it.</li>
</ul>
<h2>[0.0.9] - 2023-12-31</h2>
<h3>Added</h3>
<ul>
  <li>Initial release.</li>
</ul>
"""

# Mirrors a fetched web page the SafeWebBaseLoader extracts text + metadata from.
_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <title>Example Page Title</title>
    <meta name="description" content="A short page description.">
    <meta name="keywords" content="ignored">
  </head>
  <body>
    <h1>Heading</h1>
    <p>First paragraph of body text.</p>
    <p>Second paragraph.</p>
  </body>
</html>
"""

# A page deliberately missing <title>/<meta description>/lang so the loader's
# `if title := soup.find(...)` guards take the else branch.
_PAGE_HTML_NO_META = """
<html>
  <body><p>No head, no title, no meta, no lang.</p></body>
</html>
"""

# XML / sitemap-style content: the loader uses the "xml" parser for *.xml URLs.
_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/a</loc></url>
  <url><loc>https://example.com/b</loc></url>
</urlset>
"""


def _bs(depcheck):
    """Load bs4 (or skip) and return the BeautifulSoup class."""
    mod = depcheck.load(IMPORT_NAME)
    return mod, mod.BeautifulSoup


def _available_parsers(depcheck):
    """The parser feature-names usable in this env: stdlib + importable optionals."""
    parsers = [STDLIB_PARSER]
    for feature, import_name in OPTIONAL_TREE_BUILDERS.items():
        if depcheck.try_load(import_name) is not None:
            parsers.append(feature)
    return parsers


# --------------------------------------------------------------------------- #
# Import / symbol-surface tests.
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    """bs4 must import and identify as the bs4 package."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "bs4"


def test_used_symbols_exist(depcheck):
    """Every bs4 symbol the codebase relies on must still resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_beautifulsoup_is_class(depcheck):
    """``from bs4 import BeautifulSoup`` must give a constructible class."""
    _mod, BeautifulSoup = _bs(depcheck)
    assert inspect.isclass(BeautifulSoup), "BeautifulSoup is not a class"


def test_tag_and_navigablestring_are_classes(depcheck):
    """Tag/NavigableString (the result element types) remain classes."""
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Tag), "bs4.Tag is not a class"
    assert inspect.isclass(mod.NavigableString), "bs4.NavigableString is not a class"


def test_version_attribute_is_string(depcheck):
    """bs4.__version__ is a non-empty string."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str) and mod.__version__


def test_dist_version_resolvable(depcheck):
    """The installed distribution (beautifulsoup4) version is resolvable."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_constructor_accepts_markup_and_features_positionally(depcheck):
    """The code calls ``BeautifulSoup(markup, parser)`` positionally.

    Pin that the constructor still takes the markup as the first positional
    arg and a parser/features as the second (any name; **kwargs covers the
    loader's ``**bs_kwargs``).
    """
    _mod, BeautifulSoup = _bs(depcheck)
    try:
        sig = inspect.signature(BeautifulSoup)
    except (TypeError, ValueError):
        pytest.skip("BeautifulSoup has no introspectable signature")
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) >= 2, (
        f"BeautifulSoup must accept (markup, features) positionally (sig: {sig})"
    )
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    assert has_var_kw, f"BeautifulSoup must accept **kwargs (loader passes **bs_kwargs); sig: {sig}"


# --------------------------------------------------------------------------- #
# Soup-instance method/attribute surface (driven on a real parse).
# --------------------------------------------------------------------------- #


def test_soup_instance_method_surface(depcheck):
    """A parsed soup exposes the methods/attributes both consumers call."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    for name in ("find", "find_all", "get_text", "get"):
        assert callable(getattr(soup, name, None)), f"soup.{name} missing/not callable"


def test_tag_instance_method_surface(depcheck):
    """A Tag exposes find_next_sibling/get_text/get and a .name attribute."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    h2 = soup.find("h2")
    assert h2 is not None
    for name in ("find_next_sibling", "get_text", "find", "find_all", "get"):
        assert callable(getattr(h2, name, None)), f"Tag.{name} missing/not callable"
    # .name is the tag-name attribute env.py compares against ("h2"/"h3").
    assert h2.name == "h2", f"Tag.name expected 'h2', got {h2.name!r}"


# --------------------------------------------------------------------------- #
# Behavioural contract: env.py CHANGELOG parsing (html.parser).
# This path runs at *import time*, so a regression breaks backend startup.
# --------------------------------------------------------------------------- #


def test_find_all_returns_tags_in_document_order(depcheck):
    """``soup.find_all("h2")`` returns every <h2>, in document order.

    env.py iterates these to build one changelog entry per version.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    h2s = soup.find_all("h2")
    assert len(h2s) == 2, f"expected 2 <h2>, got {len(h2s)}"
    texts = [h.get_text() for h in h2s]
    assert "[0.1.0] - 2024-01-01" in texts[0]
    assert "[0.0.9] - 2023-12-31" in texts[1]


def test_get_text_strip_split_matches_env_parsing(depcheck):
    """Reproduce env.py's exact version-header parsing on a header tag.

    ``version.get_text().strip().split(" - ")`` must yield
    ``["[x.y.z]", "date"]`` so ``[0][1:-1]`` (strip brackets) and ``[1]``
    (date) work.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    h2 = soup.find("h2")
    parts = h2.get_text().strip().split(" - ")
    assert len(parts) == 2, f"version header did not split into 2 parts: {parts!r}"
    version_number = parts[0][1:-1]  # strip surrounding [ ]
    assert version_number == "0.1.0", f"version number parsed as {version_number!r}"
    assert parts[1] == "2024-01-01", f"date parsed as {parts[1]!r}"


def test_find_next_sibling_walks_section_tags(depcheck):
    """``find_next_sibling()`` / ``find_next_sibling(name)`` drive env.py's walk.

    From an <h2>, the next element sibling is the first <h3>; from that <h3>,
    ``find_next_sibling("ul")`` returns its item list. The walk stops at the
    next <h2>.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    h2 = soup.find("h2")

    # No-arg find_next_sibling skips whitespace NavigableStrings and lands on a Tag.
    current = h2.find_next_sibling()
    assert current is not None and current.name == "h3", (
        f"first sibling after <h2> expected <h3>, got {current!r}"
    )
    assert current.get_text().lower() == "added"

    # find_next_sibling("ul") returns the <ul> following the <h3>.
    ul = current.find_next_sibling("ul")
    assert ul is not None and ul.name == "ul", f"expected <ul> after <h3>, got {ul!r}"

    # Walk to the second section (<h3>Fixed) then confirm we eventually hit <h2>.
    saw_fixed = False
    while current and current.name != "h2":
        if current.name == "h3" and current.get_text().lower() == "fixed":
            saw_fixed = True
        current = current.find_next_sibling()
    assert saw_fixed, "walk did not encounter the 'Fixed' <h3> section"
    assert current is not None and current.name == "h2", (
        "walk did not terminate on the next <h2> version header"
    )


def test_parse_section_extraction_contract(depcheck):
    """Reproduce env.py ``parse_section``: list items -> {title, content, raw}.

    Pins ``find_all("li")``, ``str(li)`` (tag -> HTML round-trip) and
    ``li.get_text(separator=" ", strip=True)`` plus the ``": "`` split.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    first_ul = soup.find("ul")
    assert first_ul is not None

    items = []
    for li in first_ul.find_all("li"):
        raw_html = str(li)
        text = li.get_text(separator=" ", strip=True)
        parts = text.split(": ", 1)
        title = parts[0].strip() if len(parts) > 1 else ""
        content = parts[1].strip() if len(parts) > 1 else text
        items.append({"title": title, "content": content, "raw": raw_html})

    assert len(items) == 2, f"expected 2 <li> items, got {len(items)}"
    assert items[0]["title"] == "Feature one"
    assert items[0]["content"] == "did a thing."
    # str(li) must round-trip the element back into an <li>...</li> HTML string.
    assert items[0]["raw"].startswith("<li") and items[0]["raw"].endswith("</li>")


def test_get_text_separator_and_strip(depcheck):
    """``get_text(separator=" ", strip=True)`` joins descendant text with the
    separator and trims edges — the exact call parse_section relies on."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup("<li>  <b>Bold</b> and <i>italic</i>  </li>", STDLIB_PARSER)
    li = soup.find("li")
    text = li.get_text(separator=" ", strip=True)
    assert text == "Bold and italic", f"separator/strip extraction got {text!r}"


def test_get_text_default_no_separator(depcheck):
    """Plain ``get_text()`` concatenates descendant strings (no inserted sep)."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup("<h2>[0.1.0]</h2>", STDLIB_PARSER)
    assert soup.find("h2").get_text() == "[0.1.0]"


# --------------------------------------------------------------------------- #
# Behavioural contract: SafeWebBaseLoader page text + metadata (html.parser).
# --------------------------------------------------------------------------- #


def test_soup_get_text_extracts_body_text(depcheck):
    """``soup.get_text()`` over a full page yields the human-readable text and
    drops tags — the loader's ``page_content`` source."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    text = soup.get_text()
    assert "First paragraph of body text." in text
    assert "Second paragraph." in text
    assert "<p>" not in text and "<html" not in text, "tags leaked into get_text()"


def test_find_title_and_get_text(depcheck):
    """``soup.find("title").get_text()`` -> the page title (metadata['title'])."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    title = soup.find("title")
    assert title is not None, "find('title') returned None on a page with a <title>"
    assert title.get_text() == "Example Page Title"


def test_find_meta_with_attrs_and_get_default(depcheck):
    """``find("meta", attrs={"name": "description"})`` selects the right <meta>,
    and ``.get("content", default)`` reads its attribute.

    This is the loader's metadata['description'] extraction verbatim,
    including that the right meta is chosen among several.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    description = soup.find("meta", attrs={"name": "description"})
    assert description is not None, "attrs-filtered find('meta') returned None"
    assert description.get("content", "No description found.") == "A short page description."


def test_find_html_lang_attribute_via_get(depcheck):
    """``soup.find("html").get("lang", default)`` -> the document language."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    html = soup.find("html")
    assert html is not None
    assert html.get("lang", "No language found.") == "en"


def test_tag_get_returns_default_for_missing_attr(depcheck):
    """``Tag.get(key, default)`` returns the default when the attr is absent —
    the loader depends on this for its 'No ... found.' fallbacks."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup("<html><body>x</body></html>", STDLIB_PARSER)
    html = soup.find("html")
    assert html is not None
    assert html.get("lang", "No language found.") == "No language found."
    # And get() with no default yields None (never raises) for a missing attr.
    assert html.get("lang") is None


def test_find_missing_tag_is_falsy_none(depcheck):
    """``soup.find(tag)`` returns None (falsy) when absent.

    The loader guards every metadata read with ``if x := soup.find(...)``;
    that idiom is only correct if a miss is falsy. Pin it.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML_NO_META, STDLIB_PARSER)
    assert soup.find("title") is None
    assert soup.find("meta", attrs={"name": "description"}) is None
    # The walrus-guarded branch must simply not execute; emulate it:
    metadata = {"source": "x"}
    if title := soup.find("title"):
        metadata["title"] = title.get_text()
    if description := soup.find("meta", attrs={"name": "description"}):
        metadata["description"] = description.get("content", "No description found.")
    assert "title" not in metadata and "description" not in metadata


def test_found_tag_is_truthy(depcheck):
    """A present tag is truthy so the walrus guard executes its branch."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_PAGE_HTML, STDLIB_PARSER)
    title = soup.find("title")
    assert bool(title) is True, "a found tag must be truthy for the loader's `if` guards"


# --------------------------------------------------------------------------- #
# Behavioural contract: the "xml" parser for *.xml URLs (requires lxml).
# --------------------------------------------------------------------------- #


def test_xml_parser_parses_sitemap_when_available(depcheck):
    """``BeautifulSoup(xml, "xml")`` parses sitemap-style XML.

    The loader selects ``parser="xml"`` for URLs ending in ``.xml``; that mode
    needs lxml. Skip cleanly if lxml is absent, else assert tag extraction.
    """
    if depcheck.try_load("lxml") is None:
        pytest.skip("lxml not importable; bs4 'xml' parser unavailable")
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup(_SITEMAP_XML, "xml")
    locs = soup.find_all("loc")
    assert len(locs) == 2, f"expected 2 <loc> in sitemap, got {len(locs)}"
    assert locs[0].get_text() == "https://example.com/a"
    assert locs[1].get_text() == "https://example.com/b"


# --------------------------------------------------------------------------- #
# Cross-parser behavioural contract: the loader's default_parser is
# "html.parser" but it can be configured to use lxml/html5lib. The extraction
# the loader does must behave identically across whichever HTML parser is in
# play, so run the core contract against every importable HTML parser.
# --------------------------------------------------------------------------- #


def test_core_extraction_consistent_across_html_parsers(depcheck):
    """find/find_all/get_text/get behave the same on every available HTML parser.

    Pins that swapping WEB_LOADER's parser (html.parser <-> lxml <-> html5lib)
    does not change the metadata/text the loader extracts.
    """
    _mod, BeautifulSoup = _bs(depcheck)
    parsers = _available_parsers(depcheck)
    for parser in parsers:
        soup = BeautifulSoup(_PAGE_HTML, parser)

        title = soup.find("title")
        assert title is not None, f"[{parser}] find('title') was None"
        assert title.get_text() == "Example Page Title", f"[{parser}] wrong title"

        desc = soup.find("meta", attrs={"name": "description"})
        assert desc is not None, f"[{parser}] find('meta', attrs=...) was None"
        assert desc.get("content", "x") == "A short page description.", (
            f"[{parser}] wrong meta description"
        )

        html = soup.find("html")
        assert html is not None and html.get("lang", "x") == "en", f"[{parser}] wrong html lang"

        text = soup.get_text()
        assert "First paragraph of body text." in text, f"[{parser}] body text missing"

        assert len(soup.find_all("p")) == 2, f"[{parser}] expected 2 <p>"


def test_html_parser_always_available(depcheck):
    """The stdlib 'html.parser' backend must always work (no optional dep)."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup("<p>hi</p>", STDLIB_PARSER)
    assert soup.get_text() == "hi"


# --------------------------------------------------------------------------- #
# Determinism / robustness.
# --------------------------------------------------------------------------- #


def test_parsing_is_deterministic_across_calls(depcheck):
    """Parsing identical markup twice yields identical extracted text —
    the loaders assume a stable result for a given input."""
    _mod, BeautifulSoup = _bs(depcheck)
    a = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    b = BeautifulSoup(_CHANGELOG_HTML, STDLIB_PARSER)
    assert [h.get_text() for h in a.find_all("h2")] == [h.get_text() for h in b.find_all("h2")]
    assert a.get_text() == b.get_text()


def test_get_text_on_empty_and_textless_markup(depcheck):
    """get_text() never raises and returns '' for empty / tag-only markup —
    the loader has no guard around ``soup.get_text(...)``."""
    _mod, BeautifulSoup = _bs(depcheck)
    assert BeautifulSoup("", STDLIB_PARSER).get_text() == ""
    assert BeautifulSoup("<br/>", STDLIB_PARSER).get_text() == ""


def test_find_all_empty_when_no_match(depcheck):
    """find_all returns an empty (iterable) list when nothing matches —
    env.py iterates the result directly, so a non-list/None would break it."""
    _mod, BeautifulSoup = _bs(depcheck)
    soup = BeautifulSoup("<p>no headers here</p>", STDLIB_PARSER)
    result = soup.find_all("h2")
    assert list(result) == [], f"find_all with no match returned {result!r}"
