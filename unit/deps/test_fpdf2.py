"""Dependency contract: fpdf2 (PyPI ``fpdf2``, import name ``fpdf``).

Open WebUI builds chat-export PDFs with fpdf2 in
``utils/pdf_generator.py::PDFGenerator.generate_chat_pdf``:

    from fpdf import FPDF
    ...
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font('NotoSans', '', f'{FONTS_DIR}/NotoSans-Regular.ttf')   # + others
    pdf.set_font('NotoSans', size=12)
    pdf.set_fallback_fonts(['NotoSansKR', 'NotoSansJP', 'NotoSansSC', 'Twemoji'])
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.write_html(self.html_body)        # HTML (built from markdown) -> PDF
    pdf_bytes = pdf.output()
    return bytes(pdf_bytes)

So the contract is: construct an ``FPDF``, add a page, register Unicode TTF
fonts (and fallbacks), set the active font, enable auto page-break, render an
HTML document via ``write_html``, and emit the document as bytes via
``output()`` (which returns a ``bytearray`` the backend wraps in ``bytes()``).

This module pins exactly that method surface + signatures and **builds a real
PDF in memory** end to end — including the ``write_html`` HTML-to-PDF path the
generator depends on — asserting the result is a valid PDF byte stream (``%PDF``
magic). To stay offline and font-file-free it uses fpdf2's built-in core fonts
(Helvetica) for the behavioural builds, and separately pins that ``add_font``
accepts the ``(family, style, fname)`` shape the backend uses for its TTFs. A
fpdf2 bump that removed/renamed any of it, or changed ``output()``'s byte
contract, fails loudly here instead of breaking chat export.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks,
plus offline behavioural PDF builds (pure CPU, no network/services). Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "fpdf"
DIST_NAME = "fpdf2"

# Every FPDF method the PDFGenerator calls.
USED_METHODS = [
    "add_page",
    "add_font",
    "set_font",
    "set_fallback_fonts",
    "set_auto_page_break",
    "write_html",
    "output",
]


def _new_pdf(mod):
    """Construct a fresh FPDF with a page and a core font set, ready to render —
    the offline (no TTF file) analogue of the generator's setup."""
    pdf = mod.FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.set_auto_page_break(auto=True, margin=15)
    return pdf


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "fpdf"


def test_version_reported(depcheck):
    """The installed distribution version (PyPI name ``fpdf2``) must be
    resolvable so bump tooling and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_fpdf_class_exists_and_callable(depcheck):
    """``from fpdf import FPDF`` then ``FPDF()`` — pin the class is present and
    constructible."""
    mod = depcheck.load(IMPORT_NAME)
    assert hasattr(mod, "FPDF")
    assert callable(mod.FPDF)


def test_used_methods_exist(depcheck):
    """Every FPDF method the generator calls must exist on the class."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.FPDF))
    missing = [m for m in USED_METHODS if m not in names]
    assert not missing, f"fpdf.FPDF missing method(s) the generator calls: {missing}"


def test_used_methods_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in USED_METHODS:
        assert callable(getattr(mod.FPDF, name)), f"FPDF.{name} not callable"


# --------------------------------------------------------------------------- #
# Signatures — the exact call shapes pdf_generator.py uses
# --------------------------------------------------------------------------- #
def test_add_font_signature(depcheck):
    """The generator calls ``pdf.add_font('NotoSans', '', '<path>.ttf')`` —
    positional (family, style, fname). Pin those parameter names."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.FPDF.add_font, ["family", "style", "fname"])


def test_set_font_accepts_family_and_size(depcheck):
    """``pdf.set_font('NotoSans', size=12)`` — family positional, size keyword.
    Pin both parameters."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.FPDF.set_font, ["family", "size"])


def test_set_auto_page_break_signature(depcheck):
    """``pdf.set_auto_page_break(auto=True, margin=15)`` — pin auto + margin."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.FPDF.set_auto_page_break, ["auto", "margin"])


def test_set_fallback_fonts_signature(depcheck):
    """``pdf.set_fallback_fonts([...])`` — first parameter is the font list."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.FPDF.set_fallback_fonts, ["fallback_fonts"])


def test_write_html_accepts_text_first(depcheck):
    """``pdf.write_html(self.html_body)`` — the first parameter is the HTML
    text. Pin it (write_html also takes **kwargs, but text must stay first)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.FPDF.write_html, ["text"])


# --------------------------------------------------------------------------- #
# Behavioural: build a real PDF in memory (the generate_chat_pdf flow)
# --------------------------------------------------------------------------- #
def test_output_returns_pdf_bytes(depcheck):
    """``pdf.output()`` must return a bytes-like object that begins with the
    ``%PDF`` magic, and ``bytes(...)`` of it (what the generator returns) is a
    valid non-empty PDF byte stream."""
    mod = depcheck.load(IMPORT_NAME)
    pdf = _new_pdf(mod)
    pdf.cell(text="hello") if "text" in _cell_params(mod) else pdf.cell(0, 10, "hello")
    out = pdf.output()
    raw = bytes(out)  # the exact bytes() hop generate_chat_pdf performs
    assert isinstance(raw, bytes)
    assert raw.startswith(b"%PDF"), f"output is not a PDF stream: {raw[:8]!r}"
    assert len(raw) > 100  # a real (if tiny) document


def test_write_html_renders_to_pdf(depcheck):
    """The headline contract: ``write_html`` turns the generator's HTML body
    into PDF content, then ``output()`` emits it. Reproduce that with an HTML
    document shaped like ``_generate_html_body`` (title heading + a message div
    with <strong> and <br/>), and assert a valid PDF results.
    """
    mod = depcheck.load(IMPORT_NAME)
    pdf = _new_pdf(mod)
    html = """
    <html><body>
      <div>
        <h2>Conversation Title</h2>
        <div>
          <h4><strong>Assistant</strong></h4>
          <div>Hello world<br/>second line</div>
        </div>
      </div>
    </body></html>
    """
    pdf.write_html(html)
    raw = bytes(pdf.output())
    assert raw.startswith(b"%PDF")
    # %%EOF is the PDF end-of-file marker — a complete document was emitted.
    assert b"%%EOF" in raw


def test_write_html_consumes_markdown_rendered_html(depcheck):
    """The generator builds its HTML from markdown (``from markdown import
    markdown``). Feed write_html the kind of HTML ``markdown`` emits (h1/p/
    strong/em/ul/li/code) and confirm it renders to a valid PDF — proving the
    markdown->HTML->PDF pipeline's final leg holds. Markdown itself is pinned in
    its own contract module; here we only need representative HTML."""
    mod = depcheck.load(IMPORT_NAME)
    markdown = depcheck.try_load("markdown")
    if markdown is not None:
        html = markdown.markdown(
            "# Heading\n\nSome **bold** and *italic* text.\n\n- one\n- two\n\n`code`"
        )
    else:
        html = "<h1>Heading</h1><p>Some <strong>bold</strong> text.</p><ul><li>one</li></ul>"
    pdf = _new_pdf(mod)
    pdf.write_html(html)
    raw = bytes(pdf.output())
    assert raw.startswith(b"%PDF")
    assert b"%%EOF" in raw


def test_multi_page_pdf_with_auto_page_break(depcheck):
    """A long chat overflows one page; ``set_auto_page_break(auto=True)`` must
    flow content onto new pages. Write enough HTML paragraphs to force >1 page
    and confirm the document records multiple page objects."""
    mod = depcheck.load(IMPORT_NAME)
    pdf = _new_pdf(mod)
    paras = "".join(
        f"<div>Paragraph number {i} with some body text.</div><br/>" for i in range(120)
    )
    pdf.write_html(f"<html><body>{paras}</body></html>")
    raw = bytes(pdf.output())
    assert raw.startswith(b"%PDF")
    # Each page is a ``/Type /Page`` object; more than one means paging worked.
    page_count = raw.count(b"/Type /Page") + raw.count(b"/Type/Page")
    assert page_count >= 2, "auto page-break did not produce multiple pages"


def test_add_page_then_output_minimal(depcheck):
    """The minimal generator skeleton: FPDF() -> add_page() -> output(). Even
    with no content this must emit a one-page PDF (guards add_page/output)."""
    mod = depcheck.load(IMPORT_NAME)
    pdf = mod.FPDF()
    pdf.add_page()
    raw = bytes(pdf.output())
    assert raw.startswith(b"%PDF")
    assert b"%%EOF" in raw


def test_set_fallback_fonts_callable_on_instance(depcheck):
    """``set_fallback_fonts([...])`` is called with a list of registered family
    names. Calling it with an empty list on a fresh instance must not raise
    (the no-fallback case); a real font list needs the TTFs, covered by the
    add_font signature test instead of a file-dependent build."""
    mod = depcheck.load(IMPORT_NAME)
    pdf = _new_pdf(mod)
    # Empty fallback list is always valid (no fonts to resolve).
    pdf.set_fallback_fonts([])
    raw = bytes(pdf.output())
    assert raw.startswith(b"%PDF")


# --------------------------------------------------------------------------- #
# Local helper (no cross-file imports — conftest exposes only fixtures).
# --------------------------------------------------------------------------- #
def _cell_params(mod):
    """Parameter names of FPDF.cell (its signature changed across 2.x: the text
    arg moved from positional ``txt`` to keyword ``text``). Used so the
    output-bytes test calls cell() in a version-compatible way."""
    import inspect

    try:
        return set(inspect.signature(mod.FPDF.cell).parameters)
    except (ValueError, TypeError):
        return set()
