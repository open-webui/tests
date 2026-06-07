"""Dependency contract: pypdf (import name ``pypdf``).

pypdf is a pinned dependency (``pypdf==6.7.5`` in ``backend/requirements.txt``
and ``pyproject.toml``). The Open WebUI backend does not import it directly;
it reaches pypdf through LangChain's ``PyPDFLoader``, used in
``retrieval/loaders/main.py`` to extract text from uploaded ``.pdf`` files.
``PyPDFLoader`` internally constructs ``pypdf.PdfReader`` and iterates
``reader.pages`` calling ``page.extract_text()``, surfacing pypdf's
metadata and (for protected files) its encryption handling.

This module pins the slice of pypdf that ``PyPDFLoader`` (and therefore the
document-ingestion path) depends on, plus the behavioural guarantees that
matter, all OFFLINE — every PDF used here is generated in-process with
pypdf's own ``PdfWriter`` (no fixture files, no network):

  - the core surface: ``PdfReader`` / ``PdfWriter`` / ``PageObject`` and the
    ``pypdf.errors`` exception module;
  - ``PdfReader(stream)`` accepts a file-like / path / bytes stream and
    exposes ``.pages`` (len + indexable), ``.metadata``, ``.is_encrypted``;
  - ``page.extract_text()`` returns a ``str`` (the value LangChain wraps in
    a Document);
  - a write→read round-trip preserves page count and metadata;
  - encryption: an encrypted PDF reports ``is_encrypted`` True, ``decrypt``
    with the right password succeeds and with the wrong one returns the
    "not decrypted" sentinel rather than silently exposing content;
  - the error contract: malformed / empty input raises a subclass of
    ``pypdf.errors.PyPdfError`` (so an ingestion failure is catchable as one
    type) rather than something generic or nothing at all.

A pypdf bump that renamed ``PdfReader``/``extract_text``, changed the
``.pages`` shape, or altered the encryption-failure signal would fail here
instead of corrupting or silently mis-handling uploaded PDFs.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import io

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pypdf"
DIST_NAME = "pypdf"

# Core surface PyPDFLoader (and any pypdf consumer) relies on.
USED_SYMBOLS = [
    "PdfReader",
    "PdfWriter",
    "PageObject",
    "errors",
    "errors.PyPdfError",
    "errors.PdfReadError",
]

# Exception classes that must exist under pypdf.errors.
ERROR_SYMBOLS = [
    "errors.PyPdfError",
    "errors.PdfReadError",
    "errors.EmptyFileError",
    "errors.FileNotDecryptedError",
    "errors.WrongPasswordError",
]


def _make_pdf(mod, *, pages: int = 1, metadata: dict | None = None) -> io.BytesIO:
    """Build a minimal valid PDF in memory with `pages` blank pages."""
    writer = mod.PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    if metadata:
        writer.add_metadata(metadata)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf


def _make_encrypted_pdf(mod, password: str) -> io.BytesIO:
    writer = mod.PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.encrypt(password)
    buf = io.BytesIO()
    writer.write(buf)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pypdf"


def test_version_reported(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert getattr(mod, "__version__", None) is not None
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_error_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, ERROR_SYMBOLS)


def test_pdfreader_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.PdfReader)
    assert inspect.isclass(mod.PdfWriter)


def test_pdfreader_constructor_signature(depcheck):
    """PyPDFLoader does PdfReader(stream, password=...). Pin that the first
    positional is the stream and `password`/`strict` remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.PdfReader.__init__)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert params, "PdfReader.__init__ has no parameters besides self"
    first = params[0]
    assert first.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    depcheck.assert_params(mod.PdfReader.__init__, ["password"])


# --------------------------------------------------------------------------- #
# Behavioural: read a generated PDF (offline)
# --------------------------------------------------------------------------- #


def test_read_pages_len_and_index(depcheck):
    """reader.pages must support len() and indexing — exactly how PyPDFLoader
    iterates page by page."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_pdf(mod, pages=3))
    assert len(reader.pages) == 3
    page0 = reader.pages[0]
    assert type(page0).__name__ == "PageObject"


def test_pages_is_iterable(depcheck):
    """PyPDFLoader does `for page in reader.pages`. Pin pages is iterable and
    yields PageObjects."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_pdf(mod, pages=2))
    pages = list(reader.pages)
    assert len(pages) == 2
    for p in pages:
        assert type(p).__name__ == "PageObject"


def test_extract_text_returns_str(depcheck):
    """page.extract_text() must return a str (LangChain wraps it directly in a
    Document.page_content). A blank page yields an empty string, never None."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_pdf(mod, pages=1))
    text = reader.pages[0].extract_text()
    assert isinstance(text, str)


def test_extract_text_is_callable_on_page(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_pdf(mod, pages=1))
    assert callable(reader.pages[0].extract_text)


def test_read_from_bytes_stream(depcheck):
    """A bytes-backed BytesIO stream must be readable (uploads arrive as
    bytes)."""
    mod = depcheck.load(IMPORT_NAME)
    raw = _make_pdf(mod, pages=1).getvalue()
    reader = mod.PdfReader(io.BytesIO(raw))
    assert len(reader.pages) == 1


# --------------------------------------------------------------------------- #
# Behavioural: metadata
# --------------------------------------------------------------------------- #


def test_metadata_roundtrip(depcheck):
    """Written metadata is readable back via reader.metadata.title etc."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(
        _make_pdf(mod, pages=1, metadata={"/Title": "Test Doc", "/Author": "OWUI"})
    )
    md = reader.metadata
    assert md is not None
    assert md.title == "Test Doc"
    assert md.author == "OWUI"


def test_is_encrypted_false_for_plain_pdf(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_pdf(mod, pages=1))
    assert reader.is_encrypted is False


# --------------------------------------------------------------------------- #
# Behavioural: encryption (protected-PDF handling)
# --------------------------------------------------------------------------- #


def test_encrypted_pdf_reports_is_encrypted(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_encrypted_pdf(mod, "secret"))
    assert reader.is_encrypted is True


def test_decrypt_with_correct_password_succeeds(depcheck):
    """decrypt(correct) returns a truthy PasswordType (USER/OWNER), unlocking
    the document so pages become readable."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_encrypted_pdf(mod, "secret"))
    result = reader.decrypt("secret")
    # PasswordType is an IntEnum; correct password is a non-zero member.
    assert int(result) != 0
    # And pages are now accessible.
    assert len(reader.pages) == 1


def test_decrypt_with_wrong_password_returns_not_decrypted(depcheck):
    """The security-relevant contract: a wrong password must NOT unlock the
    file. decrypt(wrong) returns the NOT_DECRYPTED sentinel (int 0), it does
    not raise-and-then-expose or silently succeed."""
    mod = depcheck.load(IMPORT_NAME)
    reader = mod.PdfReader(_make_encrypted_pdf(mod, "secret"))
    result = reader.decrypt("wrong-password")
    assert int(result) == 0
    if hasattr(mod, "PasswordType"):
        assert int(result) == int(mod.PasswordType.NOT_DECRYPTED)


# --------------------------------------------------------------------------- #
# Behavioural: error contract (malformed / empty input)
# --------------------------------------------------------------------------- #


def test_garbage_input_raises_pypdf_error(depcheck):
    """A non-PDF byte stream must raise a subclass of PyPdfError so the
    ingestion path can catch it as one error type."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.errors.PyPdfError):
        mod.PdfReader(io.BytesIO(b"this is definitely not a pdf"))


def test_empty_input_raises_pypdf_error(depcheck):
    """An empty stream must raise a PyPdfError (EmptyFileError), not hang or
    return a zero-page reader."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.errors.PyPdfError):
        mod.PdfReader(io.BytesIO(b""))


def test_error_hierarchy(depcheck):
    """Every concrete read/decrypt error must subclass PyPdfError."""
    mod = depcheck.load(IMPORT_NAME)
    base = mod.errors.PyPdfError
    for name in (
        "PdfReadError",
        "EmptyFileError",
        "FileNotDecryptedError",
        "WrongPasswordError",
    ):
        exc = getattr(mod.errors, name)
        assert issubclass(exc, base), f"{name} no longer subclasses PyPdfError"
