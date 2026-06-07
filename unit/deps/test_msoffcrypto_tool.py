"""Dependency contract: msoffcrypto-tool (import name ``msoffcrypto``).

msoffcrypto-tool decrypts password-protected Microsoft Office files (the
OOXML / legacy OLE "encrypted document" container). Open WebUI pins it in
``backend/requirements.txt`` (``msoffcrypto-tool==6.0.0``) but does NOT import
it directly in ``open_webui/*``: it is a *transitive* dependency of the
document-ingestion stack (pandas / openpyxl / the unstructured + langchain
loaders used by ``retrieval/loaders/main.py``), which reach for it when an
uploaded Office file is encrypted so the spreadsheet/doc can be opened and
indexed.

Because nothing in the backend calls msoffcrypto by name, this module pins its
*core public surface* — the surface the loader stack depends on transitively —
so a bump that broke it surfaces here rather than as an opaque "can't open
this xlsx/docx" failure during ingestion. We pin: the top-level ``OfficeFile``
dispatcher (the documented entry point), the per-format API it returns
(``is_encrypted`` / ``load_key`` / ``decrypt`` with their parameter names), and
the ``msoffcrypto.exceptions`` hierarchy (notably ``FileFormatError`` and
``InvalidKeyError`` that callers distinguish). One real offline behavioural
contract confirms ``OfficeFile`` over a non-Office byte stream raises
``FileFormatError`` (no real document, no network).

Pattern mirrors test_requests.py: symbol/signature checks plus a small offline
behavioural assertion. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "msoffcrypto"
DIST_NAME = "msoffcrypto-tool"

# Format-class methods callers (pandas/openpyxl decrypt helpers) drive once
# OfficeFile() has dispatched to the concrete format handler.
FORMAT_METHODS = ["is_encrypted", "load_key", "decrypt"]

# The exception classes msoffcrypto exposes; FileFormatError / InvalidKeyError
# are the ones decrypt helpers branch on.
EXCEPTION_NAMES = [
    "FileFormatError",
    "InvalidKeyError",
    "DecryptionError",
    "ParseError",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`msoffcrypto` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "msoffcrypto"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test. NOTE the distribution name is
    `msoffcrypto-tool` while the import name is `msoffcrypto`."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface the loader stack depends on)
# ---------------------------------------------------------------------------


def test_office_file_exists_and_callable(depcheck):
    """OfficeFile is the documented entry point: OfficeFile(file_obj) sniffs the
    container and returns a format-specific handler. Must be callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "OfficeFile")


def test_exceptions_module_exists(depcheck):
    """msoffcrypto.exceptions holds the error hierarchy callers catch."""
    depcheck.load(IMPORT_NAME)
    ex = depcheck.load("msoffcrypto.exceptions")
    depcheck.assert_symbols(ex, EXCEPTION_NAMES)


def test_format_handlers_importable(depcheck):
    """The OOXML format handler (the modern xlsx/docx/pptx path) must remain
    importable and expose the decrypt API; this is the class OfficeFile()
    dispatches to for modern Office documents."""
    depcheck.load(IMPORT_NAME)
    ooxml = depcheck.load("msoffcrypto.format.ooxml")
    assert hasattr(ooxml, "OOXMLFile"), "msoffcrypto.format.ooxml.OOXMLFile is gone"
    handler = ooxml.OOXMLFile
    missing = [m for m in FORMAT_METHODS if not hasattr(handler, m)]
    assert not missing, f"OOXMLFile missing method(s): {missing}"
    for m in FORMAT_METHODS:
        assert callable(getattr(handler, m)), f"OOXMLFile.{m} is not callable"


# ---------------------------------------------------------------------------
# Signature contracts on the format handler (the decrypt workflow)
# ---------------------------------------------------------------------------


def test_load_key_accepts_password(depcheck):
    """The decrypt workflow is load_key(password=...) then decrypt(outfile).
    `password` must remain an accepted parameter on load_key."""
    depcheck.load(IMPORT_NAME)
    ooxml = depcheck.load("msoffcrypto.format.ooxml")
    depcheck.assert_params(ooxml.OOXMLFile.load_key, ["password"])


def test_decrypt_accepts_outfile(depcheck):
    """decrypt(outfile) writes the plaintext document to a file-like object;
    `outfile` must remain accepted."""
    depcheck.load(IMPORT_NAME)
    ooxml = depcheck.load("msoffcrypto.format.ooxml")
    depcheck.assert_params(ooxml.OOXMLFile.decrypt, ["outfile"])


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


def test_exceptions_are_exception_subclasses(depcheck):
    """Every msoffcrypto error must subclass Exception so callers' try/except
    around decryption keeps catching them."""
    depcheck.load(IMPORT_NAME)
    ex = depcheck.load("msoffcrypto.exceptions")
    for name in EXCEPTION_NAMES:
        cls = getattr(ex, name)
        assert isinstance(cls, type) and issubclass(cls, Exception), (
            f"msoffcrypto.exceptions.{name} is not an Exception subclass"
        )


# ---------------------------------------------------------------------------
# Behavioural: OfficeFile over a non-Office stream raises FileFormatError.
# No real document, no password, no network — just the container sniff.
# ---------------------------------------------------------------------------


def test_behaviour_office_file_rejects_non_office_stream(depcheck):
    """OfficeFile() sniffs the container format up front. Handed bytes that are
    not an Office document, it must raise FileFormatError (the signal the loader
    stack uses to fall back / report an unsupported file), not some opaque
    error. Pin that exact behaviour offline."""
    mod = depcheck.load(IMPORT_NAME)
    ex = depcheck.load("msoffcrypto.exceptions")
    bogus = io.BytesIO(b"this is plainly not an office open xml or ole2 file")
    with pytest.raises(ex.FileFormatError):
        mod.OfficeFile(bogus)


def test_behaviour_office_file_rejects_empty_stream(depcheck):
    """An empty upload must also be rejected at the sniff stage (FileFormatError),
    not crash with an unrelated exception type."""
    mod = depcheck.load(IMPORT_NAME)
    ex = depcheck.load("msoffcrypto.exceptions")
    with pytest.raises(ex.FileFormatError):
        mod.OfficeFile(io.BytesIO(b""))
