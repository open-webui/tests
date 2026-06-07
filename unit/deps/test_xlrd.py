"""Dependency contract: xlrd (import name ``xlrd``).

xlrd reads legacy ``.xls`` (BIFF / OLE2) spreadsheets. Open WebUI pins it in
``backend/requirements.txt`` (``xlrd==2.0.2``) but does NOT import it directly
in ``open_webui/*``: it is a *transitive* dependency of the spreadsheet
ingestion path — pandas' ``read_excel`` selects the ``xlrd`` engine for old
``.xls`` files, and the unstructured/langchain Excel loaders used by
``retrieval/loaders/main.py`` rely on that. When a user uploads a legacy
``.xls`` for RAG, xlrd is what parses it.

xlrd 2.0 deliberately reads ONLY ``.xls`` (support for ``.xlsx`` was removed —
that now goes through openpyxl). Because nothing in the backend names xlrd
directly, this module pins its *core read surface* so a bump that broke it
surfaces here rather than as an opaque "can't read this .xls" during
ingestion. We pin: ``open_workbook`` (the entry point) and its signature
(including the ``file_contents=`` kwarg pandas uses to read from a buffer), the
``Book`` / sheet / cell read API, the cell-type constants, the pure
coordinate helpers (``cellname`` / ``colname``), and the ``XLRDError``
hierarchy. Behavioural contracts use only pure functions and in-memory byte
buffers (no real workbook file is needed to prove the error paths and the
helpers) — deterministic, no network.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import datetime

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "xlrd"
DIST_NAME = "xlrd"

# Top-level symbols pandas' xlrd engine / loaders resolve.
TOP_LEVEL_SYMBOLS = [
    "open_workbook",  # the entry point
    "Book",  # the workbook object type
    "XLRDError",  # error raised on a malformed/unsupported file
    "biffh",  # BIFF helpers submodule (XLRDError lives here)
    "xldate",  # serial-date -> datetime conversion
    "cellname",  # pure coordinate helper
    "colname",  # pure column-letter helper
    "XL_CELL_TEXT",  # cell-type discriminators
    "XL_CELL_NUMBER",
    "XL_CELL_DATE",
    "XL_CELL_EMPTY",
]

# Book (workbook) methods the read path uses to walk sheets.
BOOK_METHODS = ["sheet_by_index", "sheet_by_name", "sheets", "sheet_names"]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`xlrd` must import (skip cleanly if absent in this env)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "xlrd"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_is_v2(depcheck):
    """xlrd 2.x is the .xls-only line the ingestion stack expects (1.x also read
    .xlsx, which would change engine-selection assumptions). Guard the major."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__version__.split(".")[0] == "2", (
        f"Expected xlrd 2.x (.xls-only), got {mod.__version__}."
    )


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface)
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    """Every top-level `xlrd.*` symbol the read path resolves must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_open_workbook_callable(depcheck):
    """pandas' xlrd engine calls xlrd.open_workbook(...). Must be callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "open_workbook")


def test_open_workbook_accepts_file_contents_kwarg(depcheck):
    """pandas reads .xls from an in-memory buffer via
    open_workbook(file_contents=bytes). Both `filename` and `file_contents` must
    remain accepted parameters."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.open_workbook, ["filename", "file_contents"])


def test_book_methods_exist(depcheck):
    """The Book class must expose the sheet-walking methods the read path uses."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Book))
    missing = [m for m in BOOK_METHODS if m not in names]
    assert not missing, f"xlrd.Book missing method(s): {missing}"


def test_xldate_as_datetime_exists(depcheck):
    """Excel stores dates as serial numbers; xldate.xldate_as_datetime converts
    them. Loaders rely on this to surface date cells as real datetimes."""
    depcheck.load(IMPORT_NAME)
    xldate = depcheck.load("xlrd.xldate")
    assert hasattr(xldate, "xldate_as_datetime"), "xlrd.xldate.xldate_as_datetime is gone"
    assert callable(xldate.xldate_as_datetime)


# ---------------------------------------------------------------------------
# Cell-type constants — discriminators the read path branches on
# ---------------------------------------------------------------------------


def test_cell_type_constants_are_distinct_ints(depcheck):
    """The XL_CELL_* constants discriminate cell value types when reading; they
    must remain distinct integers so the read path can switch on them."""
    mod = depcheck.load(IMPORT_NAME)
    consts = {
        "XL_CELL_EMPTY": mod.XL_CELL_EMPTY,
        "XL_CELL_TEXT": mod.XL_CELL_TEXT,
        "XL_CELL_NUMBER": mod.XL_CELL_NUMBER,
        "XL_CELL_DATE": mod.XL_CELL_DATE,
    }
    for name, val in consts.items():
        assert isinstance(val, int), f"{name} is not an int"
    assert len(set(consts.values())) == len(consts), (
        f"XL_CELL_* constants are no longer distinct: {consts}"
    )


# ---------------------------------------------------------------------------
# Behavioural: pure coordinate helpers (deterministic, no workbook needed)
# ---------------------------------------------------------------------------


def test_behaviour_cellname_a1_notation(depcheck):
    """cellname(row, col) returns A1-style references. Pin the mapping the read
    path uses to label cells."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.cellname(0, 0) == "A1"
    assert mod.cellname(1, 2) == "C2"
    assert mod.cellname(9, 0) == "A10"


def test_behaviour_colname_letters(depcheck):
    """colname(col) maps a 0-based column index to its spreadsheet letter(s)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.colname(0) == "A"
    assert mod.colname(25) == "Z"
    assert mod.colname(26) == "AA"


def test_behaviour_xldate_as_datetime_converts_serial(depcheck):
    """A known Excel serial date must convert to the expected calendar date.
    Serial 1 with datemode 0 (1900 system) is 1899-12-31 in xlrd's convention;
    rather than hard-code the off-by-one epoch quirk, assert a mid-range serial
    converts to a real datetime in the right year. Serial 40000 ~ mid-2009."""
    depcheck.load(IMPORT_NAME)
    xldate = depcheck.load("xlrd.xldate")
    dt = xldate.xldate_as_datetime(40000, 0)
    assert isinstance(dt, datetime.datetime)
    assert dt.year == 2009, f"xldate serial 40000 mapped to unexpected year {dt.year}"


# ---------------------------------------------------------------------------
# Behavioural: error paths over in-memory buffers (no real workbook file)
# ---------------------------------------------------------------------------


def test_behaviour_garbage_bytes_raise_xlrderror(depcheck):
    """Handed bytes that are neither BIFF nor a zip, open_workbook must raise
    XLRDError — the signal the ingestion stack uses to reject/skip an
    unparseable file. Pin that exact error type."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(mod.XLRDError):
        mod.open_workbook(file_contents=b"this is definitely not a BIFF .xls stream")


def test_behaviour_xlsx_bytes_rejected_in_v2(depcheck):
    """xlrd 2.0 no longer reads .xlsx (zip/OOXML). Handed a zip-magic payload it
    must NOT silently parse it as a workbook; it raises (XLRDError or a
    zip-related error). This pins the 2.0 'xls-only' boundary that the engine
    selection in the loader stack depends on."""
    mod = depcheck.load(IMPORT_NAME)
    # "PK\x03\x04" is the zip local-file-header magic (xlsx is a zip).
    with pytest.raises(Exception):
        mod.open_workbook(file_contents=b"PK\x03\x04" + b"\x00" * 64)


def test_behaviour_empty_bytes_rejected(depcheck):
    """An empty upload must be rejected rather than yielding a phantom empty
    workbook."""
    mod = depcheck.load(IMPORT_NAME)
    with pytest.raises(Exception):
        mod.open_workbook(file_contents=b"")


def test_xlrderror_is_in_biffh(depcheck):
    """XLRDError is re-exported at the top level but defined in xlrd.biffh; pin
    that they're the same class so `except xlrd.XLRDError` keeps catching errors
    raised from the BIFF parser."""
    mod = depcheck.load(IMPORT_NAME)
    biffh = depcheck.load("xlrd.biffh")
    assert biffh.XLRDError is mod.XLRDError, (
        "xlrd.XLRDError and xlrd.biffh.XLRDError diverged; a parser error might "
        "escape an `except xlrd.XLRDError` handler."
    )
    assert issubclass(mod.XLRDError, Exception)
