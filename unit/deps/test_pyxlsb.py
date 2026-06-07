"""Dependency contract: pyxlsb.

pyxlsb is not imported anywhere in the Open WebUI backend by name; it is a
transitive dependency (pinned ``pyxlsb==1.0.10`` in requirements.txt, NOT in
requirements-min.txt) that serves as the read engine for the binary Excel
``.xlsb`` format. When the document loader / spreadsheet ingestion path
hands an ``.xlsb`` file to pandas (``pd.read_excel(..., engine='pyxlsb')``)
or openpyxl-style tooling, pyxlsb is what actually parses it. A bump that
broke ``open_workbook`` / the workbook-sheet-row object graph, or the
``convert_date`` Excel-serial converter, would make ``.xlsb`` ingestion fail
or silently mis-date cells.

pyxlsb is strictly READ-ONLY — it has no .xlsb writer — so a full
build-in-memory round trip is not possible (and the format is a binary
BIFF12 stream, not something to hand-craft). This module therefore pins:
  - the public API surface (``open_workbook``, ``Workbook``, ``Worksheet``,
    ``convert_date``, the BIFF12 reader);
  - the workbook/worksheet *method surface* the read path walks
    (Workbook.{sheets,get_sheet,close}, Worksheet.{rows,close}) and the
    context-manager protocol consumers use;
  - the ``open_workbook`` error contract (a missing file raises, not a silent
    empty workbook);
  - the full behavioural contract of ``convert_date`` (the Excel 1900-epoch
    serial-to-datetime conversion, including the fractional-day time
    component) — this is offline-testable without any .xlsb input.

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import datetime
import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pyxlsb"
DIST_NAME = "pyxlsb"

USED_SYMBOLS = [
    "open_workbook",
    "convert_date",
    "Workbook",
    "Worksheet",
    "BIFF12Reader",
    # submodules
    "workbook",
    "worksheet",
    "reader",
]


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pyxlsb"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_open_workbook_is_callable(depcheck):
    """The engine entry point pandas drives is open_workbook(name)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "open_workbook")


def test_open_workbook_signature(depcheck):
    """open_workbook(name, debug=False) — the `name` argument (path or
    file-like) must remain its first parameter."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.open_workbook)
    assert list(sig.parameters)[0] == "name"


# ---------------------------------------------------------------------------
# Workbook / Worksheet method surface — the read path walks
# workbook.sheets / workbook.get_sheet(...) / worksheet.rows().
# ---------------------------------------------------------------------------


def test_workbook_method_surface(depcheck):
    """The Workbook class must expose sheets / get_sheet / close — the methods
    any reader uses to enumerate and open sheets."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Workbook))
    for attr in ("sheets", "get_sheet", "close"):
        assert attr in names, f"pyxlsb.Workbook.{attr} missing"


def test_worksheet_method_surface(depcheck):
    """The Worksheet class must expose rows (the cell iterator) and close."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Worksheet))
    for attr in ("rows", "close"):
        assert attr in names, f"pyxlsb.Worksheet.{attr} missing"


def test_workbook_is_context_manager(depcheck):
    """Consumers use `with open_workbook(path) as wb:`; the Workbook class must
    implement the context-manager protocol."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Workbook))
    assert "__enter__" in names and "__exit__" in names, "Workbook is not a context manager"


def test_worksheet_is_context_manager(depcheck):
    """`with wb.get_sheet(1) as ws:` is the per-sheet idiom; Worksheet must be
    a context manager too."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Worksheet))
    assert "__enter__" in names and "__exit__" in names, "Worksheet is not a context manager"


# ---------------------------------------------------------------------------
# open_workbook error contract — a missing file must raise (no silent empty).
# ---------------------------------------------------------------------------


def test_open_workbook_missing_file_raises(depcheck, tmp_path):
    """Opening a non-existent path must raise (FileNotFoundError/OSError), so
    a bad upload surfaces as an error rather than an empty workbook."""
    mod = depcheck.load(IMPORT_NAME)
    missing = tmp_path / "does-not-exist.xlsb"
    with pytest.raises((FileNotFoundError, OSError)):
        mod.open_workbook(str(missing))


def test_open_workbook_non_xlsb_raises(depcheck, tmp_path):
    """A file that exists but is not a valid .xlsb (here: plain text) must
    raise rather than parse garbage — pyxlsb opens it as a zip/OOXML container
    and fails. Guards against silently treating arbitrary bytes as a workbook."""
    mod = depcheck.load(IMPORT_NAME)
    bogus = tmp_path / "not-really.xlsb"
    bogus.write_bytes(b"this is plainly not a binary excel workbook")
    with pytest.raises(Exception):  # noqa: B017 - any parse/format error is acceptable here
        mod.open_workbook(str(bogus))


# ---------------------------------------------------------------------------
# convert_date — the Excel 1900-epoch serial -> datetime contract. Fully
# offline (no .xlsb needed); this is the value-correctness guarantee that a
# date cell read from an .xlsb maps to the right calendar date.
# ---------------------------------------------------------------------------


def test_convert_date_is_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "convert_date")


def test_convert_date_known_serials(depcheck):
    """Pin known Excel serial -> date mappings. Excel's day 1 is 1900-01-01;
    serial 43831 == 2020-01-01 and 44927 == 2023-01-01 (well past the
    fictional 1900-02-29 leap-day quirk, so these are unambiguous)."""
    mod = depcheck.load(IMPORT_NAME)
    d2020 = mod.convert_date(43831)
    d2023 = mod.convert_date(44927)
    assert d2020.year == 2020 and d2020.month == 1 and d2020.day == 1
    assert d2023.year == 2023 and d2023.month == 1 and d2023.day == 1


def test_convert_date_returns_datetime(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    result = mod.convert_date(44927)
    assert isinstance(result, datetime.datetime)


def test_convert_date_fractional_is_time(depcheck):
    """A fractional serial encodes the time of day: .5 == noon. The loader
    relies on this so datetime cells keep their time component."""
    mod = depcheck.load(IMPORT_NAME)
    noon = mod.convert_date(44927.5)
    assert noon.hour == 12
    assert noon.minute == 0
    # quarter day == 06:00
    six_am = mod.convert_date(44927.25)
    assert six_am.hour == 6


def test_convert_date_day_step_is_24h(depcheck):
    """Two serials one integer apart must be exactly 24h apart — pins the
    one-serial-equals-one-day invariant."""
    mod = depcheck.load(IMPORT_NAME)
    a = mod.convert_date(44927)
    b = mod.convert_date(44928)
    assert (b - a) == datetime.timedelta(days=1)
