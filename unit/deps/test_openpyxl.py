"""Dependency contract: openpyxl.

``openpyxl`` is the ``.xlsx`` engine the Open WebUI backend relies on for
spreadsheet ingestion in the retrieval pipeline. It is a *declared*
requirement (``openpyxl==3.1.5``) used transitively: pandas reads
``.xlsx`` via ``engine="openpyxl"`` and LangChain's Excel loaders sit on
top of that. A breaking bump (renamed ``load_workbook`` kwargs, changed
``Workbook`` / worksheet cell API) would surface as failed Excel
ingestion rather than at import time.

This module pins the load-bearing surface (``Workbook`` /
``load_workbook`` and the worksheet cell/iteration API) and exercises the
real read/write path offline: build a workbook in a ``BytesIO`` buffer,
write cells across multiple sheets, save, reload, and read the values
back — including the ``read_only`` / ``data_only`` modes and the pandas
interop that is the dominant transitive consumer. No files on disk, no
network.

Pattern mirrors test_requests.py / test_pandas.py. Uses ``depcheck``.
"""

from __future__ import annotations

import inspect
from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "openpyxl"
DIST_NAME = "openpyxl"

TOP_LEVEL_SYMBOLS = [
    "Workbook",  # openpyxl.Workbook() — create
    "load_workbook",  # openpyxl.load_workbook(...) — read
    "open",  # alias of load_workbook
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`openpyxl` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "openpyxl"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence + signature checks.
# ---------------------------------------------------------------------------


def test_top_level_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_workbook_is_class_and_load_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Workbook)
    assert callable(mod.load_workbook)


def test_load_workbook_signature(depcheck):
    """pandas/loaders call load_workbook(src, read_only=, data_only=). Those
    keyword arguments must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.load_workbook,
        ["filename", "read_only", "data_only"],
    )


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE, in-memory) — real read/write round-trips.
# ---------------------------------------------------------------------------


def test_behaviour_write_read_cells(depcheck):
    """Write cells by coordinate (A1/B1) and via append(), save to bytes,
    reload, and read the values back. This is the fundamental contract Excel
    ingestion depends on."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    ws = wb.active
    ws.title = "Data"
    ws["A1"] = "hello"
    ws["B1"] = 42
    ws.append(["row2a", 3.14])

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    wb2 = mod.load_workbook(buf)
    assert "Data" in wb2.sheetnames
    ws2 = wb2["Data"]
    assert ws2["A1"].value == "hello"
    assert ws2["B1"].value == 42
    assert ws2["A2"].value == "row2a"
    assert ws2["B2"].value == 3.14


def test_behaviour_dimensions_and_iter_rows(depcheck):
    """max_row / max_column and iter_rows(values_only=True) are the iteration
    surface loaders use to walk a sheet; pin them."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    ws = wb.active
    ws.append(["id", "name"])
    ws.append([1, "a"])
    ws.append([2, "b"])

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    ws2 = mod.load_workbook(buf).active
    assert ws2.max_row == 3
    assert ws2.max_column == 2
    rows = [list(r) for r in ws2.iter_rows(values_only=True)]
    assert rows == [["id", "name"], [1, "a"], [2, "b"]]


def test_behaviour_multiple_sheets(depcheck):
    """create_sheet adds named worksheets; multi-sheet workbooks must round-trip
    with their sheet names preserved (Excel files commonly have many tabs)."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    wb.active.title = "first"
    s2 = wb.create_sheet("second")
    s2["A1"] = "on second"

    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    wb2 = mod.load_workbook(buf)
    assert wb2.sheetnames == ["first", "second"]
    assert wb2["second"]["A1"].value == "on second"


def test_behaviour_read_only_mode(depcheck):
    """read_only=True is the streaming mode used for large spreadsheets; it must
    still yield the row values and be closeable."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    ws = wb.active
    ws.append(["x", "y"])
    ws.append([1, 2])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)

    wb_ro = mod.load_workbook(buf, read_only=True)
    try:
        rows = [list(r) for r in wb_ro.active.iter_rows(values_only=True)]
        assert rows == [["x", "y"], [1, 2]]
    finally:
        wb_ro.close()


def test_behaviour_cell_access_by_row_column(depcheck):
    """ws.cell(row=, column=) is the programmatic accessor (1-based); pin both
    setting and reading by index."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    ws = wb.active
    ws.cell(row=1, column=1, value="r1c1")
    ws.cell(row=2, column=3, value="r2c3")
    assert ws.cell(row=1, column=1).value == "r1c1"
    assert ws.cell(row=2, column=3).value == "r2c3"


def test_behaviour_data_types_preserved(depcheck):
    """Excel cells carry typed values; int / float / str / bool / None must each
    round-trip with their Python type intact (loaders depend on this typing)."""
    mod = depcheck.load(IMPORT_NAME)
    wb = mod.Workbook()
    ws = wb.active
    # None is placed mid-row (a trailing None is trimmed by openpyxl); the
    # trailing cell is non-empty so all five positions survive.
    ws.append([1, 2.5, "text", None, True])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    row = next(mod.load_workbook(buf).active.iter_rows(values_only=True))
    assert row[0] == 1 and isinstance(row[0], int)
    assert row[1] == 2.5 and isinstance(row[1], float)
    assert row[2] == "text"
    assert row[3] is None
    assert row[4] is True


def test_behaviour_pandas_reads_via_openpyxl(depcheck):
    """The dominant transitive use is pandas.read_excel(engine='openpyxl').
    Verify a workbook written by openpyxl reads back into a DataFrame with the
    right columns/values — the actual Open WebUI Excel ingestion path."""
    mod = depcheck.load(IMPORT_NAME)
    pd = depcheck.try_load("pandas")
    if pd is None:
        pytest.skip("pandas not installed; openpyxl's native contracts still apply")
    wb = mod.Workbook()
    ws = wb.active
    ws.append(["id", "name"])
    ws.append([1, "a"])
    ws.append([2, "b"])
    buf = BytesIO()
    wb.save(buf)
    buf.seek(0)
    df = pd.read_excel(buf, engine="openpyxl")
    assert list(df.columns) == ["id", "name"]
    assert df.shape == (2, 2)
    assert df["name"].tolist() == ["a", "b"]


def test_behaviour_pandas_writes_via_openpyxl(depcheck):
    """pandas.to_excel(engine='openpyxl') is the write counterpart; verify a
    DataFrame written through openpyxl reloads with openpyxl intact."""
    mod = depcheck.load(IMPORT_NAME)
    pd = depcheck.try_load("pandas")
    if pd is None:
        pytest.skip("pandas not installed")
    df = pd.DataFrame({"a": [10, 20], "b": ["p", "q"]})
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="out")
    buf.seek(0)
    wb = mod.load_workbook(buf)
    assert "out" in wb.sheetnames
    rows = [list(r) for r in wb["out"].iter_rows(values_only=True)]
    assert rows[0] == ["a", "b"]
    assert rows[1] == [10, "p"]
