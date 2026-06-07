"""Dependency contract: pandas.

pandas is Open WebUI's fallback tabular reader in the retrieval pipeline.
When `unstructured` is not installed, `retrieval/loaders/main.py:ExcelLoader`
parses spreadsheets with pandas: it opens the workbook with
`pd.ExcelFile(path)`, iterates `xls.sheet_names`, reads each sheet via
`pd.read_excel(xls, sheet_name=...)`, and renders it to text with
`df.to_string(index=False)`. That whole loader silently produces empty or
malformed document text if a pandas bump removed/renamed any of those
symbols, changed the `read_excel(io, sheet_name=...)` signature, or altered
`to_string(index=False)`'s shape — so this module pins both the API surface
and the behaviour the loader depends on, plus the closely-related
CSV-reading surface (`read_csv`) and the DataFrame selection/cleaning ops
(`to_dict`, `iterrows`, `fillna`, `dropna`, `.columns`, `.head`) that a
tabular reader of this kind relies on.

All contracts run fully offline and deterministically: DataFrames are built
in memory, CSVs are parsed from `io.StringIO`, and the Excel round-trip
writes an xlsx into an `io.BytesIO` via the openpyxl engine (no disk, no
network). Excel cases skip cleanly if openpyxl is unavailable.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface)
+ offline behavioural contracts. Uses the `depcheck` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import io

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pandas"
DIST_NAME = "pandas"

# Top-level pandas symbols the Open WebUI backend touches. ExcelFile /
# read_excel / DataFrame are used directly by the ExcelLoader; read_csv and
# the DataFrame ops below are the rest of the tabular-reader surface a CSV/
# spreadsheet loader of this kind depends on.
USED_SYMBOLS = [
    "ExcelFile",
    "read_excel",
    "read_csv",
    "DataFrame",
    "Series",
    "ExcelWriter",
    "isna",
    "concat",
]

# Methods/attrs the codebase (and any tabular reader) calls on a DataFrame.
DATAFRAME_API = [
    "to_string",
    "to_dict",
    "to_json",
    "to_markdown",
    "iterrows",
    "fillna",
    "dropna",
    "isna",
    "columns",
    "head",
    "shape",
    "values",
]


# ---------------------------------------------------------------------------
# Local helpers (offline, deterministic). Defined at module scope so many
# tests share them without paying for a session fixture.
# ---------------------------------------------------------------------------

CSV_TEXT = "name,age,city\nAlice,30,NYC\nBob,25,LA\n"
CSV_WITH_GAPS = "a,b\n1,\n,4\n"


def _frame(pd):
    """A small, fully-specified DataFrame mirroring a parsed sheet."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "age": [30, 25],
            "city": ["NYC", "LA"],
        }
    )


def _xlsx_bytes(pd, openpyxl_mod):
    """Build a two-sheet xlsx into a BytesIO via the openpyxl engine.

    Mirrors the ExcelLoader's input (a multi-sheet workbook) without
    touching disk. Caller must have confirmed openpyxl is importable.
    """
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2], "y": ["a", "b"]}).to_excel(writer, sheet_name="S1", index=False)
        pd.DataFrame({"z": [9]}).to_excel(writer, sheet_name="S2", index=False)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# API surface: import + symbol existence + version.
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pandas"


def test_used_symbols_exist(depcheck):
    """Every top-level pandas symbol the codebase relies on must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable, and the
    module-level __version__ agrees with it (bump tooling reads both)."""
    mod = depcheck.load(IMPORT_NAME)
    dist = depcheck.dist_version(DIST_NAME)
    assert dist is not None
    assert getattr(mod, "__version__", None) is not None


def test_loader_entrypoints_callable(depcheck):
    """The three callables the ExcelLoader invokes directly."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "ExcelFile")
    depcheck.assert_callable(mod, "read_excel")
    depcheck.assert_callable(mod, "read_csv")


def test_dataframe_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.DataFrame, type)
    assert isinstance(mod.Series, type)


# ---------------------------------------------------------------------------
# API surface: signatures of the read functions the loader configures.
# ---------------------------------------------------------------------------


def test_read_excel_signature(depcheck):
    """ExcelLoader calls `read_excel(xls, sheet_name=...)`. Both the
    first-positional source param (`io`) and the `sheet_name` kwarg must
    remain; `engine` is the implicit openpyxl selector for .xlsx."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.read_excel, ["io", "sheet_name", "engine"])


def test_read_csv_signature(depcheck):
    """A CSV reader of this kind relies on read_csv's stable kwargs:
    source buffer + sep/header/dtype/na_values/usecols/nrows."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.read_csv,
        ["filepath_or_buffer", "sep", "header", "dtype", "na_values", "usecols", "nrows"],
    )


def test_excelfile_signature(depcheck):
    """`pd.ExcelFile(self.file_path)` — the constructor must still accept a
    path/buffer first-positionally and an `engine` selector."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.ExcelFile.__init__, ["path_or_buffer", "engine"])


def test_to_string_accepts_index_kwarg(depcheck):
    """ExcelLoader renders each sheet with `df.to_string(index=False)`; the
    `index` kwarg (suppressing the row-index column) must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.DataFrame.to_string, ["index"])


def test_dataframe_api_present(depcheck):
    """A constructed DataFrame must expose the method/attr surface the
    loader (and a tabular reader generally) calls — read names off `dir`
    so property descriptors aren't executed."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(_frame(mod)))
    for attr in DATAFRAME_API:
        assert attr in names, f"DataFrame.{attr} missing in this pandas"
    df = _frame(mod)
    for meth in ("to_string", "to_dict", "iterrows", "fillna", "dropna", "head"):
        assert callable(getattr(df, meth)), f"DataFrame.{meth} not callable"


# ---------------------------------------------------------------------------
# Behavioural contracts: DataFrame construction, selection, attributes.
# ---------------------------------------------------------------------------


def test_dataframe_shape_and_columns(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["name", "age", "city"]


def test_dataframe_dtype_inference(depcheck):
    """An integer column infers an integer dtype (the loader relies on
    pandas typing the cells, then stringifying them)."""
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    assert str(df["age"].dtype).startswith("int")
    assert df["age"].sum() == 55


def test_dataframe_row_selection(depcheck):
    """`.iloc[i]` / column indexing yield the expected scalar values."""
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    assert df.iloc[0]["name"] == "Alice"
    assert df.iloc[1]["age"] == 25
    assert list(df["city"]) == ["NYC", "LA"]


def test_dataframe_head(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    assert df.head(1).shape == (1, 3)
    assert list(df.head(1)["name"]) == ["Alice"]


# ---------------------------------------------------------------------------
# Behavioural contracts: rendering / serialisation (to_string is the one the
# ExcelLoader uses; to_dict/to_json/to_markdown round out the surface).
# ---------------------------------------------------------------------------


def test_to_string_index_false_drops_row_index(depcheck):
    """The loader's exact call: `to_string(index=False)`. The rendered text
    must contain the header and cell values and must NOT prefix rows with a
    0/1 positional index column."""
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    rendered = df.to_string(index=False)
    assert isinstance(rendered, str)
    for token in ("name", "age", "city", "Alice", "Bob", "NYC"):
        assert token in rendered
    # index=False => the first non-space token is the first column header,
    # not a row-index number.
    assert rendered.lstrip().startswith("name")
    # And the default (index=True) DOES prefix with the row index, proving
    # the kwarg actually toggles behaviour rather than being a no-op.
    with_index = df.to_string(index=True)
    assert with_index != rendered


def test_to_dict_records(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    records = df.to_dict(orient="records")
    assert isinstance(records, list)
    assert records[0] == {"name": "Alice", "age": 30, "city": "NYC"}
    assert records[1]["name"] == "Bob"


def test_to_json_is_string(depcheck):
    import json

    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    payload = df.to_json(orient="records")
    assert isinstance(payload, str)
    parsed = json.loads(payload)
    assert parsed[0]["name"] == "Alice"
    assert parsed[1]["age"] == 25


def test_to_markdown_is_string(depcheck):
    """to_markdown renders a pipe table (needs `tabulate`); skip cleanly if
    that optional dep is absent rather than failing the contract suite."""
    mod = depcheck.load(IMPORT_NAME)
    if depcheck.try_load("tabulate") is None:
        pytest.skip("tabulate not installed; pandas.to_markdown unavailable")
    df = _frame(mod)
    md = df.to_markdown(index=False)
    assert isinstance(md, str)
    assert "|" in md
    assert "name" in md and "Alice" in md


def test_iterrows_yields_index_and_series(depcheck):
    """`for _, row in df.iterrows()` yields (label, Series) pairs; row[col]
    indexing must work."""
    mod = depcheck.load(IMPORT_NAME)
    df = _frame(mod)
    rows = list(df.iterrows())
    assert len(rows) == 2
    idx0, row0 = rows[0]
    assert isinstance(row0, mod.Series)
    assert row0["name"] == "Alice"
    assert [row["age"] for _, row in df.iterrows()] == [30, 25]


# ---------------------------------------------------------------------------
# Behavioural contracts: missing-data cleaning (fillna / dropna / isna).
# ---------------------------------------------------------------------------


def test_isna_counts_missing(depcheck):
    """Empty CSV cells parse to NaN; isna() locates them."""
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO(CSV_WITH_GAPS))
    assert int(df.isna().sum().sum()) == 2


def test_fillna_replaces_missing(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO(CSV_WITH_GAPS))
    filled = df.fillna(0)
    assert int(filled.isna().sum().sum()) == 0
    assert filled.to_dict(orient="records") == [
        {"a": 1.0, "b": 0.0},
        {"a": 0.0, "b": 4.0},
    ]


def test_dropna_removes_rows_with_missing(depcheck):
    """Every row in the fixture has a missing cell, so dropna() empties it;
    confirms dropna operates row-wise by default."""
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO(CSV_WITH_GAPS))
    assert df.dropna().shape == (0, 2)


# ---------------------------------------------------------------------------
# Behavioural contracts: read_csv from an in-memory buffer.
# ---------------------------------------------------------------------------


def test_read_csv_from_stringio(depcheck):
    """read_csv parses an in-memory text buffer into a typed DataFrame —
    header row promoted to columns, numeric column typed as int."""
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO(CSV_TEXT))
    assert df.shape == (2, 3)
    assert list(df.columns) == ["name", "age", "city"]
    assert str(df["age"].dtype).startswith("int")
    assert df.iloc[0]["name"] == "Alice"
    assert df.iloc[1]["city"] == "LA"


def test_read_csv_respects_dtype_override(depcheck):
    """read_csv(dtype=...) forces a column's type — the kwarg must apply.
    Asserting on the cell value (a Python str), not the numpy/StringDtype
    spelling, which differs across pandas versions."""
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO(CSV_TEXT), dtype={"age": str})
    val = df["age"].iloc[0]
    assert isinstance(val, str)
    assert val == "30"


def test_read_csv_custom_separator(depcheck):
    """The `sep` kwarg switches the delimiter (semicolon-separated input)."""
    mod = depcheck.load(IMPORT_NAME)
    df = mod.read_csv(io.StringIO("p;q\n1;2\n"), sep=";")
    assert list(df.columns) == ["p", "q"]
    assert df.iloc[0]["q"] == 2


# ---------------------------------------------------------------------------
# Behavioural contracts: the full ExcelLoader path, in memory via openpyxl.
# Each skips cleanly (not fails) when openpyxl is unavailable.
# ---------------------------------------------------------------------------


def test_excelfile_lists_sheet_names(depcheck):
    """`pd.ExcelFile(buf).sheet_names` must enumerate every sheet — the
    loop the ExcelLoader iterates over."""
    mod = depcheck.load(IMPORT_NAME)
    openpyxl_mod = depcheck.try_load("openpyxl")
    if openpyxl_mod is None:
        pytest.skip("openpyxl not installed; cannot build/read an xlsx")
    xls = mod.ExcelFile(_xlsx_bytes(mod, openpyxl_mod))
    try:
        assert xls.sheet_names == ["S1", "S2"]
    finally:
        xls.close()


def test_read_excel_from_bytesio_specific_sheet(depcheck):
    """`pd.read_excel(xls, sheet_name=name)` reads one named sheet back into
    a DataFrame with the right shape/columns/values."""
    mod = depcheck.load(IMPORT_NAME)
    openpyxl_mod = depcheck.try_load("openpyxl")
    if openpyxl_mod is None:
        pytest.skip("openpyxl not installed; cannot build/read an xlsx")
    xls = mod.ExcelFile(_xlsx_bytes(mod, openpyxl_mod))
    try:
        s1 = mod.read_excel(xls, sheet_name="S1")
        assert s1.shape == (2, 2)
        assert list(s1.columns) == ["x", "y"]
        assert s1.iloc[0]["x"] == 1
        assert s1.iloc[1]["y"] == "b"
        s2 = mod.read_excel(xls, sheet_name="S2")
        assert list(s2.columns) == ["z"]
        assert s2.iloc[0]["z"] == 9
    finally:
        xls.close()


def test_excel_loader_end_to_end_text(depcheck):
    """Reproduce the ExcelLoader's full transform offline: open the workbook,
    iterate sheet_names, read each sheet, and join
    `f"Sheet: {name}\\n{df.to_string(index=False)}"`. The assembled text must
    contain every sheet header and its cell values — the exact contract the
    loader's document output depends on."""
    mod = depcheck.load(IMPORT_NAME)
    openpyxl_mod = depcheck.try_load("openpyxl")
    if openpyxl_mod is None:
        pytest.skip("openpyxl not installed; cannot build/read an xlsx")
    xls = mod.ExcelFile(_xlsx_bytes(mod, openpyxl_mod))
    try:
        parts = []
        for sheet_name in xls.sheet_names:
            df = mod.read_excel(xls, sheet_name=sheet_name)
            parts.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")
        text = "\n\n".join(parts)
    finally:
        xls.close()
    assert "Sheet: S1" in text
    assert "Sheet: S2" in text
    for token in ("x", "y", "a", "b", "z", "9"):
        assert token in text
