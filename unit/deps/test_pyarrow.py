"""Dependency contract: pyarrow.

``pyarrow`` is a *declared* requirement of the Open WebUI backend
(``pyarrow==20.0.0`` — explicitly pinned to 20 for Raspberry-Pi
compatibility, see PR #15897). It is not imported under a stable internal
chokepoint in the application code; it is pulled in transitively (pandas
parquet/feather I/O, the ``datasets`` library used for speaker embeddings,
etc.) and the hard version pin makes its surface load-bearing for the
install. pyarrow is a large native (C++/Arrow) extension, so this module
pins it the way the exemplar prescribes for C/Rust objects: a small set of
*behavioural* contracts (build arrays/tables/schemas in memory, round-trip
through Parquet and the Arrow IPC stream via ``BytesIO``/``BufferOutput``,
and convert to/from pandas) rather than introspecting opaque native
attributes. Everything is in-memory — no files on disk, no network.

Pattern mirrors test_requests.py / test_pandas.py. Uses ``depcheck``.
"""

from __future__ import annotations

from io import BytesIO

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pyarrow"
DIST_NAME = "pyarrow"

# Core constructors / types the Arrow API surface exposes (used by every
# transitive consumer). These are module-level callables, safe to resolve.
CORE_SYMBOLS = [
    "array",  # pa.array(...)
    "table",  # pa.table({...})
    "Table",  # pa.Table
    "schema",  # pa.schema([...])
    "field",  # pa.field(name, type)
    "chunked_array",
    "RecordBatch",
    "Array",
    "ChunkedArray",
    "Schema",
    "Field",
    "DataType",
    # primitive type factories
    "int64",
    "int32",
    "float64",
    "string",
    "bool_",
    "list_",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`pyarrow` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pyarrow"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_version_is_pinned_major(depcheck):
    """requirements pin pyarrow to 20 (rpi compatibility, #15897). A different
    major is a meaningful drift worth flagging — but only warn-skip if the env
    differs from the pin rather than hard-failing an unrelated install."""
    mod = depcheck.load(IMPORT_NAME)
    ver = getattr(mod, "__version__", None) or depcheck.dist_version(DIST_NAME)
    assert ver is not None
    # Soft check: surface a clear message if not the pinned major.
    if int(str(ver).split(".")[0]) != 20:
        pytest.skip(f"pyarrow {ver} installed; requirements pin 20 (#15897)")


def test_parquet_submodule_imports(depcheck):
    """pandas/datasets reach pyarrow.parquet for Parquet I/O; it must import."""
    mod = depcheck.load("pyarrow.parquet")
    assert mod.__name__ == "pyarrow.parquet"


def test_ipc_submodule_imports(depcheck):
    """The Arrow IPC (Feather/stream) path lives in pyarrow.ipc."""
    mod = depcheck.load("pyarrow.ipc")
    assert mod.__name__ == "pyarrow.ipc"


# ---------------------------------------------------------------------------
# Symbol-existence checks (module-level constructors — safe to resolve).
# ---------------------------------------------------------------------------


def test_core_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, CORE_SYMBOLS)


def test_parquet_read_write_callable(depcheck):
    """pq.write_table / pq.read_table are the canonical Parquet entry points."""
    pq = depcheck.load("pyarrow.parquet")
    assert callable(pq.write_table)
    assert callable(pq.read_table)


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE, in-memory) — the right way to pin a native
# extension: exercise real semantics, not opaque attribute shapes.
# ---------------------------------------------------------------------------


def test_behaviour_array_roundtrip(depcheck):
    """pa.array([...], type=) builds a typed columnar array; to_pylist must
    round-trip the values and the dtype must be the requested one."""
    pa = depcheck.load(IMPORT_NAME)
    arr = pa.array([1, 2, 3], type=pa.int64())
    assert arr.to_pylist() == [1, 2, 3]
    assert arr.type == pa.int64()
    assert len(arr) == 3


def test_behaviour_array_with_nulls(depcheck):
    """Null handling is core Arrow semantics: a None element becomes a null and
    is reported by null_count, surviving the round-trip as None."""
    pa = depcheck.load(IMPORT_NAME)
    arr = pa.array([1, None, 3])
    assert arr.null_count == 1
    assert arr.to_pylist() == [1, None, 3]


def test_behaviour_table_construction_and_columns(depcheck):
    """pa.table({col: values}) builds a Table with named columns; pin
    column_names / num_rows / num_columns and a column read."""
    pa = depcheck.load(IMPORT_NAME)
    t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    assert t.column_names == ["id", "name"]
    assert t.num_rows == 3
    assert t.num_columns == 2
    assert t.column("id").to_pylist() == [1, 2, 3]
    assert t.to_pydict() == {"id": [1, 2, 3], "name": ["a", "b", "c"]}


def test_behaviour_schema_construction(depcheck):
    """pa.schema([(name, type)]) / pa.field define table structure; pin the
    names and per-field type lookup."""
    pa = depcheck.load(IMPORT_NAME)
    s = pa.schema([pa.field("a", pa.int64()), pa.field("b", pa.string())])
    assert s.names == ["a", "b"]
    assert s.field("a").type == pa.int64()
    assert s.field("b").type == pa.string()


def test_behaviour_parquet_roundtrip_in_memory(depcheck):
    """Write a Table to Parquet bytes (BytesIO) and read it back — the exact
    operation pandas.to_parquet/read_parquet delegate to. The data must be
    byte-for-byte equivalent. No file touches disk."""
    pa = depcheck.load(IMPORT_NAME)
    pq = depcheck.load("pyarrow.parquet")
    t = pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    buf = BytesIO()
    pq.write_table(t, buf)
    buf.seek(0)
    out = pq.read_table(buf)
    assert out.equals(t)
    assert out.to_pydict() == {"id": [1, 2, 3], "name": ["a", "b", "c"]}


def test_behaviour_ipc_stream_roundtrip(depcheck):
    """The Arrow IPC stream (Feather/flight wire format) must round-trip a Table
    through an in-memory sink — pin new_stream/write_table + open_stream/read."""
    pa = depcheck.load(IMPORT_NAME)
    t = pa.table({"x": [10, 20], "y": [1.5, 2.5]})
    sink = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, t.schema)
    writer.write_table(t)
    writer.close()
    payload = sink.getvalue()
    assert payload.size > 0
    reader = pa.ipc.open_stream(payload)
    out = reader.read_all()
    assert out.equals(t)


def test_behaviour_pandas_interop(depcheck):
    """Table <-> pandas DataFrame conversion is the dominant transitive path
    (pandas-backed parquet/feather). Round-trip a table through pandas and
    back, preserving columns and values."""
    pa = depcheck.load(IMPORT_NAME)
    pd = depcheck.try_load("pandas")
    if pd is None:
        pytest.skip("pandas not installed; pyarrow's other contracts still apply")
    t = pa.table({"id": [1, 2], "label": ["p", "q"]})
    df = t.to_pandas()
    assert list(df.columns) == ["id", "label"]
    assert df["id"].tolist() == [1, 2]
    back = pa.Table.from_pandas(df, preserve_index=False)
    assert back.column("id").to_pylist() == [1, 2]
    assert back.column("label").to_pylist() == ["p", "q"]


def test_behaviour_chunked_array_concat(depcheck):
    """ChunkedArray is how Arrow represents multi-chunk columns; pin that a
    chunked array reports its combined length and flattens to_pylist."""
    pa = depcheck.load(IMPORT_NAME)
    ca = pa.chunked_array([[1, 2], [3, 4, 5]])
    assert len(ca) == 5
    assert ca.to_pylist() == [1, 2, 3, 4, 5]
    assert ca.num_chunks == 2


def test_behaviour_table_select_and_filter(depcheck):
    """Column projection (select) is used by parquet readers pushing down
    columns; pin that selecting a subset yields a narrower table."""
    pa = depcheck.load(IMPORT_NAME)
    t = pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    narrowed = t.select(["a", "c"])
    assert narrowed.column_names == ["a", "c"]
    assert narrowed.num_rows == 3
