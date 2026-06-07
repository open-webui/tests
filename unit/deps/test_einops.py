"""Dependency contract: einops.

einops is not imported anywhere in the Open WebUI backend by name; it is a
transitive dependency (pinned ``einops==0.8.2`` in requirements.txt, NOT in
requirements-min.txt) pulled in by the ML stack — sentence-transformers /
faster-whisper / transformers model code use einops' tensor-rearrangement
primitives internally. If an einops bump broke ``rearrange`` / ``reduce`` /
``repeat`` / ``pack`` / ``einsum`` or the ``EinopsError`` type, embedding /
reranking / speech-to-text would fail deep inside a model's forward pass.

Because nothing in the backend names a symbol of its own, this module pins
einops' *core public surface* and exercises each primitive offline against
small NumPy arrays (no torch, no GPU, no model download): shape-only
transforms that any consumer relies on. It does not assume any particular
deep-learning backend beyond NumPy (always present in the env).

Pattern mirrors test_requests.py. Uses the ``depcheck`` fixture.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "einops"
DIST_NAME = "einops"

USED_SYMBOLS = [
    "rearrange",
    "reduce",
    "repeat",
    "einsum",
    "pack",
    "unpack",
    "parse_shape",
    "asnumpy",
    "EinopsError",
]


def _np(depcheck):
    np = depcheck.try_load("numpy")
    if np is None:
        pytest.skip("numpy not installed; einops behavioural tests need an array backend")
    return np


# ---------------------------------------------------------------------------
# Import + version + surface
# ---------------------------------------------------------------------------


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "einops"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_core_callables(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in ("rearrange", "reduce", "repeat", "einsum", "pack", "unpack"):
        depcheck.assert_callable(mod, name)


def test_rearrange_signature(depcheck):
    """rearrange(tensor, pattern, **axes_lengths) is the universal call shape."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.rearrange)
    params = list(sig.parameters)
    assert params[0] in ("tensor", "x")
    assert "pattern" in params


def test_einopserror_is_exception(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.EinopsError, Exception)


# ---------------------------------------------------------------------------
# rearrange — transpose / flatten / split semantics (NumPy backend).
# ---------------------------------------------------------------------------


def test_rearrange_transpose(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = mod.rearrange(x, "b h w -> b w h")
    assert out.shape == (2, 4, 3)


def test_rearrange_flatten(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    out = mod.rearrange(x, "b h w -> b (h w)")
    assert out.shape == (2, 12)


def test_rearrange_split_with_axis_length(depcheck):
    """A grouped axis can be split when a length is supplied — the kwarg path
    consumers use to reshape attention heads etc."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(2 * 12).reshape(2, 12)
    out = mod.rearrange(x, "b (h d) -> b h d", h=3)
    assert out.shape == (2, 3, 4)


def test_rearrange_preserves_values(depcheck):
    """A pure transpose must preserve the data, not just the shape."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(6).reshape(2, 3)
    out = mod.rearrange(x, "a b -> b a")
    assert out.tolist() == [[0, 3], [1, 4], [2, 5]]


# ---------------------------------------------------------------------------
# reduce — mean/sum/max reductions.
# ---------------------------------------------------------------------------


def test_reduce_mean(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(2 * 3 * 4, dtype="float64").reshape(2, 3, 4)
    out = mod.reduce(x, "b h w -> b h", "mean")
    assert out.shape == (2, 3)


def test_reduce_sum_values(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.ones((2, 5), dtype="float64")
    out = mod.reduce(x, "b w -> b", "sum")
    assert out.tolist() == [5.0, 5.0]


def test_reduce_max(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.array([[1.0, 9.0, 3.0], [4.0, 2.0, 8.0]])
    out = mod.reduce(x, "b w -> b", "max")
    assert out.tolist() == [9.0, 8.0]


# ---------------------------------------------------------------------------
# repeat — broadcasting a new axis.
# ---------------------------------------------------------------------------


def test_repeat_adds_axis(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.array([1, 2, 3])
    out = mod.repeat(x, "w -> h w", h=2)
    assert out.shape == (2, 3)
    assert out.tolist() == [[1, 2, 3], [1, 2, 3]]


# ---------------------------------------------------------------------------
# pack / unpack — the 0.6+ ragged-concat helpers.
# ---------------------------------------------------------------------------


def test_pack_and_unpack_round_trip(depcheck):
    """pack concatenates tensors along a '*' axis and returns the packed
    shapes; unpack restores them. Model code uses this to bundle/unbundle
    variable-width features."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    a = np.zeros((2, 3))
    b = np.zeros((2, 5))
    packed, ps = mod.pack([a, b], "batch *")
    assert packed.shape == (2, 8)
    restored = mod.unpack(packed, ps, "batch *")
    assert [r.shape for r in restored] == [(2, 3), (2, 5)]


# ---------------------------------------------------------------------------
# einsum + parse_shape.
# ---------------------------------------------------------------------------


def test_einsum_matmul(depcheck):
    """einops.einsum with named axes performs a matmul-style contraction."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    a = np.ones((2, 3))
    b = np.ones((3, 4))
    out = mod.einsum(a, b, "i j, j k -> i k")
    assert out.shape == (2, 4)
    # each output element is a sum over the shared axis of length 3.
    assert out.tolist() == [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]]


def test_parse_shape(depcheck):
    """parse_shape maps named axes to their sizes (consumers introspect tensor
    dims this way)."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.zeros((2, 3, 4))
    shape = mod.parse_shape(x, "b h w")
    assert shape == {"b": 2, "h": 3, "w": 4}


# ---------------------------------------------------------------------------
# Error contract — a malformed/incompatible pattern raises EinopsError.
# ---------------------------------------------------------------------------


def test_rearrange_bad_pattern_raises_einopserror(depcheck):
    """A pattern whose axis count doesn't match the tensor rank must raise
    EinopsError (not a bare ValueError), so consumers' `except EinopsError`
    handlers fire."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.arange(24).reshape(2, 3, 4)
    with pytest.raises(mod.EinopsError):
        mod.rearrange(x, "b h -> h b")  # 2 axes named for a rank-3 tensor


def test_reduce_unknown_operation_raises(depcheck):
    """An unknown reduction operation must raise (EinopsError) rather than
    silently producing garbage."""
    mod = depcheck.load(IMPORT_NAME)
    np = _np(depcheck)
    x = np.ones((2, 3))
    with pytest.raises(mod.EinopsError):
        mod.reduce(x, "b w -> b", "not_a_real_op")
