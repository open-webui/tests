"""Dependency contract: onnxruntime (import name ``onnxruntime``).

onnxruntime is a **pinned direct requirement** of the Open WebUI backend
(``onnxruntime==1.26.0``, and via ``rapidocr-onnxruntime``) but is *not*
imported anywhere in the ``open_webui`` package source directly. It is the
inference engine underneath several optional retrieval/OCR features: ChromaDB's
default ONNX MiniLM embedding function, the fastembed path, and rapidocr's
ONNX text recognition. Those consumers all reach onnxruntime the same way —
build a ``SessionOptions``, construct an ``InferenceSession(model, providers=
['CPUExecutionProvider', ...])``, then ``session.run(...)`` with numpy inputs.

Running an actual embedding/OCR model requires a multi-MB model download, which
this offline suite must NOT do. So this module pins the core onnxruntime surface
(``InferenceSession`` + its run/IO-introspection methods, ``SessionOptions`` and
its tunables, ``GraphOptimizationLevel``, the provider-enumeration helpers, and
``OrtValue``) and runs a **model-free smoke** — an ``OrtValue`` round-trip from a
numpy array (a real ORT tensor object, no model) and a ``SessionOptions``
configuration — plus an OPTIONAL real in-memory inference if the ``onnx`` model
builder happens to be installed (skipped cleanly otherwise). A onnxruntime bump
that removed/renamed any of it, dropped the CPU execution provider, or changed
the ``InferenceSession`` constructor shape fails loudly here.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks +
offline behavioural smoke (no model downloads, no network, CPU only). Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "onnxruntime"
DIST_NAME = "onnxruntime"

# Core public names consumers (chromadb/fastembed/rapidocr) resolve on ort.
USED_SYMBOLS = [
    "InferenceSession",
    "SessionOptions",
    "RunOptions",
    "GraphOptimizationLevel",
    "OrtValue",
    "get_available_providers",
    "get_all_providers",
    "get_device",
]

# Methods consumers call on a constructed InferenceSession.
SESSION_METHODS = ["run", "get_inputs", "get_outputs", "get_providers", "get_modelmeta"]

# GraphOptimizationLevel enum members tuning code selects from.
OPT_LEVELS = ["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"]


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "onnxruntime"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Every core onnxruntime symbol consumers reach for must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_inference_session_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.InferenceSession)


def test_inference_session_constructor_signature(depcheck):
    """Consumers construct ``InferenceSession(model, sess_options=,
    providers=)``. Pin those parameter names so the call shape stays valid."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.InferenceSession.__init__,
        ["path_or_bytes", "sess_options", "providers"],
    )


def test_inference_session_method_surface(depcheck):
    """The methods consumers call on a session must exist on the class
    (checked without instantiating — instantiation needs a model)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.InferenceSession))
    missing = [m for m in SESSION_METHODS if m not in names]
    assert not missing, f"InferenceSession missing method(s): {missing}"


# --------------------------------------------------------------------------- #
# Execution providers — CPU must always be available
# --------------------------------------------------------------------------- #
def test_get_available_providers_includes_cpu(depcheck):
    """``get_available_providers()`` returns the providers usable on this host;
    ``CPUExecutionProvider`` must always be present — it is the fallback every
    consumer relies on when no GPU provider is available."""
    mod = depcheck.load(IMPORT_NAME)
    providers = mod.get_available_providers()
    assert isinstance(providers, list)
    assert "CPUExecutionProvider" in providers, f"CPUExecutionProvider not available: {providers}"


def test_get_all_providers_superset(depcheck):
    """``get_all_providers()`` lists every provider onnxruntime knows about and
    must be a superset of the available ones (and include CPU)."""
    mod = depcheck.load(IMPORT_NAME)
    all_providers = set(mod.get_all_providers())
    available = set(mod.get_available_providers())
    assert "CPUExecutionProvider" in all_providers
    assert available.issubset(all_providers)


def test_get_device_returns_str(depcheck):
    """``get_device()`` returns the device string (e.g. 'CPU'); pin it stays a
    non-empty str."""
    mod = depcheck.load(IMPORT_NAME)
    device = mod.get_device()
    assert isinstance(device, str)
    assert device


# --------------------------------------------------------------------------- #
# SessionOptions — the tuning surface consumers configure
# --------------------------------------------------------------------------- #
def test_session_options_constructs(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    so = mod.SessionOptions()
    assert so is not None


def test_session_options_tunables_settable(depcheck):
    """Consumers set graph_optimization_level / intra_op_num_threads /
    inter_op_num_threads on SessionOptions. Pin those attributes accept
    assignment (round-trip the value)."""
    mod = depcheck.load(IMPORT_NAME)
    so = mod.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    so.graph_optimization_level = mod.GraphOptimizationLevel.ORT_ENABLE_ALL
    assert so.intra_op_num_threads == 2
    assert so.inter_op_num_threads == 1
    assert so.graph_optimization_level == mod.GraphOptimizationLevel.ORT_ENABLE_ALL


def test_graph_optimization_level_members(depcheck):
    """The optimisation-level enum members consumers select must all exist."""
    mod = depcheck.load(IMPORT_NAME)
    level_enum = mod.GraphOptimizationLevel
    for name in OPT_LEVELS:
        assert hasattr(level_enum, name), f"GraphOptimizationLevel.{name} missing"


# --------------------------------------------------------------------------- #
# OrtValue — model-free tensor round-trip (real ORT object, no inference)
# --------------------------------------------------------------------------- #
def test_ortvalue_from_numpy_roundtrip(depcheck):
    """``OrtValue.ortvalue_from_numpy(arr)`` wraps a numpy array into an ORT
    tensor without any model. The wrapped value must report the array's shape +
    dtype and round-trip back via ``.numpy()`` byte-for-byte. This exercises
    ORT's tensor plumbing — the exact data hand-off consumers make to run() —
    with no model download.
    """
    mod = depcheck.load(IMPORT_NAME)
    np = depcheck.load("numpy")

    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    ov = mod.OrtValue.ortvalue_from_numpy(arr)
    assert list(ov.shape()) == [2, 3]
    assert "float" in ov.data_type()  # 'tensor(float)'
    back = ov.numpy()
    assert back.shape == (2, 3)
    assert np.array_equal(back, arr)


def test_ortvalue_on_cpu_device(depcheck):
    """An OrtValue built from a numpy array lives on the CPU device — the
    default placement consumers run inference against."""
    mod = depcheck.load(IMPORT_NAME)
    np = depcheck.load("numpy")
    ov = mod.OrtValue.ortvalue_from_numpy(np.zeros((1, 4), dtype=np.float32))
    assert ov.device_name().lower() == "cpu"


# --------------------------------------------------------------------------- #
# OPTIONAL: a real end-to-end inference on a tiny in-memory model, only if the
# `onnx` model builder is installed. Skips cleanly when it is not (no download).
# --------------------------------------------------------------------------- #
def test_inference_session_runs_inmemory_model_if_onnx_present(depcheck):
    """If the ``onnx`` package is available, build the smallest valid model in
    memory (Y = X + X), construct an ``InferenceSession`` from the serialized
    bytes with the CPU provider, and run a real inference — proving the
    construct -> run path the consumers depend on. No model is downloaded; the
    graph is generated in-process. Skipped if ``onnx`` is not installed (the
    surface tests above still cover the contract).
    """
    mod = depcheck.load(IMPORT_NAME)
    np = depcheck.load("numpy")
    onnx = depcheck.try_load("onnx")
    if onnx is None:
        pytest.skip("onnx model builder not installed; surface tests cover the contract")

    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])
    node = helper.make_node("Add", ["X", "X"], ["Y"])
    graph = helper.make_graph([node], "doubler", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    blob = model.SerializeToString()

    session = mod.InferenceSession(blob, providers=["CPUExecutionProvider"])
    # IO introspection consumers use to map inputs/outputs.
    assert session.get_inputs()[0].name == "X"
    assert session.get_outputs()[0].name == "Y"
    assert "CPUExecutionProvider" in session.get_providers()

    out = session.run(None, {"X": np.array([3.0, 4.0], dtype=np.float32)})
    assert out[0].tolist() == [6.0, 8.0]  # X + X
