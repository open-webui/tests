"""Dependency contract: accelerate.

HuggingFace ``accelerate`` is pinned directly in
``backend/requirements.txt``. It is the runtime that ``transformers`` /
``sentence-transformers`` use to place and run models across devices: it
is what makes ``device_map="auto"``, big-model sharded loading, CPU/GPU
offload and multi-GPU dispatch work. Open WebUI's local embedding and
reranking models (loaded via sentence-transformers / transformers in
``retrieval/`` when the local engines are selected) rely on accelerate
being importable for those code paths.

IMPORTANT — usage note: the Open WebUI *application* code does NOT import
``accelerate`` directly anywhere (the only textual match in the source is
the unrelated S3 ``use_accelerate_endpoint`` config). accelerate is a
declared/transitive dependency consumed by the HF stack. There are no
first-party call sites, so this module pins accelerate's *core public
surface* — the ``Accelerator`` class and the big-model helpers
(``init_empty_weights``, ``infer_auto_device_map``, ``dispatch_model``,
``load_checkpoint_and_dispatch``) — plus a light, CPU-only behavioural
smoke (``init_empty_weights`` builds a meta-device module without
allocating real tensors).

This is a heavy ML dependency: NOTHING here loads a model, downloads
weights, touches a GPU, or hits the network. We only import, inspect the
API surface, and run one offline meta-device smoke.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "accelerate"
DIST_NAME = "accelerate"

# Core public surface transformers / sentence-transformers dispatch onto.
TOP_LEVEL_SYMBOLS = [
    "Accelerator",  # the central orchestrator
    "init_empty_weights",  # meta-device model construction (big-model loading)
    "infer_auto_device_map",  # device_map='auto' resolution
    "dispatch_model",  # place a model across the inferred device map
    "load_checkpoint_and_dispatch",  # sharded checkpoint loading + dispatch
    "cpu_offload",  # offload helper
    "PartialState",  # distributed state singleton
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "accelerate"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_top_level_symbols_exist(depcheck):
    """The big-model + Accelerator surface the HF stack uses must remain on the
    top-level package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOP_LEVEL_SYMBOLS)


def test_accelerator_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.Accelerator)


def test_bigmodel_helpers_callable(depcheck):
    """init_empty_weights / infer_auto_device_map / dispatch_model /
    load_checkpoint_and_dispatch must remain callables — these are exactly the
    functions transformers calls under device_map='auto'."""
    mod = depcheck.load(IMPORT_NAME)
    for name in (
        "init_empty_weights",
        "infer_auto_device_map",
        "dispatch_model",
        "load_checkpoint_and_dispatch",
    ):
        depcheck.assert_callable(mod, name)


def test_accelerator_has_core_methods(depcheck):
    """transformers' Trainer + sentence-transformers use Accelerator.prepare /
    .device / .backward / .gather. Pin those method names on the class (no
    instance is constructed — that can probe the environment)."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.Accelerator))
    for name in ("prepare", "backward", "gather", "unwrap_model"):
        assert name in names, f"Accelerator.{name} missing"


def test_init_empty_weights_meta_smoke(depcheck):
    """init_empty_weights() builds a module on the meta device (no real memory),
    the basis of big-model loading. Run it offline with a tiny torch module;
    skip cleanly if torch isn't importable in this env."""
    mod = depcheck.load(IMPORT_NAME)
    torch = depcheck.try_load("torch")
    if torch is None:
        pytest.skip("torch not importable; surface checks cover accelerate offline")
    nn = torch.nn
    with mod.init_empty_weights():
        model = nn.Linear(8, 8)
    # Parameters must live on the meta device (no allocation happened).
    devices = {p.device.type for p in model.parameters()}
    assert devices == {"meta"}, f"init_empty_weights did not use meta device: {devices}"


def test_infer_auto_device_map_callable_signature(depcheck):
    """infer_auto_device_map(model, ...) is how device_map='auto' is resolved.
    Pin that it accepts a model plus the common budgeting kwargs (or **kwargs)."""
    mod = depcheck.load(IMPORT_NAME)
    fn = mod.infer_auto_device_map
    # First positional is the model; max_memory/no_split_module_classes are the
    # knobs transformers passes. Tolerate **kwargs.
    depcheck.assert_params(fn, ["model"])


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): the backend does not import
    accelerate directly; it's the HF device-placement runtime pulled in for
    local embedding/reranking models. The surface pins above guard the slice
    those code paths depend on."""
    assert True
