"""Dependency contract: colbert-ai (import name ``colbert``).

colbert-ai provides the ColBERT late-interaction reranker. Open WebUI's
``retrieval/models/colbert.py`` wraps it: it imports ``ColBERTConfig`` from
``colbert.infra`` and ``Checkpoint`` from ``colbert.modeling.checkpoint``,
then in ``ColBERT.__init__`` builds

    self.ckpt = Checkpoint(name, colbert_config=ColBERTConfig(model_name=name)).to(self.device)

and in ``predict`` calls ``self.ckpt.docFromText(docs, bsize=batch_size)`` and
``self.ckpt.queryFromText([query], bsize=batch_size)`` to embed documents and
queries (the resulting multi-vector embeddings are scored with torch matmul +
max-pooling). This path is only active when the reranker is configured.

This module pins exactly that surface so a colbert-ai bump that renamed/moved
``ColBERTConfig`` / ``Checkpoint`` or changed the ``Checkpoint`` constructor or
the ``docFromText`` / ``queryFromText`` signatures fails loudly here instead of
as a runtime ``ImportError`` / ``TypeError`` the moment the ColBERT reranker is
enabled. Two layers, mirroring test_requests.py: import + symbol/signature
checks, plus the ONE behavioural contract that is safe offline —
``ColBERTConfig(model_name=name)`` (a lightweight config dataclass; it does NOT
download any model). We deliberately do NOT construct a ``Checkpoint`` or call
``docFromText`` / ``queryFromText``: those load model weights from the network
and run inference, which would violate the offline/deterministic rule. The
``Checkpoint`` constructor and embedding methods are pinned by SIGNATURE only.

colbert-ai is a heavy import (it pulls torch), so the import itself is the
costly step; everything here stays offline. Uses the ``depcheck`` fixture from
unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "colbert"
DIST_NAME = "colbert-ai"


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`colbert` must import (skip cleanly if absent in this env). NOTE the
    distribution name is `colbert-ai` while the import name is `colbert`."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "colbert"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling and
    this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (the exact import sites colbert.py uses)
# ---------------------------------------------------------------------------


def test_colbert_config_importable(depcheck):
    """colbert.py: `from colbert.infra import ColBERTConfig`."""
    depcheck.load(IMPORT_NAME)
    infra = depcheck.load("colbert.infra")
    assert hasattr(infra, "ColBERTConfig"), "colbert.infra.ColBERTConfig is gone"


def test_checkpoint_importable(depcheck):
    """colbert.py: `from colbert.modeling.checkpoint import Checkpoint`."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    assert hasattr(ckpt_mod, "Checkpoint"), "colbert.modeling.checkpoint.Checkpoint is gone"


def test_checkpoint_is_callable(depcheck):
    """Checkpoint(name, colbert_config=...) is the constructor colbert.py calls;
    it must be a callable class."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    assert callable(ckpt_mod.Checkpoint)


# ---------------------------------------------------------------------------
# Signature contracts (the exact kwargs/positionals colbert.py passes)
# ---------------------------------------------------------------------------


def test_colbert_config_accepts_model_name(depcheck):
    """colbert.py: ColBERTConfig(model_name=name). `model_name` must remain an
    accepted field/parameter."""
    depcheck.load(IMPORT_NAME)
    infra = depcheck.load("colbert.infra")
    cfg_cls = infra.ColBERTConfig
    # ColBERTConfig is a dataclass; model_name should be a declared field.
    fields = getattr(cfg_cls, "__dataclass_fields__", {})
    assert "model_name" in fields, (
        "ColBERTConfig no longer declares a `model_name` field; "
        "colbert.py's ColBERTConfig(model_name=name) would break."
    )


def test_checkpoint_init_accepts_name_and_colbert_config(depcheck):
    """colbert.py: Checkpoint(name, colbert_config=ColBERTConfig(...)). The
    positional `name` and the `colbert_config` kwarg must remain accepted."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    depcheck.assert_params(ckpt_mod.Checkpoint.__init__, ["name", "colbert_config"])


def test_checkpoint_is_torch_module_with_to(depcheck):
    """colbert.py chains `.to(self.device)` on the Checkpoint. That works because
    Checkpoint is a torch.nn.Module; pin the subclass relationship and the .to
    method so the device-placement call keeps working. Skips if torch is
    unavailable (colbert would not import either, so this is belt-and-suspenders)."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    torch = depcheck.try_load("torch")
    if torch is None:
        pytest.skip("torch not importable; colbert itself would not load")
    assert issubclass(ckpt_mod.Checkpoint, torch.nn.Module), (
        "Checkpoint is no longer a torch.nn.Module; `.to(device)` in colbert.py would fail."
    )
    assert hasattr(ckpt_mod.Checkpoint, "to") and callable(ckpt_mod.Checkpoint.to)


def test_doc_from_text_signature(depcheck):
    """colbert.py: self.ckpt.docFromText(docs, bsize=batch_size). The `docs`
    positional and the `bsize` kwarg must remain accepted (signature-only — we
    never call it, as it loads weights and runs inference)."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    assert hasattr(ckpt_mod.Checkpoint, "docFromText"), "Checkpoint.docFromText is gone"
    depcheck.assert_params(ckpt_mod.Checkpoint.docFromText, ["docs", "bsize"])


def test_query_from_text_signature(depcheck):
    """colbert.py: self.ckpt.queryFromText([query], bsize=batch_size). The
    `queries` positional and `bsize` kwarg must remain accepted (signature-only)."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    assert hasattr(ckpt_mod.Checkpoint, "queryFromText"), "Checkpoint.queryFromText is gone"
    depcheck.assert_params(ckpt_mod.Checkpoint.queryFromText, ["queries", "bsize"])


# ---------------------------------------------------------------------------
# Behavioural: ColBERTConfig construction (lightweight, NO model download).
# This is the only colbert call safe to run offline — it builds a config
# dataclass and touches no weights / no network.
# ---------------------------------------------------------------------------


def test_behaviour_colbert_config_constructs_with_model_name(depcheck):
    """ColBERTConfig(model_name=name) must construct offline and round-trip the
    model_name (this is the exact object colbert.py hands to Checkpoint). No
    weights are loaded — ColBERTConfig is pure configuration."""
    depcheck.load(IMPORT_NAME)
    infra = depcheck.load("colbert.infra")
    name = "colbert-ir/colbertv2.0"
    cfg = infra.ColBERTConfig(model_name=name)
    assert cfg is not None
    assert cfg.model_name == name, (
        "ColBERTConfig did not retain model_name; the config colbert.py builds "
        "would not carry the model identifier."
    )


def test_behaviour_checkpoint_not_constructed_here(depcheck):
    """Guard-rail documentation test: constructing a Checkpoint downloads model
    weights and is therefore intentionally NOT exercised here (offline rule).
    We only assert the constructor exists and is introspectable, so the
    signature contract above is meaningful."""
    depcheck.load(IMPORT_NAME)
    ckpt_mod = depcheck.load("colbert.modeling.checkpoint")
    # Introspectable signature == the contract tests can actually validate kwargs.
    sig = inspect.signature(ckpt_mod.Checkpoint.__init__)
    assert "name" in sig.parameters
