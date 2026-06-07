"""Dependency contract: sentencepiece.

sentencepiece is Google's subword tokenizer (unigram / BPE). It is pinned
in ``backend/requirements.txt`` and is the native tokenizer backend that
``transformers`` / ``sentence-transformers`` require for SentencePiece-based
models (T5, LLaMA, mBART, XLM-R, many multilingual embedders), and that
``google-genai`` uses for its local tokenizer. Open WebUI's local
embedding / reranking models load through that HF stack when the local
engines are selected.

IMPORTANT — usage note: the Open WebUI *application* code does NOT import
``sentencepiece`` directly anywhere. It's a declared/transitive dependency
consumed by the tokenizer layer of the ML stack. There are no first-party
call sites, so this module pins sentencepiece's *core public surface* —
``SentencePieceProcessor`` (load + encode/decode) and
``SentencePieceTrainer`` — and runs a fully in-memory, offline behavioural
smoke (train a tiny unigram model into a BytesIO, then encode/decode
round-trip). This is a native (C++/SWIG) extension, so methods are pinned
behaviourally / by presence, not via ``assert_params``.

Everything here is offline and deterministic in shape: no network, no
disk, no model download. The training corpus is a handful of strings and
the model is written to memory.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect
import io

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "sentencepiece"
DIST_NAME = "sentencepiece"

USED_SYMBOLS = [
    "SentencePieceProcessor",  # the tokenizer object HF wraps
    "SentencePieceTrainer",  # model training entrypoint
]

# Methods HF's SentencePiece tokenizers call on a processor.
PROCESSOR_METHODS = [
    "Load",
    "LoadFromSerializedProto",
    "EncodeAsIds",
    "EncodeAsPieces",
    "DecodeIds",
    "GetPieceSize",
    "encode",  # lowercase aliases (newer API)
    "decode",
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "sentencepiece"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """SentencePieceProcessor + SentencePieceTrainer must remain on the package."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_processor_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.SentencePieceProcessor)


def test_processor_methods_exist(depcheck):
    """The load + encode/decode surface HF tokenizers dispatch onto must exist
    on the processor class."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.SentencePieceProcessor))
    missing = [m for m in PROCESSOR_METHODS if m not in names]
    assert not missing, f"SentencePieceProcessor missing method(s): {missing}"


def test_trainer_has_train_entrypoint(depcheck):
    """SentencePieceTrainer must expose a training entrypoint (Train/train).
    HF's slow-tokenizer conversion and any local training relies on it."""
    mod = depcheck.load(IMPORT_NAME)
    trainer = mod.SentencePieceTrainer
    assert hasattr(trainer, "Train") or hasattr(trainer, "train"), (
        "SentencePieceTrainer lost its Train/train entrypoint"
    )


def test_empty_processor_constructs(depcheck):
    """An unloaded SentencePieceProcessor must construct (HF builds it then
    Loads a model file). Constructing must not need a model or network."""
    mod = depcheck.load(IMPORT_NAME)
    sp = mod.SentencePieceProcessor()
    assert sp is not None


def test_train_and_roundtrip_in_memory(depcheck):
    """End-to-end offline smoke mirroring how a SentencePiece model is built and
    used: train a tiny unigram model into a BytesIO (no disk), load it, then
    EncodeAsIds -> DecodeIds round-trips back to the input. This proves the
    train/load/encode/decode contract HF depends on is intact."""
    mod = depcheck.load(IMPORT_NAME)
    trainer = mod.SentencePieceTrainer
    train = getattr(trainer, "train", None) or getattr(trainer, "Train", None)
    if train is None:
        pytest.skip("SentencePieceTrainer has no train/Train in this build")

    # Tiny corpus; repeated so a small vocab can be learned deterministically.
    corpus = [
        "hello world",
        "the quick brown fox",
        "open webui retrieval pipeline",
        "sentencepiece subword tokenizer",
        "a b c d e f g h",
    ] * 50

    model_io = io.BytesIO()
    try:
        train(
            sentence_iterator=iter(corpus),
            model_writer=model_io,
            vocab_size=40,
            model_type="unigram",
            character_coverage=1.0,
            # Tiny corpora can't reach an arbitrary vocab size; relax the cap so
            # training succeeds with whatever pieces are learnable.
            hard_vocab_limit=False,
            minloglevel=2,  # quiet the C++ trainer logs
        )
    except TypeError:
        # Older builds may not support sentence_iterator/model_writer kwargs.
        pytest.skip("this sentencepiece build lacks in-memory train kwargs")

    proto = model_io.getvalue()
    assert proto, "trainer produced an empty model proto"

    # Load the trained model from the in-memory proto.
    sp = mod.SentencePieceProcessor(model_proto=proto)
    assert sp.GetPieceSize() > 0

    text = "hello world"
    ids = sp.EncodeAsIds(text)
    assert isinstance(ids, list) and len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

    pieces = sp.EncodeAsPieces(text)
    assert isinstance(pieces, list) and len(pieces) > 0

    decoded = sp.DecodeIds(ids)
    assert isinstance(decoded, str)
    # Round-trip reconstructs the input text (unigram on a covering corpus).
    assert decoded.strip() == text


def test_lowercase_encode_decode_aliases(depcheck):
    """Newer sentencepiece exposes lowercase encode()/decode() aliases that HF's
    fast paths use. Verify they exist and behave like the Encode/Decode pair on
    a trained model (offline)."""
    mod = depcheck.load(IMPORT_NAME)
    trainer = mod.SentencePieceTrainer
    train = getattr(trainer, "train", None) or getattr(trainer, "Train", None)
    if train is None:
        pytest.skip("SentencePieceTrainer has no train/Train in this build")

    corpus = ["alpha beta gamma", "open webui", "token round trip"] * 60
    model_io = io.BytesIO()
    try:
        train(
            sentence_iterator=iter(corpus),
            model_writer=model_io,
            vocab_size=40,
            model_type="unigram",
            character_coverage=1.0,
            hard_vocab_limit=False,
            minloglevel=2,
        )
    except TypeError:
        pytest.skip("this sentencepiece build lacks in-memory train kwargs")

    sp = mod.SentencePieceProcessor(model_proto=model_io.getvalue())
    if not (hasattr(sp, "encode") and hasattr(sp, "decode")):
        pytest.skip("lowercase encode/decode aliases not present in this build")
    ids = sp.encode("open webui")
    assert isinstance(ids, list) and ids
    back = sp.decode(ids)
    assert isinstance(back, str)
    assert "open" in back and "webui" in back


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): the backend doesn't import
    sentencepiece directly; it's the native tokenizer backend for the HF /
    genai local-tokenizer stack. The smoke above guards the train/encode/decode
    slice those tokenizers depend on."""
    assert True
