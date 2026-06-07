"""Dependency contract: pycrdt (imported as ``Y``).

pycrdt is the Rust-backed Yjs CRDT binding that powers Open WebUI's
collaborative-document (Yjs) editing over WebSockets. Both
``socket/utils.py`` (the ``YdocManager`` update store/compaction) and
``socket/main.py`` (the ``ydoc:document:join`` / ``ydoc:document:state``
handlers) do ``import pycrdt as Y`` and use a tiny but load-bearing slice:

    ydoc = Y.Doc()
    ydoc.apply_update(bytes(update))     # replay each stored binary update
    snapshot = ydoc.get_update()         # encode whole-doc state as one update

Server-side the document content is opaque: the backend never inspects the
shared types, it only *replays* the stream of client-produced binary
updates into a fresh ``Doc`` and re-encodes the merged state to broadcast
to peers (``socket/main.py``) or to compact the Redis/in-memory update log
(``YdocManager._compact_updates_*``). So the contract that actually
matters is:

  * ``Y.Doc()`` constructs with no args,
  * ``Doc.apply_update`` accepts a ``bytes`` update and merges it,
  * ``Doc.get_update`` returns a ``bytes`` update that, replayed into
    another fresh ``Doc``, reproduces the merged state (round-trip /
    associativity of update merging).

This module pins exactly that. pycrdt is a compiled (maturin/Rust)
extension, so its methods have no introspectable Python signatures; we
verify the contract *behaviourally* (offline, no network, no I/O) rather
than via ``assert_params``.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "pycrdt"
DIST_NAME = "pycrdt"

# Symbols the backend resolves on the package.
USED_SYMBOLS = [
    "Doc",  # Y.Doc() in socket/utils.py + socket/main.py
]

# Broader public surface a Yjs binding is expected to keep (used indirectly
# by clients producing the updates the server replays; pinned so a major
# pycrdt reshuffle that drops the core shared types is caught early).
CORE_PUBLIC_TYPES = ["Doc", "Text", "Array", "Map"]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "pycrdt"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_used_symbols_exist(depcheck):
    """Doc must remain importable off the top-level package (the only name the
    backend resolves on pycrdt)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_core_public_types_present(depcheck):
    """The standard Yjs shared types remain on the package. The server doesn't
    use Text/Array/Map directly, but clients do, and dropping them would signal
    a breaking pycrdt rewrite that could also move Doc."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod))
    missing = [t for t in CORE_PUBLIC_TYPES if t not in names]
    assert not missing, f"pycrdt missing core public type(s): {missing}"


def test_doc_constructs_no_args(depcheck):
    """socket/* always do `Y.Doc()` with no arguments."""
    mod = depcheck.load(IMPORT_NAME)
    doc = mod.Doc()
    assert doc is not None


def test_doc_has_used_methods(depcheck):
    """apply_update + get_update are the only two Doc methods the backend calls.
    They must exist and be callable (compiled methods => check on the instance,
    not via signature)."""
    mod = depcheck.load(IMPORT_NAME)
    doc = mod.Doc()
    for name in ("apply_update", "get_update"):
        attr = getattr(doc, name, None)
        assert attr is not None, f"Doc.{name} missing"
        assert callable(attr), f"Doc.{name} is not callable"


def test_get_update_returns_bytes(depcheck):
    """socket/main.py does `state_update = ydoc.get_update()` then
    `list(state_update)` to JSON-serialise it, and YdocManager does
    `json.dumps(list(ydoc.get_update()))`. Both require get_update() to return
    a bytes-like object iterable into ints (0..255)."""
    mod = depcheck.load(IMPORT_NAME)
    doc = mod.Doc()
    update = doc.get_update()
    assert isinstance(update, (bytes, bytearray)), (
        f"Doc.get_update() returned {type(update)!r}, expected bytes"
    )
    # list(...) of it must be JSON-serialisable ints, matching the backend.
    as_list = list(update)
    assert all(isinstance(b, int) and 0 <= b <= 255 for b in as_list)
    # Empty doc still yields a valid (possibly empty) update payload.
    json.dumps(as_list)  # must not raise


def test_apply_update_accepts_bytes(depcheck):
    """The backend always passes `bytes(update)` to apply_update. Feeding the
    bytes produced by another doc's get_update() must be accepted (the binary
    Yjs update format is the interchange contract)."""
    mod = depcheck.load(IMPORT_NAME)
    src = mod.Doc()
    update = src.get_update()
    dst = mod.Doc()
    # Must accept a bytes update without raising.
    dst.apply_update(bytes(update))


def test_apply_update_then_get_update_roundtrip(depcheck):
    """Core server contract: replay a producer doc's updates into a fresh doc
    and re-encode. We mutate a doc via a shared Array (what a Yjs client would
    do), capture its update, replay it into a fresh Doc (exactly what
    socket/main.py does), and assert the merged state re-encodes to a non-empty
    update — i.e. the content survived the binary round-trip."""
    mod = depcheck.load(IMPORT_NAME)
    producer = mod.Doc()
    # Attach a shared Array and push some entries inside a transaction.
    arr = mod.Array()
    producer["shared"] = arr
    with producer.transaction():
        arr.append("a")
        arr.append("b")
        arr.append("c")
    producer_update = producer.get_update()
    assert isinstance(producer_update, (bytes, bytearray))
    assert len(producer_update) > 0, "non-trivial doc produced an empty update"

    # Replay into a fresh consumer doc, exactly as the server does.
    consumer = mod.Doc()
    consumer.apply_update(bytes(producer_update))
    merged = consumer.get_update()
    assert isinstance(merged, (bytes, bytearray))
    # The merged state must itself encode the content (non-empty), so a third
    # peer replaying `merged` would see the same document.
    assert len(merged) > 0


def test_merge_multiple_updates_like_compaction(depcheck):
    """YdocManager._compact_updates_* replays a *list* of stored updates into
    one Doc and re-encodes a single snapshot. Verify that applying several
    sequential updates accumulates state (associativity of merge), so the
    compacted snapshot represents all of them."""
    mod = depcheck.load(IMPORT_NAME)

    # Produce three independent binary updates from one evolving producer doc,
    # capturing the *incremental* update after each change (a realistic Yjs
    # update stream: each is a diff against the prior state vector).
    producer = mod.Doc()
    arr = mod.Array()
    producer["log"] = arr

    updates = []
    prev_state = producer.get_state()
    for token in ("u1", "u2", "u3"):
        with producer.transaction():
            arr.append(token)
        diff = producer.get_update(prev_state)
        updates.append(bytes(diff))
        prev_state = producer.get_state()

    # Compaction: replay the whole list into one fresh doc (server behaviour).
    compacted_doc = mod.Doc()
    for raw in updates:
        compacted_doc.apply_update(raw)
    snapshot = compacted_doc.get_update()
    assert isinstance(snapshot, (bytes, bytearray))
    assert len(snapshot) > 0

    # A peer replaying just the single snapshot must reconstruct all entries.
    peer = mod.Doc()
    peer.apply_update(bytes(snapshot))
    peer_arr = peer.get("log", type=mod.Array)
    assert list(peer_arr) == ["u1", "u2", "u3"]


def test_apply_update_is_idempotent_for_same_update(depcheck):
    """Yjs updates are idempotent: re-applying an already-merged update must not
    duplicate content. The server can replay overlapping updates from the
    Redis log, so this property keeps compaction from corrupting state."""
    mod = depcheck.load(IMPORT_NAME)
    producer = mod.Doc()
    arr = mod.Array()
    producer["x"] = arr
    with producer.transaction():
        arr.append("only-once")
    update = bytes(producer.get_update())

    consumer = mod.Doc()
    consumer.apply_update(update)
    consumer.apply_update(update)  # apply the SAME update twice
    result = consumer.get("x", type=mod.Array)
    assert list(result) == ["only-once"], "re-applying an update duplicated state"


def test_get_state_exists_for_diffs(depcheck):
    """get_update(state_vector) is how incremental diffs are produced (the basis
    for the compaction stream above). Pin that get_state()/get_update(diff) is
    available — a regression here would break efficient update encoding."""
    mod = depcheck.load(IMPORT_NAME)
    doc = mod.Doc()
    assert callable(getattr(doc, "get_state", None)), "Doc.get_state missing"
    sv = doc.get_state()
    assert isinstance(sv, (bytes, bytearray))
    # get_update must accept the state vector to produce a diff.
    diff = doc.get_update(sv)
    assert isinstance(diff, (bytes, bytearray))
