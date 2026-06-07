"""Dependency contract: googleapis-common-protos.

``googleapis-common-protos`` ships the *compiled* common Protocol Buffer
message types that the wider Google/gRPC ecosystem builds on — the
``google.rpc`` (Status / Code / error-details), ``google.api`` (HTTP
annotations), ``google.longrunning`` (Operations) and ``google.type``
families. It is a *transitive* dependency of the Open WebUI backend, pulled
in by gRPC-based clients (e.g. the ChromaDB / vector-store stack and
anything speaking a Google API surface). The application code does not
import it directly, and there is no stable internal chokepoint, so this
module pins the *core public surface* — the canonical ``*_pb2`` modules
and the ``Status`` round-trip — so a breaking bump (these protos being
removed, renamed, or recompiled incompatibly) is caught here.

The package contributes to the implicit ``google`` *namespace* package, so
the contract is expressed as submodule imports (``from google.rpc import
status_pb2``) plus an offline protobuf serialise/parse round-trip. Pure
in-memory protobuf — no network, no gRPC channel.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

# Import name is the implicit `google` namespace; the distribution is
# googleapis-common-protos. We pin the package via its canonical submodules.
IMPORT_NAME = "google"
DIST_NAME = "googleapis-common-protos"

# Canonical compiled-proto modules this distribution provides.
PROTO_MODULES = [
    "google.rpc.status_pb2",  # Status (code, message, details) — gRPC errors
    "google.rpc.code_pb2",  # canonical Code enum (OK, NOT_FOUND, ...)
    "google.rpc.error_details_pb2",  # rich error detail messages
    "google.api.http_pb2",  # HttpRule (REST<->gRPC transcoding)
    "google.api.annotations_pb2",  # method http annotations
    "google.longrunning.operations_pb2",  # Operation / long-running ops
    "google.type.latlng_pb2",  # common scalar types
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_namespace_imports(depcheck):
    """The `google` namespace package must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "google"


def test_version_reported(depcheck):
    """The installed distribution (googleapis-common-protos) version must
    resolve — this is how we identify the package behind the shared namespace."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Submodule import checks — the compiled protos this distribution provides.
# ---------------------------------------------------------------------------


def test_proto_submodules_import(depcheck):
    """Every canonical *_pb2 module the ecosystem depends on must import."""
    for name in PROTO_MODULES:
        mod = depcheck.load(name)
        assert mod.__name__ == name


def test_rpc_status_message_present(depcheck):
    """google.rpc.status_pb2.Status is the message used to carry gRPC error
    status across the wire; pin the class exists."""
    mod = depcheck.load("google.rpc.status_pb2")
    assert hasattr(mod, "Status")


def test_rpc_code_enum_present(depcheck):
    """google.rpc.code_pb2 defines the canonical status Code enum values."""
    mod = depcheck.load("google.rpc.code_pb2")
    for name in ("OK", "NOT_FOUND", "INVALID_ARGUMENT", "UNAVAILABLE"):
        assert hasattr(mod, name), f"google.rpc.code_pb2.{name} missing"


def test_longrunning_operation_present(depcheck):
    """google.longrunning.operations_pb2.Operation backs long-running API ops."""
    mod = depcheck.load("google.longrunning.operations_pb2")
    assert hasattr(mod, "Operation")


def test_api_http_rule_present(depcheck):
    """google.api.http_pb2.HttpRule drives REST<->gRPC transcoding annotations."""
    mod = depcheck.load("google.api.http_pb2")
    assert hasattr(mod, "HttpRule")


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — real protobuf serialise/parse round-trips.
# ---------------------------------------------------------------------------


def test_behaviour_status_roundtrip(depcheck):
    """Build a Status, serialise to bytes, parse it back — the exact lifecycle
    gRPC uses to ship error status. Fields must survive the round-trip."""
    mod = depcheck.load("google.rpc.status_pb2")
    status = mod.Status(code=5, message="Not Found")
    payload = status.SerializeToString()
    assert isinstance(payload, bytes) and payload

    parsed = mod.Status()
    parsed.ParseFromString(payload)
    assert parsed.code == 5
    assert parsed.message == "Not Found"


def test_behaviour_code_enum_values(depcheck):
    """The canonical Code enum values are fixed by the gRPC spec (OK=0,
    NOT_FOUND=5); pin them so a recompile can't silently renumber."""
    mod = depcheck.load("google.rpc.code_pb2")
    assert mod.OK == 0
    assert mod.NOT_FOUND == 5
    assert mod.INVALID_ARGUMENT == 3


def test_behaviour_latlng_roundtrip(depcheck):
    """google.type.LatLng is a representative scalar message; round-trip its
    float fields to prove the type-family protos serialise correctly."""
    mod = depcheck.load("google.type.latlng_pb2")
    ll = mod.LatLng(latitude=37.4219, longitude=-122.0841)
    payload = ll.SerializeToString()
    parsed = mod.LatLng()
    parsed.ParseFromString(payload)
    assert abs(parsed.latitude - 37.4219) < 1e-6
    assert abs(parsed.longitude + 122.0841) < 1e-6


def test_behaviour_operation_message_constructs(depcheck):
    """An Operation with a name and done flag must construct + round-trip — the
    long-running-operation envelope other clients depend on."""
    mod = depcheck.load("google.longrunning.operations_pb2")
    op = mod.Operation(name="operations/abc123", done=True)
    payload = op.SerializeToString()
    parsed = mod.Operation()
    parsed.ParseFromString(payload)
    assert parsed.name == "operations/abc123"
    assert parsed.done is True


def test_behaviour_protobuf_runtime_available(depcheck):
    """These compiled protos require the protobuf runtime; confirm it is present
    (a missing/incompatible runtime is the usual failure mode of a bad bump)."""
    pb = depcheck.try_load("google.protobuf")
    if pb is None:
        pytest.skip("google.protobuf runtime not importable; proto tests cover usage")
    # The descriptor machinery the *_pb2 modules build against must exist.
    assert depcheck.has(pb, "message.Message")
