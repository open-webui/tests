"""Document-ingestion regressions fixed in v0.11.0 that no endpoint can see.

- Text recognition package (PR #26851, issues #26646/#26994): `rapidocr-onnxruntime==1.4.4` was
  unresolvable, so the image-text extraction dependency could not be installed. Repinned to
  `rapidocr`. A packaging guard over requirements.txt.
- Milvus (PR #26911, `f4a6ea930`; PR #27521, `a15e44a5f`, issue #26978): the ORM-style PyMilvus
  API (`Collection`, `connections`, `utility`) was replaced with `MilvusClient`, and the
  `resource_id` scalar index now falls back to an explicit INVERTED index and never fails
  collection creation, which is what broke on embedded Milvus Lite. Seeing it needs a Milvus
  server, so the pymilvus client is the stubbed boundary; its index and schema builders stay real.

The upload, splitting, routing, knowledge base and embedding-prefix parts of this file moved to
integration/retrieval/test_document_ingestion.py.

Discriminates: passes on dev bbfa876af; fails with the old OCR pin back in requirements.txt, with
the INVERTED fallback removed (the server's index error escapes collection creation) and with an
ORM name imported from pymilvus again.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import create_autospec

import pytest

pytestmark = pytest.mark.regression


def _source(backend: Path, relative: str) -> str:
    return (backend / relative).read_text(encoding="utf-8")


# =============================================================================
# Text recognition package pin (PR #26851)
# =============================================================================


def _requirement_names(requirements: str) -> dict[str, str]:
    pins = {}
    for raw_line in requirements.splitlines():
        line = raw_line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line.split("==")[0].split(">=")[0].split("[")[0].strip().lower()
        pins[name] = line
    return pins


def test_ocr_dependency_uses_the_installable_rapidocr_distribution(open_webui_backend):
    pins = _requirement_names(_source(open_webui_backend, "requirements.txt"))
    assert "rapidocr-onnxruntime" not in pins, "the unresolvable OCR pin is back"
    assert "rapidocr" in pins, "no OCR distribution pinned"
    assert "==" in pins["rapidocr"], "OCR distribution is not pinned to a version"


def test_requirements_still_pin_the_rest_of_the_image_stack(open_webui_backend):
    pins = _requirement_names(_source(open_webui_backend, "requirements.txt"))
    for package in ("pillow", "opencv-python-headless", "onnxruntime"):
        assert package in pins, f"{package} missing from requirements"


# =============================================================================
# Milvus (PR #26911, PR #27521)
# =============================================================================

RESOURCE_ID_INDEX = "resource_id"
ITEM = {"id": "doc-1", "text": "hello", "vector": [0.1, 0.2, 0.3, 0.4], "metadata": {}}


@pytest.fixture
def milvus(monkeypatch, owui_module):
    """The multitenancy wrapper over a stubbed pymilvus client with no collections yet."""
    pymilvus = pytest.importorskip("pymilvus", reason="pymilvus not installed in this env")
    module = owui_module("open_webui.retrieval.vector.dbs.milvus_multitenancy")
    server = create_autospec(pymilvus.MilvusClient, instance=True)
    server.has_collection.return_value = False
    server.create_schema.side_effect = pymilvus.MilvusClient.create_schema
    server.prepare_index_params.side_effect = pymilvus.MilvusClient.prepare_index_params
    monkeypatch.setattr(module, "Client", lambda **kwargs: server)
    return module.MilvusClient(), server


def _index_requests(server) -> list[tuple[str, str]]:
    return [
        (index.field_name, index.index_type)
        for request in server.create_index.call_args_list
        for index in request.kwargs["index_params"]
    ]


def test_a_new_collection_is_created_through_the_milvus_client(milvus):
    client, server = milvus

    client.upsert(collection_name="some-knowledge", items=[ITEM])

    server.create_collection.assert_called_once()
    assert (RESOURCE_ID_INDEX, "") in _index_requests(server)
    server.insert.assert_called_once()


def test_a_rejected_scalar_index_falls_back_to_inverted(milvus):
    from pymilvus.exceptions import MilvusException

    client, server = milvus

    def reject_an_untyped_scalar_index(collection_name, index_params, **kwargs):
        if any(
            index.field_name == RESOURCE_ID_INDEX and not index.index_type for index in index_params
        ):
            raise MilvusException(message="index type not supported")

    server.create_index.side_effect = reject_an_untyped_scalar_index

    client.upsert(collection_name="some-knowledge", items=[ITEM])

    assert (RESOURCE_ID_INDEX, "INVERTED") in _index_requests(server), (
        "no INVERTED fallback for the resource_id index"
    )
    server.insert.assert_called_once()


@pytest.mark.parametrize(
    "relative",
    [
        "open_webui/retrieval/vector/dbs/milvus.py",
        "open_webui/retrieval/vector/dbs/milvus_multitenancy.py",
    ],
)
def test_milvus_modules_do_not_import_the_deprecated_orm_api(open_webui_backend, relative):
    source = _source(open_webui_backend, relative)
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pymilvus")
        for alias in node.names
    }
    assert imported, f"{relative} no longer imports pymilvus; retarget this audit"
    deprecated = imported & {
        "Collection",
        "CollectionSchema",
        "FieldSchema",
        "connections",
        "utility",
    }
    assert deprecated == set(), (
        f"{relative} still imports the deprecated ORM API: {sorted(deprecated)}"
    )
