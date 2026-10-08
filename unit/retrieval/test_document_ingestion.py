"""Document-ingestion guards fixed in v0.11.0 that no endpoint can see.

- Text recognition package (PR #26851, issues #26646/#26994): `rapidocr-onnxruntime==1.4.4` was
  unresolvable, so the image-text extraction dependency could not be installed. Repinned to
  `rapidocr`. A packaging guard over requirements.txt.
- Milvus (PR #26911, `f4a6ea930`): the ORM-style PyMilvus API (`Collection`, `connections`,
  `utility`) was replaced with `MilvusClient`. The ORM API still works against a server, so no
  request tells the two apart; this audit keeps it out of both Milvus stores. `CollectionSchema`
  stays allowed: `MilvusClient.create_schema()` returns one, and PR #31660 names it as a type.

The Milvus stores themselves, including the INVERTED fallback for a refused `resource_id` index
(PR #27521, issue #26978), are driven against a Milvus stand-in in
integration/retrieval/test_milvus_store.py. The upload, splitting, routing, knowledge base and
embedding-prefix parts of this file moved to integration/retrieval/test_document_ingestion.py.

Discriminates: passes on dev b5a20423e; fails with the old OCR pin back in requirements.txt and
with an ORM name (`Collection`) imported from pymilvus again.
"""

from __future__ import annotations

import ast
from pathlib import Path

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
        "FieldSchema",
        "connections",
        "utility",
    }
    assert deprecated == set(), (
        f"{relative} still imports the deprecated ORM API: {sorted(deprecated)}"
    )
