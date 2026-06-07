"""Dependency contract: unstructured (import name ``unstructured``).

``unstructured`` is an optional-but-pinned dependency of the Open WebUI
backend. The backend never imports it directly; instead
``retrieval/loaders/main.py`` dispatches a family of LangChain loaders that
wrap it — ``UnstructuredRSTLoader``, ``UnstructuredXMLLoader``,
``UnstructuredEPubLoader``, ``UnstructuredWordDocumentLoader`` (.doc),
``UnstructuredExcelLoader``, ``UnstructuredPowerPointLoader``,
``UnstructuredODTLoader`` — each guarded by a ``try/except ImportError``
that tells the operator to ``pip install unstructured``. Those loaders call
``unstructured``'s ``partition`` functions internally and convert the
returned ``Element`` objects into LangChain ``Document`` objects (one per
element, using ``str(element)`` for the text and ``element.metadata`` for
the metadata).

So the contract the backend depends on is: the ``unstructured.partition.*``
partitioners exist and return a list of ``Element`` objects whose text is
``str(element)`` and which carry an ``.metadata`` attribute. This module
pins that surface and exercises it OFFLINE with the lightest partitioner
(``partition_text``), which needs no model, no network, and no external
binary — the same machinery the heavier file loaders feed into.

NOTE (version drift): both ``backend/requirements.txt`` and
``pyproject.toml`` pin ``unstructured==0.22.31``, but the test venv here has
``0.18.31`` installed. These tests validate against whatever is actually
importable; the pin mismatch is a packaging observation, not something this
contract enforces.

If ``unstructured`` is not importable, every test SKIPS cleanly (it is an
optional feature: the backend degrades to fallback loaders without it).

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "unstructured"
DIST_NAME = "unstructured"

# Submodules the langchain Unstructured*Loader stack reaches into.
USED_SUBMODULES = [
    "partition",
    "partition.text",
    "documents.elements",
]

# The element data-model classes langchain converts into Documents.
ELEMENT_SYMBOLS = [
    "documents.elements.Element",
    "documents.elements.Text",
    "documents.elements.Title",
    "documents.elements.NarrativeText",
    "documents.elements.ElementMetadata",
]

SAMPLE = "Hello world.\n\nThis is a second paragraph with some narrative text."


def _partition_text(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    return depcheck.resolve(mod, "partition.text.partition_text")


def _elements(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    return depcheck.resolve(mod, "documents.elements")


# --------------------------------------------------------------------------- #
# Import / version
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "unstructured"


def test_version_reported(depcheck):
    """Sanity: a version is resolvable (so the operator knows what's installed,
    even though the pin and the env may differ)."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------- #
# Symbol existence (API surface)
# --------------------------------------------------------------------------- #


def test_submodules_importable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SUBMODULES)


def test_element_classes_exist(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, ELEMENT_SYMBOLS)


def test_partition_text_callable(depcheck):
    fn = _partition_text(depcheck)
    assert callable(fn)


def test_partition_subpackage_exposes_common_partitioners(depcheck):
    """The file loaders the backend dispatches map onto partition.* modules
    (text/xml/...). Pin that the partition subpackage at least exposes the
    text partitioner module the contract is exercised through; others are
    optional (need extra system deps), so only assert the lightest exists."""
    mod = depcheck.load(IMPORT_NAME)
    assert depcheck.has(mod, "partition.text.partition_text")


# --------------------------------------------------------------------------- #
# Behavioural: partition_text -> Element list (offline)
# --------------------------------------------------------------------------- #


def test_partition_text_returns_list_of_elements(depcheck):
    """The core contract LangChain relies on: a partitioner returns a list of
    Element objects."""
    partition_text = _partition_text(depcheck)
    el = _elements(depcheck)
    result = partition_text(text=SAMPLE)
    assert isinstance(result, list)
    assert result, "partition_text returned no elements for non-empty input"
    for e in result:
        assert isinstance(e, el.Element), f"{type(e)!r} is not an unstructured Element"


def test_element_str_is_the_text(depcheck):
    """LangChain builds Document.page_content from str(element). Pin that
    str(element) yields the element's text content."""
    partition_text = _partition_text(depcheck)
    result = partition_text(text=SAMPLE)
    joined = "\n".join(str(e) for e in result)
    assert "Hello world." in joined
    assert "second paragraph" in joined


def test_element_has_text_attribute(depcheck):
    """Element instances expose a .text attribute mirroring str(element)."""
    partition_text = _partition_text(depcheck)
    result = partition_text(text=SAMPLE)
    first = result[0]
    assert hasattr(first, "text")
    assert first.text == str(first)


def test_element_has_metadata(depcheck):
    """LangChain reads element.metadata (an ElementMetadata) when building the
    Document.metadata dict. Pin that attribute is present and convertible to a
    dict."""
    partition_text = _partition_text(depcheck)
    el = _elements(depcheck)
    result = partition_text(text=SAMPLE)
    first = result[0]
    assert hasattr(first, "metadata")
    assert isinstance(first.metadata, el.ElementMetadata)
    # ElementMetadata exposes to_dict() that LangChain merges into Document meta.
    assert hasattr(first.metadata, "to_dict")
    assert isinstance(first.metadata.to_dict(), dict)


def test_element_has_category(depcheck):
    """Elements classify content (Title / NarrativeText / ...); the category
    attribute is part of the data model LangChain may surface."""
    partition_text = _partition_text(depcheck)
    result = partition_text(text=SAMPLE)
    for e in result:
        assert hasattr(e, "category")
        assert isinstance(e.category, str)


def test_partition_splits_paragraphs(depcheck):
    """Two blank-line-separated paragraphs partition into at least two
    elements — the chunking behaviour downstream retrieval depends on."""
    partition_text = _partition_text(depcheck)
    result = partition_text(text=SAMPLE)
    assert len(result) >= 2


def test_partition_empty_text(depcheck):
    """Empty input must return a list (possibly empty), never raise — so an
    empty uploaded file degrades to zero Documents rather than crashing."""
    partition_text = _partition_text(depcheck)
    result = partition_text(text="")
    assert isinstance(result, list)


# --------------------------------------------------------------------------- #
# Element class hierarchy (the isinstance checks consumers rely on)
# --------------------------------------------------------------------------- #


def test_element_hierarchy(depcheck):
    """Title and NarrativeText are Text are Element. Code that does
    isinstance(e, Text) / isinstance(e, Element) relies on this layering."""
    el = _elements(depcheck)
    assert issubclass(el.Text, el.Element)
    assert issubclass(el.Title, el.Text)
    assert issubclass(el.NarrativeText, el.Text)
