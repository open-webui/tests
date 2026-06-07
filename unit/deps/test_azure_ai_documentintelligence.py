"""Dependency contract: azure-ai-documentintelligence.

The Azure AI Document Intelligence SDK (import root
``azure.ai.documentintelligence``) is pinned in
``backend/requirements.txt`` (``azure-ai-documentintelligence==1.0.2``).
Open WebUI uses it as one of its document-extraction engines for RAG
ingestion: when ``CONTENT_EXTRACTION_ENGINE == 'document_intelligence'``,
``retrieval/loaders/main.py`` constructs langchain's
``AzureAIDocumentIntelligenceLoader``:

    AzureAIDocumentIntelligenceLoader(
        file_path=...,
        api_endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT,
        api_key=DOCUMENT_INTELLIGENCE_KEY,         # key-auth branch
        # or azure_credential=DefaultAzureCredential()  # AAD-auth branch
        api_model=DOCUMENT_INTELLIGENCE_MODEL,
    )

IMPORTANT — usage note: Open WebUI does NOT import
``azure.ai.documentintelligence`` directly. It is reached *through*
langchain's loader, whose parser internally does::

    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import DocumentAnalysisFeature
    from azure.core.credentials import AzureKeyCredential
    client = DocumentIntelligenceClient(endpoint, credential, **kwargs)
    poller = client.begin_analyze_document(api_model, body=..., ...)

So this is the SDK surface langchain (and therefore Open WebUI) depends
on. This module pins exactly that: the ``DocumentIntelligenceClient`` and
its ``begin_analyze_document`` method, the ``models.DocumentAnalysisFeature``
enum the parser imports, and an OFFLINE client construction (the client is
lazy — it only contacts Azure when an ``analyze`` call is made, which we
never do). No network, no Azure account, no document is sent.

Uses the ``depcheck`` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "azure.ai.documentintelligence"
DIST_NAME = "azure-ai-documentintelligence"

# Top-level package symbols.
PACKAGE_SYMBOLS = [
    "DocumentIntelligenceClient",  # langchain parser imports this
    "DocumentIntelligenceAdministrationClient",
    "models",  # submodule with the request/result/feature types
]

# Client methods the analyze flow uses.
CLIENT_METHODS = [
    "begin_analyze_document",  # langchain parser: client.begin_analyze_document(...)
]


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "azure.ai.documentintelligence"


def test_version_reported(depcheck):
    assert depcheck.dist_version(DIST_NAME) is not None


def test_package_symbols_exist(depcheck):
    """The client classes + models submodule the langchain loader imports must
    remain on the package root."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, PACKAGE_SYMBOLS)


def test_client_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.DocumentIntelligenceClient)


def test_client_constructor_accepts_endpoint_and_credential(depcheck):
    """The parser does DocumentIntelligenceClient(endpoint, credential, **kwargs).
    Pin that the constructor accepts endpoint + credential (it also takes
    **kwargs for api_version)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.DocumentIntelligenceClient.__init__,
        ["endpoint", "credential"],
    )


def test_client_has_begin_analyze_document(depcheck):
    """begin_analyze_document is the call the parser makes to extract a document.
    It must exist and be callable on the client class."""
    mod = depcheck.load(IMPORT_NAME)
    names = set(dir(mod.DocumentIntelligenceClient))
    missing = [m for m in CLIENT_METHODS if m not in names]
    assert not missing, f"DocumentIntelligenceClient missing method(s): {missing}"
    assert callable(mod.DocumentIntelligenceClient.begin_analyze_document)


def test_models_document_analysis_feature_exists(depcheck):
    """langchain's parser does `from azure.ai.documentintelligence.models import
    DocumentAnalysisFeature` and passes analysis_features through. The enum must
    remain importable off the models submodule."""
    depcheck.load(IMPORT_NAME)
    models = depcheck.load("azure.ai.documentintelligence.models")
    assert hasattr(models, "DocumentAnalysisFeature"), (
        "azure.ai.documentintelligence.models lost DocumentAnalysisFeature"
    )


def test_models_core_result_types_exist(depcheck):
    """The analyze flow returns an AnalyzeResult (built from an
    AnalyzeDocumentRequest body). Pin both types so a poller.result() consumer
    keeps compiling against the SDK."""
    depcheck.load(IMPORT_NAME)
    models = depcheck.load("azure.ai.documentintelligence.models")
    for name in ("AnalyzeResult", "AnalyzeDocumentRequest"):
        assert hasattr(models, name), f"azure.ai.documentintelligence.models lost {name}"


def test_client_constructs_offline_with_key_credential(depcheck):
    """Mirror the key-auth branch (api_key set): build a client with an
    AzureKeyCredential and a dummy endpoint. Construction must NOT contact Azure
    (the client is lazy); begin_analyze_document must be present on the instance.
    Skips cleanly if azure-core (the credential type) isn't importable."""
    mod = depcheck.load(IMPORT_NAME)
    creds = depcheck.try_load("azure.core.credentials")
    if creds is None or not hasattr(creds, "AzureKeyCredential"):
        pytest.skip("azure.core.credentials.AzureKeyCredential unavailable")
    client = mod.DocumentIntelligenceClient(
        endpoint="https://example.cognitiveservices.azure.com/",
        credential=creds.AzureKeyCredential("dummy-offline-key"),
    )
    assert client is not None
    assert hasattr(client, "begin_analyze_document")
    assert callable(client.begin_analyze_document)


def test_begin_analyze_document_first_arg_is_model_id(depcheck):
    """The parser calls begin_analyze_document(api_model, body=..., ...), passing
    the model id (e.g. 'prebuilt-layout') first. Pin that the method accepts a
    model-id parameter as its leading argument."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.DocumentIntelligenceClient.begin_analyze_document)
    params = [p for p in sig.parameters if p != "self"]
    assert params, "begin_analyze_document takes no arguments besides self"
    # Historically the leading param is the model id (model_id / modelId).
    assert "model" in params[0].lower(), (
        f"begin_analyze_document's first parameter is {params[0]!r}, expected the model id"
    )


def test_not_imported_by_backend_marker():
    """Documentation guard (no dep assertion): the backend reaches this SDK only
    through langchain's AzureAIDocumentIntelligenceLoader, not via a direct
    import. The surface pins above guard exactly what that loader's parser
    imports and calls."""
    assert True
