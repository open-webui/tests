"""Dependency contract: azure-identity (import name ``azure.identity``).

``azure-identity`` provides the credential objects Open WebUI uses for
keyless/Managed-Identity authentication to Azure services:

  * ``storage/provider.py``: ``BlobServiceClient(..., credential=DefaultAzureCredential())``
    for Azure Blob storage when no account key is configured.
  * ``retrieval/loaders/main.py``: ``azure_credential=DefaultAzureCredential()``
    for the Azure Document Intelligence loader.
  * ``routers/openai.py``: ``get_bearer_token_provider(DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default")`` to mint Entra ID
    tokens for Azure OpenAI.

So the load-bearing surface is exactly two names: ``DefaultAzureCredential``
and ``get_bearer_token_provider``. A breaking bump (renamed credential,
changed ``get_token`` shape, or a different token-provider factory) would
break Azure auth at runtime.

This module pins those symbols + their call shapes and verifies the
*construction* contract offline: ``DefaultAzureCredential()`` builds a
credential lazily (authentication only happens later inside ``get_token``,
which contacts Azure — never called here), and
``get_bearer_token_provider`` returns a zero-arg callable. NO network, no
real Azure, no token acquisition.

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "azure.identity"
DIST_NAME = "azure-identity"

# The two names the backend imports, plus the credential family + error
# types that make up the stable public surface.
USED_SYMBOLS = [
    "DefaultAzureCredential",  # constructed in 3 modules
    "get_bearer_token_provider",  # routers/openai.py token provider factory
]
SURFACE_SYMBOLS = [
    "DefaultAzureCredential",
    "ManagedIdentityCredential",  # the MI path DefaultAzureCredential chains to
    "ChainedTokenCredential",
    "ClientSecretCredential",
    "EnvironmentCredential",
    "CredentialUnavailableError",  # raised when no credential in the chain works
    "get_bearer_token_provider",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`azure.identity` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "azure.identity"


def test_version_reported(depcheck):
    """The installed distribution (azure-identity) version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


# ---------------------------------------------------------------------------
# Symbol-existence checks (API surface).
# ---------------------------------------------------------------------------


def test_used_symbols_exist(depcheck):
    """The two names the backend imports must exist on azure.identity."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_surface_symbols_exist(depcheck):
    """The wider credential family the backend depends on (directly or via the
    DefaultAzureCredential chain) must remain present."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, SURFACE_SYMBOLS)


def test_default_credential_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert inspect.isclass(mod.DefaultAzureCredential)


def test_get_bearer_token_provider_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert callable(mod.get_bearer_token_provider)


# ---------------------------------------------------------------------------
# Signature contracts — the backend's exact call shapes.
# ---------------------------------------------------------------------------


def test_get_bearer_token_provider_signature(depcheck):
    """routers/openai.py calls get_bearer_token_provider(credential, scope).
    The factory must accept a credential positionally plus one-or-more scope
    strings (it takes *scopes)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.get_bearer_token_provider)
    params = list(sig.parameters.values())
    assert params, "get_bearer_token_provider lost its parameters"
    # First param is the credential.
    assert params[0].name in ("credential", "token_credential"), (
        f"unexpected first param: {[p.name for p in params]}"
    )
    # A *scopes var-positional (or an explicit scope param) must exist.
    has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params)
    has_scope = any("scope" in p.name for p in params)
    assert has_varargs or has_scope, "get_bearer_token_provider can't take scopes"


def test_default_credential_get_token_signature(depcheck):
    """The credential's get_token(*scopes, ...) is what the SDK clients call;
    pin that it accepts scopes (var-positional). We never call it (it networks)."""
    mod = depcheck.load(IMPORT_NAME)
    sig = inspect.signature(mod.DefaultAzureCredential.get_token)
    has_varargs = any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
    has_scope = any("scope" in n for n in sig.parameters)
    assert has_varargs or has_scope, f"get_token can't take scopes: {sig}"


# ---------------------------------------------------------------------------
# Construction contract (OFFLINE) — build credentials WITHOUT authenticating.
# Authentication (get_token) contacts Azure and is never invoked here.
# ---------------------------------------------------------------------------


def test_default_credential_constructs_without_auth(depcheck):
    """DefaultAzureCredential() is constructed at call time in all three
    modules; construction must be lazy (no Azure contact) and yield an object
    exposing get_token + close + the context-manager protocol the SDK uses."""
    mod = depcheck.load(IMPORT_NAME)
    cred = mod.DefaultAzureCredential()
    try:
        assert cred is not None
        assert callable(cred.get_token), "credential lost get_token"
        # SDK clients use the credential as a context manager / close it.
        assert hasattr(cred, "close")
        assert hasattr(cred, "__enter__") and hasattr(cred, "__exit__")
    finally:
        _safe_close(cred)


def test_credential_satisfies_tokencredential_protocol(depcheck):
    """The azure SDK clients (BlobServiceClient, the DI loader, AzureOpenAI)
    accept any object implementing the TokenCredential protocol — i.e. a
    get_token method. Pin that DefaultAzureCredential still satisfies it."""
    mod = depcheck.load(IMPORT_NAME)
    cred = mod.DefaultAzureCredential()
    try:
        assert hasattr(cred, "get_token") and callable(cred.get_token)
        # azure.core defines the protocol; confirm structural compatibility.
        core = depcheck.try_load("azure.core.credentials")
        if core is not None and hasattr(core, "TokenCredential"):
            assert hasattr(cred, "get_token")  # protocol member present
    finally:
        _safe_close(cred)


def test_get_bearer_token_provider_returns_callable(depcheck):
    """routers/openai.py builds token_provider = get_bearer_token_provider(cred,
    scope) and later calls token_provider() to get a string. Building it must
    not authenticate (no network); it must return a zero-arg callable. We do
    NOT call the returned provider (that would contact Azure)."""
    mod = depcheck.load(IMPORT_NAME)
    cred = mod.DefaultAzureCredential()
    try:
        provider = mod.get_bearer_token_provider(
            cred, "https://cognitiveservices.azure.com/.default"
        )
        assert callable(provider), "get_bearer_token_provider did not return a callable"
        # A zero-arg callable: its signature must accept being called with no args.
        sig = inspect.signature(provider)
        required = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        assert not required, f"token provider is not zero-arg callable: {sig}"
    finally:
        _safe_close(cred)


def test_credential_unavailable_error_is_exception(depcheck):
    """CredentialUnavailableError is what surfaces when no credential in the
    DefaultAzureCredential chain can authenticate; callers may catch it."""
    mod = depcheck.load(IMPORT_NAME)
    assert issubclass(mod.CredentialUnavailableError, Exception)


# ---------------------------------------------------------------------------
# Local helper (no cross-file imports).
# ---------------------------------------------------------------------------


def _safe_close(cred) -> None:
    """Close a credential best-effort (it never opened a connection here)."""
    try:
        cred.close()
    except Exception:
        pass
