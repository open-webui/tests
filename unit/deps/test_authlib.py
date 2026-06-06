"""Dependency contract: authlib (OAuth/OIDC SSO).

`authlib` is the OAuth 2.0 / OpenID Connect client library behind every
Open WebUI single-sign-on flow. `utils/oauth.py` and `config.py` build an
`authlib.integrations.starlette_client.OAuth` registry, register each
provider (Google, Microsoft, GitHub, OIDC, Feishu, plus per-MCP-server
dynamic clients) via `oauth.register(...)`, then drive the returned client
through `authorize_redirect` / `authorize_access_token` / `userinfo` to log
users in. ID-token verification, the userinfo payload type (`UserInfo`),
and the OAuth/JOSE error classes the callback handler catches all come from
authlib too.

This is security-critical surface: a silent API change across an authlib
bump (1.6.10 -> 1.7.2) could break SSO login, JWKS-rotation recovery, or
the callback error path. This module pins exactly the slice the backend
relies on — symbol existence, the `register()` kwargs, the registered
client's method/attribute shape, the error-class hierarchy and attributes,
and offline JOSE sign/verify behaviour — so any such change fails loudly
here instead of at a user's login.

Exemplar for the unit/deps/ pattern: symbol-existence checks (API surface)
+ offline behavioural contracts (no network — no real IdP is contacted).
Uses the `depcheck` fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "authlib"
DIST_NAME = "authlib"

# Dotted symbol paths the Open WebUI backend imports/uses off `authlib`.
# (Resolved against the top-level `authlib` module; depcheck imports
# submodules on demand.)
USED_SYMBOLS = [
    # config.py + utils/oauth.py: the OAuth registry class
    "integrations.starlette_client.OAuth",
    # utils/oauth.py: JOSE bad-signature error -> evict cached JWKS & retry
    "jose.errors.BadSignatureError",
    # utils/oauth.py: OAuth2 error -> callback error message formatting
    "oauth2.rfc6749.errors.OAuth2Error",
    # utils/oauth.py: userinfo payload type
    "oidc.core.UserInfo",
    # JOSE primitives authlib uses internally to verify ID tokens; pinned
    # so the offline sign/verify contracts below stay anchored.
    "jose.jwt",
    "jose.JsonWebToken",
    "jose.JsonWebKey",
]

# Kwargs the codebase passes to `oauth.register(...)`. Collected across
# config.py (google/microsoft/github/oidc/feishu provider registrations)
# and utils/oauth.py OAuthClientManager.add_client (per-MCP-server clients).
REGISTER_KWARGS = [
    "name",
    "client_id",
    "client_secret",
    "server_metadata_url",
    "client_kwargs",
    "redirect_uri",
    "access_token_url",
    "authorize_url",
    "api_base_url",
    "userinfo_endpoint",
    "authorize_params",
    "code_challenge_method",
    "token_endpoint_auth_method",
]

# Methods the backend invokes on a *registered* client object.
CLIENT_METHODS = [
    "authorize_redirect",
    "authorize_access_token",
    "userinfo",
    "create_authorization_url",
]

# Attributes the backend reads off a registered client object.
CLIENT_ATTRS = [
    "client_id",
    "client_secret",
    "client_kwargs",
    "_server_metadata_url",
    "server_metadata",
]


# --------------------------------------------------------------------------
# Local helpers (no cross-test imports; conftest exposes only fixtures).
# --------------------------------------------------------------------------
def _oauth_cls(depcheck):
    """Return the OAuth registry class, or skip if unavailable."""
    mod = depcheck.load(IMPORT_NAME)
    return depcheck.resolve(mod, "integrations.starlette_client.OAuth")


def _register_oidc_client(oauth):
    """Register an OIDC-style client (google/oidc provider shape).

    Mirrors config.oidc_oauth_register: server_metadata_url + client_kwargs.
    No network: registration only records config; discovery is lazy.
    """
    return oauth.register(
        name="oidc",
        client_id="cid",
        client_secret="csec",
        server_metadata_url="https://idp.test/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email", "follow_redirects": True},
    )


def _register_explicit_client(oauth):
    """Register an explicit-endpoint client (github/feishu provider shape)."""
    return oauth.register(
        name="github",
        client_id="cid",
        client_secret="csec",
        access_token_url="https://example.test/login/oauth/access_token",
        authorize_url="https://example.test/login/oauth/authorize",
        api_base_url="https://example.test/api",
        userinfo_endpoint="https://example.test/user",
        client_kwargs={"scope": "user:email"},
    )


# --------------------------------------------------------------------------
# Import + version
# --------------------------------------------------------------------------
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "authlib"


def test_version_reported(depcheck):
    """Sanity: the installed distribution version is resolvable, so bump
    tooling and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


# --------------------------------------------------------------------------
# Symbol existence — every authlib symbol the backend imports/uses.
# --------------------------------------------------------------------------
def test_used_symbols_exist(depcheck):
    """Every authlib symbol the codebase imports must still resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, USED_SYMBOLS)


def test_starlette_client_oauth_importable(depcheck):
    """`from authlib.integrations.starlette_client import OAuth` (config.py:17,
    utils/oauth.py:19) must keep working."""
    OAuth = _oauth_cls(depcheck)
    assert OAuth.__name__ == "OAuth"
    assert inspect.isclass(OAuth)


def test_bad_signature_error_importable(depcheck):
    """`from authlib.jose.errors import BadSignatureError` (utils/oauth.py:20)."""
    mod = depcheck.load(IMPORT_NAME)
    exc = depcheck.resolve(mod, "jose.errors.BadSignatureError")
    assert inspect.isclass(exc) and issubclass(exc, Exception)


def test_oauth2_error_importable(depcheck):
    """`from authlib.oauth2.rfc6749.errors import OAuth2Error` (utils/oauth.py:21)."""
    mod = depcheck.load(IMPORT_NAME)
    exc = depcheck.resolve(mod, "oauth2.rfc6749.errors.OAuth2Error")
    assert inspect.isclass(exc) and issubclass(exc, Exception)


def test_userinfo_importable(depcheck):
    """`from authlib.oidc.core import UserInfo` (utils/oauth.py:22)."""
    mod = depcheck.load(IMPORT_NAME)
    ui = depcheck.resolve(mod, "oidc.core.UserInfo")
    assert inspect.isclass(ui)


# --------------------------------------------------------------------------
# OAuth registry class — construction & registration API.
# --------------------------------------------------------------------------
def test_oauth_constructible_no_args(depcheck):
    """Both OAuthManager and OAuthClientManager do `self.oauth = OAuth()`
    with no args; the bare constructor must stay supported."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    assert oauth is not None


def test_oauth_register_and_create_client_callable(depcheck):
    """`oauth.register(...)` and `oauth.create_client(name)` are the two
    entry points the backend uses to define and fetch providers."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    assert callable(oauth.register)
    assert callable(oauth.create_client)


def test_oauth_register_accepts_var_kwargs(depcheck):
    """config.*_oauth_register / add_client pass many provider-specific
    kwargs; register must keep its **kwargs catch-all."""
    OAuth = _oauth_cls(depcheck)
    sig = inspect.signature(OAuth.register)
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()), (
        f"OAuth.register no longer accepts **kwargs (signature: {sig})"
    )
    # `name` is the one positional the backend always passes by keyword.
    assert "name" in sig.parameters


def test_oauth_create_client_signature(depcheck):
    """OAuthManager.get_client -> oauth.create_client(provider_name)."""
    OAuth = _oauth_cls(depcheck)
    depcheck.assert_params(OAuth.create_client, ["name"])


# --------------------------------------------------------------------------
# register() behaviour — offline; no IdP contacted at registration time.
# --------------------------------------------------------------------------
def test_register_oidc_style_returns_client(depcheck):
    """Registering with server_metadata_url + client_kwargs (google/oidc
    shape) returns a usable client object."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    assert client is not None


def test_register_explicit_endpoints_style_returns_client(depcheck):
    """Registering with access_token_url/authorize_url/api_base_url/
    userinfo_endpoint (github/feishu shape) returns a usable client."""
    OAuth = _oauth_cls(depcheck)
    client = _register_explicit_client(OAuth())
    assert client is not None


def test_register_every_used_kwarg_accepted(depcheck):
    """Every kwarg the backend passes to register() across all providers
    must be accepted in a single call (no rejected/renamed kwarg)."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    client = oauth.register(
        name="combo",
        client_id="cid",
        client_secret="csec",
        server_metadata_url="https://idp.test/.well-known/openid-configuration",
        client_kwargs={"scope": "openid"},
        redirect_uri="https://app.test/callback",
        access_token_url="https://idp.test/token",
        authorize_url="https://idp.test/authorize",
        api_base_url="https://idp.test/api",
        userinfo_endpoint="https://idp.test/userinfo",
        authorize_params={"prompt": "consent"},
        code_challenge_method="S256",
        token_endpoint_auth_method="client_secret_post",
    )
    assert client is not None


def test_register_kwargs_documented_match_signature(depcheck):
    """If register() ever drops **kwargs, assert each named kwarg the
    backend relies on is an explicit parameter (defensive — today it's
    **kwargs, so this passes trivially)."""
    OAuth = _oauth_cls(depcheck)
    sig = inspect.signature(OAuth.register)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        pytest.skip("register() still has **kwargs; explicit-param check N/A")
    missing = [k for k in REGISTER_KWARGS if k not in sig.parameters]
    assert not missing, f"register() lost kwargs the backend uses: {missing}"


# --------------------------------------------------------------------------
# Registered client object — method & attribute shape.
# --------------------------------------------------------------------------
def test_client_exposes_called_methods(depcheck):
    """The login flow calls authorize_redirect / authorize_access_token /
    userinfo / create_authorization_url on the registered client."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    for m in CLIENT_METHODS:
        assert callable(getattr(client, m, None)), (
            f"registered client missing callable {m!r} (login flow / preflight relies on it)"
        )


def test_client_async_methods_are_coroutines(depcheck):
    """utils/oauth.py `await`s these — the starlette integration must keep
    them as coroutine functions, not sync (a sync swap would break every
    `await client.authorize_access_token(...)`)."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    for m in (
        "authorize_redirect",
        "authorize_access_token",
        "userinfo",
        "create_authorization_url",
    ):
        fn = getattr(client, m)
        assert inspect.iscoroutinefunction(fn), f"client.{m} is no longer async"


def test_client_authorize_access_token_accepts_request_and_kwargs(depcheck):
    """handle_callback calls client.authorize_access_token(request, **kwargs)
    (passing resource / client_id). request positional + **kwargs required."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    sig = inspect.signature(client.authorize_access_token)
    params = sig.parameters
    assert "request" in params, f"authorize_access_token lost `request` ({sig})"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        f"authorize_access_token no longer accepts **kwargs ({sig})"
    )


def test_client_authorize_redirect_accepts_request_and_redirect_uri(depcheck):
    """handle_login / handle_authorize call
    client.authorize_redirect(request, redirect_uri, **kwargs)."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    sig = inspect.signature(client.authorize_redirect)
    params = sig.parameters
    assert "request" in params, f"authorize_redirect lost `request` ({sig})"
    has_redirect = "redirect_uri" in params or any(
        p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()
    )
    assert has_redirect, f"authorize_redirect can't take redirect_uri ({sig})"
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
        f"authorize_redirect no longer accepts **kwargs ({sig})"
    )


def test_client_userinfo_accepts_token_kwarg(depcheck):
    """handle_callback calls `client.userinfo(token=token)`."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    depcheck.assert_params(client.userinfo, ["token"])


def test_client_exposes_credential_attributes(depcheck):
    """_perform_token_refresh reads client.client_id / client.client_secret /
    client.client_kwargs to build the refresh request."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    assert client.client_id == "cid"
    assert client.client_secret == "csec"
    assert isinstance(client.client_kwargs, dict)
    assert client.client_kwargs.get("scope") == "openid email"


def test_client_records_server_metadata_url(depcheck):
    """get_server_metadata_url reads client._server_metadata_url; it must be
    populated from the server_metadata_url passed to register()."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    assert (
        getattr(client, "_server_metadata_url", None)
        == "https://idp.test/.well-known/openid-configuration"
    )


def test_client_has_server_metadata_attribute(depcheck):
    """handle_login reads `client.server_metadata` (a dict-or-empty) and the
    BadSignatureError path does `client.server_metadata.pop('jwks', None)`.
    The attribute must exist on the client surface."""
    OAuth = _oauth_cls(depcheck)
    client = _register_oidc_client(OAuth())
    # dir() lists the name without executing a property getter (which could
    # trigger lazy network metadata loading); existence is what we pin.
    assert "server_metadata" in dir(client), "client.server_metadata gone"


# --------------------------------------------------------------------------
# Registry retrieval & internals (create_client / _clients / _registry).
# --------------------------------------------------------------------------
def test_create_client_returns_registered_client(depcheck):
    """OAuthManager.get_client falls back to oauth.create_client(name); it
    must return the previously registered client."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    registered = _register_oidc_client(oauth)
    fetched = oauth.create_client("oidc")
    assert fetched is not None
    assert fetched.client_id == registered.client_id == "cid"


def test_create_client_unknown_returns_none(depcheck):
    """get_client relies on create_client(unknown) -> None (not raising) so
    it can return None for an unconfigured provider."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    assert oauth.create_client("never-registered") is None


def test_registry_internal_dicts_exist(depcheck):
    """OAuthClientManager.remove_client pops the client id out of
    `self.oauth._clients` and `self.oauth._registry`; both private dicts
    must still exist and contain a registered client's name."""
    OAuth = _oauth_cls(depcheck)
    oauth = OAuth()
    _register_oidc_client(oauth)
    assert hasattr(oauth, "_clients"), "OAuth._clients internal dict gone"
    assert hasattr(oauth, "_registry"), "OAuth._registry internal dict gone"
    assert "oidc" in oauth._registry, "registered name absent from _registry"


# --------------------------------------------------------------------------
# UserInfo — used as a plain dict throughout handle_callback.
# --------------------------------------------------------------------------
def test_userinfo_is_dict_subclass(depcheck):
    """token.get('userinfo') / client.userinfo() are typed UserInfo and
    consumed as dicts: dict(user_data), user_data.get(claim),
    `claim in user_data`, user_data[k] = v. Pin the dict semantics."""
    mod = depcheck.load(IMPORT_NAME)
    UserInfo = depcheck.resolve(mod, "oidc.core.UserInfo")
    assert issubclass(UserInfo, dict)


def test_userinfo_dict_operations(depcheck):
    """Exercise every UserInfo dict op handle_callback performs."""
    mod = depcheck.load(IMPORT_NAME)
    UserInfo = depcheck.resolve(mod, "oidc.core.UserInfo")
    ui = UserInfo({"sub": "abc123", "email": "user@example.test"})
    # .get(claim) — sub/email/groups/roles extraction
    assert ui.get("sub") == "abc123"
    assert ui.get("missing") is None
    # `claim in user_data` — OAUTH_EMAIL_CLAIM membership checks
    assert "email" in ui
    assert "nope" not in ui
    # dict(user_data) — id_token_claims = dict(user_data)
    assert dict(ui) == {"sub": "abc123", "email": "user@example.test"}
    # user_data[k] = v — backfilling id-token claims into userinfo
    ui["custom_role"] = "admin"
    assert ui["custom_role"] == "admin"
    # .items() — iterating to merge id_token_claims
    assert ("sub", "abc123") in list(ui.items())


# --------------------------------------------------------------------------
# Error classes — hierarchy & attributes the callback handler depends on.
# --------------------------------------------------------------------------
def test_oauth2_error_is_exception(depcheck):
    """_build_oauth_callback_error_message does `isinstance(e, OAuth2Error)`;
    it must be a catchable Exception subclass."""
    mod = depcheck.load(IMPORT_NAME)
    OAuth2Error = depcheck.resolve(mod, "oauth2.rfc6749.errors.OAuth2Error")
    assert issubclass(OAuth2Error, Exception)


def test_oauth2_error_exposes_error_and_description(depcheck):
    """_build_oauth_callback_error_message reads `e.error` and
    `e.description` to format the user-facing callback error."""
    mod = depcheck.load(IMPORT_NAME)
    OAuth2Error = depcheck.resolve(mod, "oauth2.rfc6749.errors.OAuth2Error")
    err = OAuth2Error(error="invalid_grant", description="token expired")
    assert err.error == "invalid_grant"
    assert err.description == "token expired"


def test_bad_signature_error_is_exception(depcheck):
    """handle_callback catches BadSignatureError to evict stale JWKS and
    retry; it must be a catchable Exception subclass."""
    mod = depcheck.load(IMPORT_NAME)
    BadSignatureError = depcheck.resolve(mod, "jose.errors.BadSignatureError")
    assert issubclass(BadSignatureError, Exception)


def test_bad_signature_error_is_jose_error(depcheck):
    """BadSignatureError should remain under authlib's JOSE error base so
    the JWKS-rotation `except BadSignatureError` stays narrowly scoped."""
    mod = depcheck.load(IMPORT_NAME)
    BadSignatureError = depcheck.resolve(mod, "jose.errors.BadSignatureError")
    base = depcheck.resolve(mod, "jose.errors.JoseError")
    assert issubclass(BadSignatureError, base)


# --------------------------------------------------------------------------
# JOSE jwt — offline sign/verify (authlib's ID-token verification core).
# --------------------------------------------------------------------------
def test_jose_jwt_has_encode_decode(depcheck):
    """authlib.jose.jwt.{encode,decode} back ID-token signing/verification."""
    mod = depcheck.load(IMPORT_NAME)
    jwt = depcheck.resolve(mod, "jose.jwt")
    assert callable(jwt.encode)
    assert callable(jwt.decode)


def test_jose_jwt_sign_verify_roundtrip(depcheck):
    """Offline HS256 sign+verify: encode a claims set, decode it back, and
    confirm the round-tripped claims match. Pins the encode/decode contract
    authlib uses to validate OIDC ID tokens — no IdP, no network."""
    mod = depcheck.load(IMPORT_NAME)
    jwt = depcheck.resolve(mod, "jose.jwt")
    key = "unit-test-shared-secret"
    payload = {"sub": "user-1", "email": "a@b.test", "iss": "https://idp.test"}
    token = jwt.encode({"alg": "HS256"}, payload, key)
    claims = jwt.decode(token, key)
    assert claims["sub"] == "user-1"
    assert claims["email"] == "a@b.test"
    assert claims["iss"] == "https://idp.test"


def test_jose_jwt_decode_rejects_wrong_key(depcheck):
    """A token verified with the wrong key must raise BadSignatureError —
    this is exactly the signal handle_callback catches to re-fetch JWKS."""
    mod = depcheck.load(IMPORT_NAME)
    jwt = depcheck.resolve(mod, "jose.jwt")
    BadSignatureError = depcheck.resolve(mod, "jose.errors.BadSignatureError")
    token = jwt.encode({"alg": "HS256"}, {"sub": "x"}, "right-key")
    with pytest.raises(BadSignatureError):
        jwt.decode(token, "wrong-key")


def test_jose_decoded_claims_support_get_and_validate(depcheck):
    """Decoded claims are dict-like (.get) and expose .validate() for exp/
    aud/iss checks — the shape authlib applies during ID-token validation."""
    mod = depcheck.load(IMPORT_NAME)
    jwt = depcheck.resolve(mod, "jose.jwt")
    token = jwt.encode({"alg": "HS256"}, {"sub": "x", "aud": "cid"}, "k")
    claims = jwt.decode(token, "k")
    assert callable(getattr(claims, "get", None))
    assert claims.get("sub") == "x"
    assert callable(getattr(claims, "validate", None))


def test_json_web_token_constructible(depcheck):
    """JsonWebToken (the class behind the jwt singleton) must stay importable
    and constructible with an allowed-algorithms list — authlib instantiates
    it for OIDC ID-token verification with the IdP's supported algs."""
    mod = depcheck.load(IMPORT_NAME)
    JsonWebToken = depcheck.resolve(mod, "jose.JsonWebToken")
    jwt_obj = JsonWebToken(["HS256", "RS256"])
    token = jwt_obj.encode({"alg": "HS256"}, {"sub": "y"}, "k")
    claims = jwt_obj.decode(token, "k")
    assert claims["sub"] == "y"


def test_json_web_key_importable(depcheck):
    """JsonWebKey backs JWKS parsing for RS256 ID-token verification; it must
    stay importable with its import_key_set / import_key entry points."""
    mod = depcheck.load(IMPORT_NAME)
    JsonWebKey = depcheck.resolve(mod, "jose.JsonWebKey")
    assert callable(getattr(JsonWebKey, "import_key_set", None)) or callable(
        getattr(JsonWebKey, "import_key", None)
    ), "JsonWebKey lost both import_key_set and import_key"
