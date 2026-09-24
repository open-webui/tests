"""A fake OpenID Connect provider that signs real tokens, and the sign-in flow that uses it.

The provider serves discovery, an authorize endpoint that approves at once, a token endpoint,
userinfo and a JWKS on a local port and records every request it gets. A test says who signs
in next with `sign_in_as(...)`: the claims the ID token and userinfo carry, extra JOSE header
entries or a userinfo answer that differs from the ID token. `rotate_key()` rolls the signing
key under the same `kid`, the way an IdP rotates without renaming its key.

`sso_env(provider)` is the environment that points an instance at it, `sign_in(instance)` walks
the browser's redirect chain with httpx and returns the session the callback handed out, and
`oauth_settings(instance, ...)` changes admin-panel OAuth settings for the length of a block.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from harness.instance import LaunchedInstance, free_port

CLIENT_ID = "owui-test-client"
CLIENT_SECRET = "owui-test-client-secret-0123456789abcdef"
OAUTH_CONFIG_PATH = "/api/v1/auths/admin/config/oauth"


def _new_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@dataclass
class ProviderRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    form: dict[str, str]


@dataclass
class OidcProvider:
    base_url: str
    client_id: str = CLIENT_ID
    client_secret: str = CLIENT_SECRET
    kid: str = "signing-key"
    key: rsa.RSAPrivateKey = field(default_factory=_new_key)
    requests: list[ProviderRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    # who the next sign-in is; see sign_in_as
    claims: dict = field(default_factory=dict)
    id_token_claims: dict | None = None
    userinfo: dict | None = None
    jose_header: dict = field(default_factory=dict)
    signature: str = "provider"
    codes: dict[str, dict] = field(default_factory=dict)
    access_tokens: dict[str, dict] = field(default_factory=dict)
    issued: list[dict] = field(default_factory=list)  # every token response, newest last

    @property
    def issuer(self) -> str:
        return self.base_url

    @property
    def discovery_url(self) -> str:
        return f"{self.base_url}/.well-known/openid-configuration"

    def reset(self) -> None:
        """Forget the request log and script a fresh person; the signing key stays."""
        with self.lock:
            self.requests.clear()
            self.issued.clear()
        self.sign_in_as()

    def sign_in_as(
        self,
        *,
        id_token_claims: dict | None = None,
        userinfo: dict | None = None,
        jose_header: dict | None = None,
        signature: str = "provider",
        **claims,
    ) -> dict:
        """Script the next sign-in; unset `sub`, `email` and `name` get fresh unique values.

        The claims go into the ID token and the userinfo answer alike, unless `id_token_claims`
        or `userinfo` replaces one of them. `signature` signs the ID token with the provider's
        key (`provider`), a key it never published (`foreign-key`) or, HS256, the client secret
        (`client-secret`).
        """
        suffix = secrets.token_hex(4)
        person = {
            "sub": f"sub-{suffix}",
            "email": f"sso-{suffix}@example.com",
            "name": f"SSO {suffix}",
            **claims,
        }
        with self.lock:
            self.claims = person
            self.id_token_claims = id_token_claims
            self.userinfo = userinfo
            self.jose_header = jose_header or {}
            self.signature = signature
        return person

    def rotate_key(self) -> None:
        """Sign with new key material under the same `kid`; the JWKS only serves the new key."""
        with self.lock:
            self.key = _new_key()

    def issue_access_token(
        self, userinfo: dict, *, claims: dict | None = None, opaque: bool = False
    ) -> str:
        """An access token whose userinfo answer is `userinfo`, a signed JWT unless `opaque`."""
        if opaque:
            token = secrets.token_urlsafe(24)
        else:
            now = int(time.time())
            payload = {"iss": self.issuer, "iat": now, "exp": now + 3600, **(claims or {})}
            payload["jti"] = secrets.token_hex(8)
            token = jwt.encode(payload, self.key, algorithm="RS256", headers={"kid": self.kid})
        with self.lock:
            self.access_tokens[token] = dict(userinfo)
        return token

    def logout_token(self, sub: str) -> str:
        """A signed back-channel logout token telling the relying party `sub` signed out."""
        payload = {
            "iss": self.issuer,
            "aud": self.client_id,
            "iat": int(time.time()),
            "jti": secrets.token_hex(8),
            "sub": sub,
            "events": {"http://schemas.openid.net/event/backchannel-logout": {}},
        }
        return jwt.encode(payload, self.key, algorithm="RS256", headers={"kid": self.kid})

    def requests_to(self, path: str) -> list[ProviderRequest]:
        with self.lock:
            return [entry for entry in self.requests if entry.path == path]

    def discovery(self) -> dict:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.base_url}/authorize",
            "token_endpoint": f"{self.base_url}/token",
            "userinfo_endpoint": f"{self.base_url}/userinfo",
            "jwks_uri": f"{self.base_url}/jwks",
            "end_session_endpoint": f"{self.base_url}/logout",
            "response_types_supported": ["code"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "email", "profile"],
            "token_endpoint_auth_methods_supported": ["client_secret_basic", "client_secret_post"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
        }

    def jwks(self) -> dict:
        public = RSAAlgorithm.to_jwk(self.key.public_key(), as_dict=True)
        return {"keys": [{**public, "kid": self.kid, "use": "sig", "alg": "RS256"}]}

    def _authorize(self, query: dict[str, str]) -> str:
        code = secrets.token_urlsafe(16)
        with self.lock:
            self.codes[code] = {
                "nonce": query.get("nonce"),
                "claims": dict(self.claims),
                "id_token_claims": self.id_token_claims,
                "userinfo": self.userinfo,
                "jose_header": dict(self.jose_header),
                "signature": self.signature,
            }
        answer = {"code": code, **({"state": query["state"]} if "state" in query else {})}
        separator = "&" if "?" in query["redirect_uri"] else "?"
        return f"{query['redirect_uri']}{separator}{urllib.parse.urlencode(answer)}"

    def _token(self, form: dict[str, str]) -> tuple[int, dict]:
        with self.lock:
            grant = self.codes.pop(form.get("code", ""), None)
        if grant is None:
            return 400, {"error": "invalid_grant"}
        person = grant["claims"]
        now = int(time.time())
        id_claims = grant["id_token_claims"] if grant["id_token_claims"] is not None else person
        id_payload = {"iss": self.issuer, "aud": self.client_id, "iat": now, "exp": now + 3600}
        id_payload.update(id_claims)
        if grant["nonce"]:
            id_payload["nonce"] = grant["nonce"]
        header = {"kid": self.kid, **grant["jose_header"]}
        if grant["signature"] == "client-secret":
            id_token = jwt.encode(id_payload, self.client_secret, algorithm="HS256", headers=header)
        else:
            signing_key = _new_key() if grant["signature"] == "foreign-key" else self.key
            id_token = jwt.encode(id_payload, signing_key, algorithm="RS256", headers=header)
        userinfo = grant["userinfo"] if grant["userinfo"] is not None else person
        access_token = self.issue_access_token(userinfo, claims={"sub": str(person["sub"])})
        answer = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": secrets.token_urlsafe(24),
            "id_token": id_token,
            "scope": "openid email profile",
        }
        with self.lock:
            self.issued.append(answer)
        return 200, answer

    def _userinfo(self, authorization: str) -> tuple[int, dict]:
        token = authorization.removeprefix("Bearer ").strip()
        with self.lock:
            answer = self.access_tokens.get(token)
        return (200, answer) if answer is not None else (401, {"error": "invalid_token"})


def serve() -> tuple[OidcProvider, Callable[[], None]]:
    """Start a provider on a daemon thread; returns it with its shutdown function."""
    port = free_port()
    provider = OidcProvider(base_url=f"http://127.0.0.1:{port}")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _record(self) -> ProviderRequest:
            url = urllib.parse.urlsplit(self.path)
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length).decode() if length else ""
            entry = ProviderRequest(
                method=self.command,
                path=url.path,
                query=dict(urllib.parse.parse_qsl(url.query)),
                headers=dict(self.headers.items()),
                form=dict(urllib.parse.parse_qsl(body)),
            )
            with provider.lock:
                provider.requests.append(entry)
            return entry

        def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            for name, value in {"Content-Type": "application/json", **(headers or {})}.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self) -> None:
            entry = self._record()
            if entry.path == "/.well-known/openid-configuration":
                self._send(200, provider.discovery())
            elif entry.path == "/jwks":
                self._send(200, provider.jwks())
            elif entry.path == "/authorize":
                self._send(302, {}, {"Location": provider._authorize(entry.query)})
            elif entry.path == "/userinfo":
                self._send(*provider._userinfo(entry.headers.get("Authorization", "")))
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            entry = self._record()
            if entry.path == "/token":
                self._send(*provider._token(entry.form))
            else:
                self._send(404, {"error": "not_found"})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def shutdown() -> None:
        server.shutdown()
        server.server_close()

    return provider, shutdown


_shared: OidcProvider | None = None


def shared_provider() -> OidcProvider:
    """The one provider of this test process, so every module shares one OAuth instance."""
    global _shared
    if _shared is None:
        _shared, _shutdown = serve()
    _shared.reset()
    return _shared


def sso_env(provider: OidcProvider) -> dict[str, str]:
    """Env for an instance that signs in through `provider`; other OAuth settings are runtime."""
    return {
        "OAUTH_CLIENT_ID": provider.client_id,
        "OAUTH_CLIENT_SECRET": provider.client_secret,
        "OPENID_PROVIDER_URL": provider.discovery_url,
        "ENABLE_OAUTH_SIGNUP": "true",
        "ENABLE_OAUTH_TOKEN_EXCHANGE": "true",
        "ENABLE_OAUTH_BACKCHANNEL_LOGOUT": "true",
    }


@dataclass
class SignIn:
    token: str | None  # the session token the callback set, None when the sign-in failed
    error: str | None  # the error the sign-in page was sent to show
    browser: httpx.Client


def browser_for(instance: LaunchedInstance) -> httpx.Client:
    """A cookie-keeping client that stops at every redirect, like a browser we can inspect."""
    return httpx.Client(base_url=instance.base_url, follow_redirects=False, timeout=60.0)


def sign_in(instance: LaunchedInstance, browser: httpx.Client | None = None) -> SignIn:
    """Press "Continue with SSO": Open WebUI, the provider's authorize, back to the callback."""
    browser = browser or browser_for(instance)
    start = browser.get("/oauth/oidc/login")
    assert start.status_code == 302, f"SSO login did not redirect: {start.status_code} {start.text}"
    approved = browser.get(start.headers["location"])
    assert approved.status_code == 302, f"the provider did not approve: {approved.text}"
    callback = browser.get(approved.headers["location"])
    assert callback.status_code in (302, 307), f"callback answered {callback.status_code}"
    landing = urllib.parse.urlsplit(callback.headers["location"])
    error = dict(urllib.parse.parse_qsl(landing.query)).get("error")
    return SignIn(token=callback.cookies.get("token"), error=error, browser=browser)


def session_user(instance: LaunchedInstance, token: str) -> dict:
    """What `GET /api/v1/auths/` says about the account a session token belongs to."""
    with instance.client(token) as client:
        answer = client.get("/api/v1/auths/")
    answer.raise_for_status()
    return answer.json()


@contextmanager
def oauth_settings(instance: LaunchedInstance, **changes) -> Iterator[None]:
    """Save admin-panel OAuth settings for the block, then put the previous ones back."""
    with instance.client() as admin:
        saved = admin.get(OAUTH_CONFIG_PATH)
        saved.raise_for_status()
        applied = admin.post(OAUTH_CONFIG_PATH, json=changes)
        assert applied.status_code == 200, f"saving OAuth settings failed: {applied.text}"
        try:
            yield
        finally:
            restored = admin.post(OAUTH_CONFIG_PATH, json=saved.json())
            assert restored.status_code == 200, f"restoring OAuth settings failed: {restored.text}"


@contextmanager
def group_named(instance: LaunchedInstance, name: str) -> Iterator[dict]:
    """A group the provider's groups claim can name, deleted again afterwards."""
    with instance.client() as admin:
        created = admin.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, f"creating group {name} failed: {created.text}"
        try:
            yield created.json()
        finally:
            admin.delete(f"/api/v1/groups/id/{created.json()['id']}/delete")


def group_member_ids(instance: LaunchedInstance, group: dict) -> list[str]:
    with instance.client() as admin:
        exported = admin.get(f"/api/v1/groups/id/{group['id']}/export")
    exported.raise_for_status()
    return exported.json()["user_ids"]
