"""An OAuth 2.1 authorization server for MCP tool servers, and an MCP server that requires it.

`serving_protected_mcp()` yields a `ProtectedMcp`: the echo server of `harness.mcp_server`
behind the SDK's bearer check, and a local authorization server with what Open WebUI's MCP OAuth
client walks through. The MCP server's 401 names its protected-resource metadata, which names
the authorization server, whose RFC 8414 metadata lists dynamic client registration, an
authorize endpoint that requires a PKCE S256 challenge and approves at once, and a token
endpoint for the code and refresh grants. Refresh tokens rotate: a spent, revoked or unknown one
answers `invalid_grant`.

The authorization server records every request. `access_token_lifetime` sets the `expires_in`
of the tokens it issues next, `registration_names_scope = False` leaves the scope out of its
registration answers, `token_base` names the token endpoint of its metadata by another base URL,
`authorize_base` its authorize endpoint (`GOOGLE_ACCOUNTS` plays Google's sign-in page), and
`reached(url)` points a URL the browser was sent there back at this server. With
`offline_access_only` a code grant carries a refresh token only when the authorize asked for
`access_type=offline`, the way Google's does. With `bot_challenge` the authorize endpoint
answers Open WebUI's own server-side requests the way Cloudflare's bot protection does: a 403
with `cf-mitigated: challenge` and an HTML page that repeats the requested URL, while a browser
passes as before. `revoke_refresh_tokens()` withdraws the live refresh tokens, `forget_clients()`
drops every registered client so the authorize answers `invalid_client`, and
`ProtectedMcp.presented` lists every bearer token the MCP server was shown.
`register_oauth_mcp()` registers the server as the admin panel does and saves the connection (after
`preserve(TOOL_SERVERS)`). `scopes` is what the MCP server's protected-resource metadata asks for
and its tools require (`mcp:tools` by default; `GOOGLE_WORKSPACE_SCOPES` is a list as long as
Google Workspace's). Both servers listen on `host` and name themselves by `name` (the address by
default), so a test reaches them by a host name.
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import secrets
import threading
import time
import urllib.parse
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterator

from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.transport_security import TransportSecuritySettings

from harness.instance import free_port
from harness.listener import IPv6HTTPServer
from harness.mcp_server import TOOL_SERVERS, serving_mcp

SCOPE = "mcp:tools"
GOOGLE_ACCOUNTS = "https://accounts.google.com"
GOOGLE_WORKSPACE_SCOPES = [
    f"https://www.googleapis.com/auth/{service}{kind}"
    for service in (
        "calendar",
        "drive",
        "gmail",
        "documents",
        "spreadsheets",
        "presentations",
        "tasks",
    )
    for kind in (
        "",
        ".readonly",
        ".file",
        ".metadata",
        ".settings.basic",
        ".events",
    )
]


@dataclass
class AuthorizationRequest:
    method: str
    path: str
    query: dict[str, str]
    headers: dict[str, str]
    body: str

    @property
    def form(self) -> dict[str, str]:
        return dict(urllib.parse.parse_qsl(self.body))

    def json(self) -> dict:
        return json.loads(self.body)


def _s256(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _client_credentials(request: AuthorizationRequest) -> tuple[str, str]:
    """The client id and secret, sent as HTTP Basic or in the form (`client_secret_post`)."""
    authorization = request.headers.get("Authorization", "")
    if authorization.startswith("Basic "):
        decoded = base64.b64decode(authorization.removeprefix("Basic ")).decode()
        client_id, _, secret = decoded.partition(":")
        return urllib.parse.unquote(client_id), urllib.parse.unquote(secret)
    form = request.form
    return form.get("client_id", ""), form.get("client_secret", "")


@dataclass
class McpAuthorizationServer:
    base_url: str
    requests: list[AuthorizationRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    clients: dict[str, dict] = field(default_factory=dict)  # by client_id, as registered
    codes: dict[str, dict] = field(default_factory=dict)
    access_tokens: dict[str, dict] = field(default_factory=dict)
    refresh_tokens: dict[str, dict] = field(default_factory=dict)  # live ones only
    issued: list[dict] = field(default_factory=list)  # every token response, newest last
    access_token_lifetime: int = 3600
    registration_names_scope: bool = True  # RFC 7591 lets a server leave `scope` out (Atlassian)
    token_base: str | None = None  # the base URL the metadata gives the token endpoint, if not ours
    authorize_base: str | None = None  # the same for the authorize endpoint
    offline_access_only: bool = False
    bot_challenge: bool = False

    @property
    def issuer(self) -> str:
        return self.base_url

    def metadata(self) -> dict:
        return {
            "issuer": self.issuer,
            "authorization_endpoint": f"{self.authorize_base or self.base_url}/authorize",
            "token_endpoint": f"{self.token_base or self.base_url}/token",
            "registration_endpoint": f"{self.base_url}/register",
            "scopes_supported": [SCOPE],
            "response_types_supported": ["code"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic"],
            "code_challenge_methods_supported": ["S256"],
        }

    def reached(self, url: str) -> str:
        """A URL on the metadata's authorize endpoint, as a browser would have reached it here."""
        return url.replace(self.authorize_base or self.base_url, self.base_url, 1)

    def requests_to(self, path: str) -> list[AuthorizationRequest]:
        with self.lock:
            return [entry for entry in self.requests if entry.path == path]

    def token_grants(self, grant_type: str) -> list[dict[str, str]]:
        """The forms of the token requests with this `grant_type`, oldest first."""
        forms = [entry.form for entry in self.requests_to("/token")]
        return [form for form in forms if form.get("grant_type") == grant_type]

    def revoke_refresh_tokens(self) -> None:
        """Withdraw every refresh token issued so far; using one then answers `invalid_grant`."""
        with self.lock:
            self.refresh_tokens.clear()

    def forget_clients(self) -> None:
        """Drop every registered client, as a provider does that expires unused registrations."""
        with self.lock:
            self.clients.clear()

    def access_token(self, token: str) -> dict | None:
        """What the server knows about an access token it issued, None for any other."""
        with self.lock:
            return self.access_tokens.get(token)

    def _register(self, metadata: dict) -> tuple[int, dict]:
        if not metadata.get("redirect_uris"):
            return 400, {"error": "invalid_redirect_uri"}
        client = {
            **metadata,
            "client_id": f"client-{secrets.token_hex(6)}",
            "client_secret": secrets.token_urlsafe(24),
            "client_id_issued_at": int(time.time()),
            "client_secret_expires_at": 0,
        }
        if not self.registration_names_scope:
            client.pop("scope", None)
        with self.lock:
            self.clients[client["client_id"]] = client
        return 201, client

    def _authorize(self, query: dict[str, str]) -> tuple[int, dict, str | None]:
        """(status, error body, redirect): a registered client and an S256 challenge get a code."""
        with self.lock:
            client = self.clients.get(query.get("client_id", ""))
        if client is None:
            return 400, {"error": "invalid_client"}, None
        if query.get("redirect_uri") not in client["redirect_uris"]:
            return 400, {"error": "invalid_request", "error_description": "bad redirect_uri"}, None
        if query.get("code_challenge_method") != "S256" or not query.get("code_challenge"):
            return (
                400,
                {"error": "invalid_request", "error_description": "PKCE S256 required"},
                None,
            )
        code = secrets.token_urlsafe(16)
        with self.lock:
            self.codes[code] = {
                "client_id": client["client_id"],
                "redirect_uri": query["redirect_uri"],
                "code_challenge": query["code_challenge"],
                "scope": query.get("scope", ""),
                "resource": query.get("resource"),
                "offline": query.get("access_type") == "offline",
            }
        answer = {"code": code, **({"state": query["state"]} if "state" in query else {})}
        return 302, {}, f"{query['redirect_uri']}?{urllib.parse.urlencode(answer)}"

    def _token(self, request: AuthorizationRequest) -> tuple[int, dict]:
        client_id, secret = _client_credentials(request)
        with self.lock:
            client = self.clients.get(client_id)
        if client is None or client["client_secret"] != secret:
            return 401, {"error": "invalid_client"}
        form = request.form
        if form.get("grant_type") == "authorization_code":
            return self._exchange_code(client_id, form)
        if form.get("grant_type") == "refresh_token":
            return self._refresh(client_id, form)
        return 400, {"error": "unsupported_grant_type"}

    def _exchange_code(self, client_id: str, form: dict[str, str]) -> tuple[int, dict]:
        with self.lock:
            grant = self.codes.pop(form.get("code", ""), None)
        if grant is None or grant["client_id"] != client_id:
            return 400, {"error": "invalid_grant"}
        if form.get("redirect_uri") != grant["redirect_uri"]:
            return 400, {"error": "invalid_grant", "error_description": "redirect_uri mismatch"}
        if _s256(form.get("code_verifier", "")) != grant["code_challenge"]:
            return 400, {"error": "invalid_grant", "error_description": "PKCE verification failed"}
        with_refresh_token = grant["offline"] or not self.offline_access_only
        return 200, self._issue_tokens(
            client_id, grant["scope"], grant["resource"], with_refresh_token
        )

    def _refresh(self, client_id: str, form: dict[str, str]) -> tuple[int, dict]:
        with self.lock:
            holder = self.refresh_tokens.get(form.get("refresh_token", ""))
            if holder is not None and holder["client_id"] == client_id:
                del self.refresh_tokens[form["refresh_token"]]
            else:
                holder = None
        if holder is None:
            return 400, {"error": "invalid_grant"}
        return 200, self._issue_tokens(client_id, holder["scope"], holder["resource"])

    def _issue_tokens(
        self, client_id: str, scope: str, resource: str | None, with_refresh_token: bool = True
    ) -> dict:
        access_token, refresh_token = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        answer = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": self.access_token_lifetime,
            "scope": scope,
        }
        grant = {"client_id": client_id, "scope": scope, "resource": resource}
        with self.lock:
            expires_at = int(time.time()) + self.access_token_lifetime
            self.access_tokens[access_token] = {**grant, "expires_at": expires_at}
            if with_refresh_token:
                answer["refresh_token"] = refresh_token
                self.refresh_tokens[refresh_token] = grant
            self.issued.append(answer)
        return answer


def _url_host(host: str, name: str | None) -> str:
    return name or (f"[{host}]" if ":" in host else host)


def serve_authorization_server(
    host: str = "127.0.0.1", name: str | None = None
) -> tuple[McpAuthorizationServer, Callable[[], None]]:
    """Start an authorization server on a daemon thread; returns it with its shutdown function."""
    port = free_port(host)
    server = McpAuthorizationServer(base_url=f"http://{_url_host(host, name)}:{port}")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:
            pass

        def _record(self) -> AuthorizationRequest:
            url = urllib.parse.urlsplit(self.path)
            length = int(self.headers.get("Content-Length", 0))
            entry = AuthorizationRequest(
                method=self.command,
                path=url.path,
                query=dict(urllib.parse.parse_qsl(url.query)),
                headers=dict(self.headers.items()),
                body=self.rfile.read(length).decode() if length else "",
            )
            with server.lock:
                server.requests.append(entry)
            return entry

        def _send(self, status: int, payload: dict, headers: dict | None = None) -> None:
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            for name, value in {"Content-Type": "application/json", **(headers or {})}.items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _send_challenge(self) -> None:
            page = f"<html><body>Just a moment... {html.escape(self.path)}</body></html>".encode()
            self.send_response(403)
            self.send_header("Content-Type", "text/html; charset=UTF-8")
            self.send_header("cf-mitigated", "challenge")
            self.send_header("Content-Length", str(len(page)))
            self.end_headers()
            self.wfile.write(page)

        def do_GET(self) -> None:
            entry = self._record()
            # Cloudflare challenges the server-side client and lets the browser through
            from_server = "aiohttp" in entry.headers.get("User-Agent", "")
            if entry.path == "/.well-known/oauth-authorization-server":
                self._send(200, server.metadata())
            elif entry.path == "/authorize" and server.bot_challenge and from_server:
                self._send_challenge()
            elif entry.path == "/authorize":
                status, error, location = server._authorize(entry.query)
                self._send(status, error, {"Location": location} if location else None)
            else:
                self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:
            entry = self._record()
            if entry.path == "/register":
                self._send(*server._register(json.loads(entry.body or "{}")))
            elif entry.path == "/token":
                self._send(*server._token(entry))
            else:
                self._send(404, {"error": "not_found"})

    server_class = IPv6HTTPServer if ":" in host else ThreadingHTTPServer
    http_server = server_class((host, port), Handler)
    threading.Thread(target=http_server.serve_forever, daemon=True).start()

    def shutdown() -> None:
        http_server.shutdown()
        http_server.server_close()

    return server, shutdown


@dataclass
class IssuedTokenVerifier:
    """The MCP server's token check: a live token of `auth_server`, issued for this resource."""

    auth_server: McpAuthorizationServer
    resource_url: str
    presented: list[str] = field(default_factory=list)

    async def verify_token(self, token: str) -> AccessToken | None:
        self.presented.append(token)
        issued = self.auth_server.access_token(token)
        if issued is None or issued["resource"] not in (None, self.resource_url):
            return None
        return AccessToken(
            token=token,
            client_id=issued["client_id"],
            scopes=issued["scope"].split(),
            expires_at=issued["expires_at"],
            resource=issued["resource"],
        )


@dataclass
class ProtectedMcp:
    url: str
    auth_server: McpAuthorizationServer
    verifier: IssuedTokenVerifier

    @property
    def presented(self) -> list[str]:
        return self.verifier.presented


@contextmanager
def serving_protected_mcp(
    host: str = "127.0.0.1", name: str | None = None, scopes: list[str] | None = None
) -> Iterator[ProtectedMcp]:
    auth_server, shutdown = serve_authorization_server(host, name)
    port = free_port(host)
    resource_url = f"http://{_url_host(host, name)}:{port}/mcp"
    verifier = IssuedTokenVerifier(auth_server, resource_url)
    settings = AuthSettings(
        issuer_url=auth_server.issuer,
        resource_server_url=resource_url,
        required_scopes=scopes or [SCOPE],
    )
    try:
        # the SDK's Host check admits only loopback names, and a test reaches this one by others
        transport = TransportSecuritySettings(enable_dns_rebinding_protection=False)
        with serving_mcp(
            port,
            host=host,
            name=name,
            auth=settings,
            token_verifier=verifier,
            transport_security=transport,
        ) as url:
            yield ProtectedMcp(url=url, auth_server=auth_server, verifier=verifier)
    finally:
        shutdown()


def register_oauth_mcp(admin, mcp: ProtectedMcp, server_id: str, access_grants: list[dict]) -> None:
    """Register `mcp` as the admin panel does and save the connection to it."""
    with admin.client() as client:
        registered = client.post(
            "/api/v1/configs/oauth/clients/register",
            params={"type": "mcp"},
            json={"url": mcp.url, "client_id": server_id},
        )
        assert registered.status_code == 200, registered.text
        connection = {
            "url": mcp.url,
            "path": "",
            "type": "mcp",
            "auth_type": "oauth_2.1",
            "key": "",
            "config": {"enable": True, "access_grants": access_grants},
            "info": {
                "id": server_id,
                "name": server_id,
                "oauth_client_info": registered.json()["oauth_client_info"],
            },
        }
        saved = client.post(TOOL_SERVERS[1], json={"TOOL_SERVER_CONNECTIONS": [connection]})
    assert saved.status_code == 200, saved.text
