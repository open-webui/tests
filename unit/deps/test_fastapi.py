"""Dependency contract: fastapi.

fastapi is the web framework the entire Open WebUI backend is built on:
`main.py` constructs the `FastAPI` app (with `lifespan=`, `docs_url`,
`openapi_url`), mounts ~40 `APIRouter`s, and stacks middleware; every
router file declares routes with `Depends(...)` auth dependencies,
`File`/`Form`/`UploadFile` uploads, `Query`/`Header` params, raises
`HTTPException(status_code=status.HTTP_*, detail=...)`, and returns
`JSONResponse`/`StreamingResponse`/`FileResponse`/`RedirectResponse`.
Hot paths offload sync work via `fastapi.concurrency.run_in_threadpool`,
and auth uses `fastapi.security.HTTPBearer`.

This module pins the slice of the fastapi API the codebase actually
relies on so a bump (e.g. 0.135 -> 0.136) that removed, renamed, or
changed any of it fails loudly here instead of surfacing as an
AttributeError / signature error deep in a request path.

Two layers, following the unit/deps/ exemplar (test_requests.py):
symbol-existence checks (API surface) + offline behavioural contracts.
The behavioural tests build a tiny in-process app and exercise it with
fastapi.testclient.TestClient; TestClient is in-process (ASGI), so there
is no real network and the suite stays deterministic. Uses the
`depcheck` fixture from unit/deps/conftest.py.
"""

import io

import pytest

# NOTE: deliberately no `from __future__ import annotations` here. These tests
# define live FastAPI routes (and a couple of local pydantic models) whose
# annotations FastAPI/pydantic evaluate at runtime. Under PEP 563 those
# annotations become strings that can't resolve against function-local scope,
# so FastAPI would misclassify typed params (e.g. treat `request: Request` as a
# query param) and the behavioural contracts would falsely fail.

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "fastapi"
DIST_NAME = "fastapi"

# ---------------------------------------------------------------------------
# Symbols the Open WebUI backend imports from fastapi and its submodules.
# Dotted names resolve nested attributes / submodules (see conftest DepCheck).
# Derived from grepping every `from fastapi(...) import` across the backend.
# ---------------------------------------------------------------------------

# Top-level `from fastapi import ...`
TOPLEVEL_SYMBOLS = [
    "FastAPI",
    "APIRouter",
    "Depends",
    "HTTPException",
    "status",
    "Request",
    "Response",
    "BackgroundTasks",
    "UploadFile",
    "File",
    "Form",
    "Query",
    "Header",
    "WebSocket",
    "applications",
]

# Submodule-qualified imports the backend uses.
SUBMODULE_SYMBOLS = [
    "concurrency.run_in_threadpool",
    "middleware.cors.CORSMiddleware",
    "openapi.docs.get_swagger_ui_html",
    "responses.JSONResponse",
    "responses.StreamingResponse",
    "responses.FileResponse",
    "responses.RedirectResponse",
    "responses.HTMLResponse",
    "responses.PlainTextResponse",
    "responses.Response",
    "security.HTTPBearer",
    "security.HTTPAuthorizationCredentials",
    "staticfiles.StaticFiles",
    "encoders.jsonable_encoder",
    "testclient.TestClient",
]

# status.HTTP_* constants referenced across the backend, with expected values.
# Pinning the values catches a constant being renamed/dropped AND any unlikely
# numeric drift.
STATUS_CONSTANTS = {
    "HTTP_200_OK": 200,
    "HTTP_201_CREATED": 201,
    "HTTP_204_NO_CONTENT": 204,
    "HTTP_302_FOUND": 302,
    "HTTP_400_BAD_REQUEST": 400,
    "HTTP_401_UNAUTHORIZED": 401,
    "HTTP_403_FORBIDDEN": 403,
    "HTTP_404_NOT_FOUND": 404,
    "HTTP_409_CONFLICT": 409,
    "HTTP_429_TOO_MANY_REQUESTS": 429,
    "HTTP_500_INTERNAL_SERVER_ERROR": 500,
    "HTTP_502_BAD_GATEWAY": 502,
    "HTTP_503_SERVICE_UNAVAILABLE": 503,
    "HTTP_504_GATEWAY_TIMEOUT": 504,
}


# --------------------------------------------------------------------------- #
# Import / version sanity
# --------------------------------------------------------------------------- #


def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "fastapi"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_has_version_attr(depcheck):
    """fastapi exposes __version__; some code/log paths read it."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.__version__, str)
    assert mod.__version__  # non-empty


# --------------------------------------------------------------------------- #
# Symbol-existence: the API surface the backend imports
# --------------------------------------------------------------------------- #


def test_toplevel_symbols_exist(depcheck):
    """Every name in `from fastapi import ...` across the backend must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, TOPLEVEL_SYMBOLS)


def test_submodule_symbols_exist(depcheck):
    """Submodule-qualified imports (responses.*, concurrency.*, security.*,
    middleware.cors.*, openapi.docs.*, staticfiles.*, encoders.*, testclient.*)
    must all resolve."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, SUBMODULE_SYMBOLS)


def test_fastapi_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.FastAPI, type)


def test_apirouter_is_class(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.APIRouter, type)


def test_httpexception_is_exception_class(depcheck):
    """HTTPException must be a class derived from Exception so the backend's
    hundreds of `raise HTTPException(...)` and `except HTTPException` work."""
    mod = depcheck.load(IMPORT_NAME)
    assert isinstance(mod.HTTPException, type)
    assert issubclass(mod.HTTPException, Exception)


def test_depends_file_form_query_header_callable(depcheck):
    """Depends/File/Form/Query/Header are the dependency/param markers used in
    route signatures; all must be callable factories."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("Depends", "File", "Form", "Query", "Header"):
        depcheck.assert_callable(mod, name)


def test_response_classes_exist_and_are_classes(depcheck):
    """The response classes the backend returns must all be classes."""
    mod = depcheck.load(IMPORT_NAME)
    responses = depcheck.resolve(mod, "responses")
    for name in (
        "JSONResponse",
        "StreamingResponse",
        "FileResponse",
        "RedirectResponse",
        "HTMLResponse",
        "PlainTextResponse",
        "Response",
    ):
        cls = getattr(responses, name)
        assert isinstance(cls, type), f"responses.{name} is not a class"


def test_run_in_threadpool_callable(depcheck):
    """Hot paths await run_in_threadpool(sync_fn, ...); it must be callable."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "concurrency.run_in_threadpool")


def test_jsonable_encoder_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "encoders.jsonable_encoder")


def test_get_swagger_ui_html_callable(depcheck):
    """main.py overrides /docs via applications.get_swagger_ui_html."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_callable(mod, "openapi.docs.get_swagger_ui_html")
    # main.py also reaches it as fastapi.applications.get_swagger_ui_html.
    depcheck.assert_callable(mod, "applications.get_swagger_ui_html")


def test_security_classes_exist(depcheck):
    """utils/auth.py uses HTTPBearer + HTTPAuthorizationCredentials."""
    mod = depcheck.load(IMPORT_NAME)
    sec = depcheck.resolve(mod, "security")
    assert isinstance(sec.HTTPBearer, type)
    assert isinstance(sec.HTTPAuthorizationCredentials, type)


# --------------------------------------------------------------------------- #
# status constants
# --------------------------------------------------------------------------- #


def test_status_constants_exist_and_have_expected_values(depcheck):
    """Every status.HTTP_* the backend references must exist with its value.
    Reported all-at-once so a bump that drops one shows the full gap."""
    mod = depcheck.load(IMPORT_NAME)
    status = mod.status
    problems = []
    for name, expected in STATUS_CONSTANTS.items():
        if not hasattr(status, name):
            problems.append(f"{name}: MISSING")
            continue
        actual = getattr(status, name)
        if actual != expected:
            problems.append(f"{name}: {actual} (expected {expected})")
    assert not problems, f"fastapi.status changed: {problems}"


# --------------------------------------------------------------------------- #
# Constructor / signature contracts
# --------------------------------------------------------------------------- #


def test_fastapi_constructor_accepts_our_kwargs(depcheck):
    """main.py: FastAPI(title=, docs_url=, openapi_url=, redoc_url=, lifespan=).
    Those keywords must remain accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.FastAPI.__init__,
        ["title", "docs_url", "openapi_url", "redoc_url", "lifespan"],
    )


def test_apirouter_constructor_accepts_prefix_tags(depcheck):
    """Routers are declared `APIRouter()` and mounted with prefix=/tags=;
    include_router(prefix=, tags=) is the call site, but APIRouter itself also
    accepts prefix/tags. Assert the constructor still takes them."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.APIRouter.__init__, ["prefix", "tags"])


def test_httpexception_constructor_accepts_status_detail_headers(depcheck):
    """The backend raises HTTPException(status_code=, detail=, headers=)."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.HTTPException.__init__,
        ["status_code", "detail", "headers"],
    )


def test_httpexception_carries_status_and_detail(depcheck):
    """A constructed HTTPException must expose .status_code and .detail (read
    by exception handlers and re-raise sites)."""
    mod = depcheck.load(IMPORT_NAME)
    exc = mod.HTTPException(status_code=404, detail="nope")
    assert exc.status_code == 404
    assert exc.detail == "nope"


def test_streamingresponse_accepts_media_type(depcheck):
    """StreamingResponse(stream(), media_type='text/event-stream') is the SSE
    pattern used across chat/functions/ollama; media_type must be accepted."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.responses.StreamingResponse.__init__,
        ["content", "media_type"],
    )


def test_jsonresponse_accepts_status_code_and_headers(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(
        mod.responses.JSONResponse.__init__,
        ["content", "status_code", "headers"],
    )


def test_add_middleware_and_include_router_and_mount_exist(depcheck):
    """main.py calls app.add_middleware(...), app.include_router(...),
    app.mount(...). Assert FastAPI exposes them as callables."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("add_middleware", "include_router", "mount", "get", "post"):
        attr = getattr(mod.FastAPI, name, None)
        assert callable(attr), f"FastAPI.{name} missing/not callable"


def test_apirouter_has_http_method_decorators(depcheck):
    """Routers decorate handlers with @router.get/post/put/delete/patch and
    @router.websocket; those decorator factories must exist."""
    mod = depcheck.load(IMPORT_NAME)
    for name in ("get", "post", "put", "delete", "patch", "websocket"):
        attr = getattr(mod.APIRouter, name, None)
        assert callable(attr), f"APIRouter.{name} missing/not callable"


# --------------------------------------------------------------------------- #
# Behavioural contracts (offline, in-process via TestClient)
# --------------------------------------------------------------------------- #


def _make_client(mod):
    """Build a tiny app mirroring the backend's shapes and wrap it in a
    TestClient. Returns (client, app). Skips if TestClient can't be built
    (e.g. httpx missing in the env)."""
    TestClient = depcheck_testclient(mod)
    app = mod.FastAPI(title="depcheck", docs_url=None, openapi_url=None)
    return TestClient, app


def depcheck_testclient(mod):
    try:
        from fastapi.testclient import TestClient
    except Exception as e:  # pragma: no cover - depends on env (needs httpx)
        pytest.skip(f"fastapi.testclient.TestClient unavailable: {e}")
    return TestClient


def test_router_mounting_and_get_route(depcheck):
    """An APIRouter mounted via include_router(prefix=...) must serve a GET and
    return the JSON body FastAPI auto-encodes from the return value."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    router = mod.APIRouter()

    @router.get("/ping")
    def ping():
        return {"pong": True}

    app.include_router(router, prefix="/api", tags=["t"])

    with TestClient(app) as client:
        resp = client.get("/api/ping")
        assert resp.status_code == 200
        assert resp.json() == {"pong": True}


def test_depends_injection(depcheck):
    """Depends(...) must inject the dependency's return value into the handler
    — this is exactly how every route receives `user=Depends(get_*_user)`."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    def current_user():
        return {"id": "u1", "role": "admin"}

    @app.get("/me")
    def me(user=mod.Depends(current_user)):
        return {"role": user["role"]}

    with TestClient(app) as client:
        resp = client.get("/me")
        assert resp.status_code == 200
        assert resp.json() == {"role": "admin"}


def test_dependency_raising_httpexception_maps_to_status(depcheck):
    """A dependency raising HTTPException(status_code=401) must short-circuit
    the request with that status and the detail in the body — the backend's
    auth-guard pattern (get_verified_user raising 401)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    def require_auth():
        raise mod.HTTPException(
            status_code=mod.status.HTTP_401_UNAUTHORIZED, detail="Not authenticated"
        )

    @app.get("/secret")
    def secret(_=mod.Depends(require_auth)):
        return {"ok": True}  # pragma: no cover - never reached

    with TestClient(app) as client:
        resp = client.get("/secret")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Not authenticated"


def test_handler_raising_httpexception_with_headers(depcheck):
    """HTTPException headers must propagate onto the HTTP response (used e.g.
    for WWW-Authenticate / rate-limit headers)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/boom")
    def boom():
        raise mod.HTTPException(
            status_code=mod.status.HTTP_403_FORBIDDEN,
            detail="forbidden",
            headers={"X-Reason": "policy"},
        )

    with TestClient(app) as client:
        resp = client.get("/boom")
        assert resp.status_code == 403
        assert resp.headers.get("X-Reason") == "policy"


def test_path_and_query_params(depcheck):
    """Path params and Query(default, ...) must bind from the URL — folders /
    knowledge / analytics routers rely on this."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/items/{item_id}")
    def get_item(item_id: str, limit: int = mod.Query(10)):
        return {"item_id": item_id, "limit": limit}

    with TestClient(app) as client:
        resp = client.get("/items/abc?limit=3")
        assert resp.status_code == 200
        assert resp.json() == {"item_id": "abc", "limit": 3}
        # default applies when omitted
        resp2 = client.get("/items/xyz")
        assert resp2.json()["limit"] == 10


def test_query_validation_returns_422(depcheck):
    """A non-coercible query value must yield FastAPI's 422 validation error,
    confirming pydantic-backed request validation is wired."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/n")
    def n(value: int = mod.Query(...)):
        return {"value": value}

    with TestClient(app) as client:
        assert client.get("/n?value=notanint").status_code == 422
        assert client.get("/n?value=5").json() == {"value": 5}


def test_header_param(depcheck):
    """Header(None) binds a request header (scim.py: authorization Header)."""
    mod = depcheck.load(IMPORT_NAME)
    from typing import Optional

    TestClient, app = _make_client(mod)

    @app.get("/whoami")
    def whoami(authorization: Optional[str] = mod.Header(None)):
        return {"authorization": authorization}

    with TestClient(app) as client:
        resp = client.get("/whoami", headers={"Authorization": "Bearer xyz"})
        assert resp.json() == {"authorization": "Bearer xyz"}
        # absent header -> None default
        assert client.get("/whoami").json() == {"authorization": None}


def test_file_upload_via_uploadfile(depcheck):
    """`file: UploadFile = File(...)` must receive multipart uploads and expose
    .filename + an awaitable .read() — files/audio/ollama upload routes."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.post("/upload")
    async def upload(file: mod.UploadFile = mod.File(...)):
        data = await file.read()
        return {"name": file.filename, "size": len(data)}

    with TestClient(app) as client:
        resp = client.post(
            "/upload",
            files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        if resp.status_code == 500 and "multipart" in resp.text.lower():
            pytest.skip("python-multipart not installed; upload path needs it")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"name": "hello.txt", "size": 11}


def test_form_field(depcheck):
    """Form(...) binds a urlencoded/multipart form field (auths/configs use
    Form heavily)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.post("/login")
    def login(username: str = mod.Form(...)):
        return {"username": username}

    with TestClient(app) as client:
        resp = client.post("/login", data={"username": "alice"})
        if resp.status_code == 500 and "multipart" in resp.text.lower():
            pytest.skip("python-multipart not installed; form path needs it")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"username": "alice"}


def test_jsonresponse_with_explicit_status(depcheck):
    """Returning a JSONResponse(content=, status_code=) must set both body and
    status — used where routes need a non-200 JSON reply."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/created")
    def created():
        return mod.responses.JSONResponse(
            content={"created": True}, status_code=mod.status.HTTP_201_CREATED
        )

    with TestClient(app) as client:
        resp = client.get("/created")
        assert resp.status_code == 201
        assert resp.json() == {"created": True}


def test_streaming_response_streams_chunks(depcheck):
    """StreamingResponse over a generator must stream the concatenated bytes
    with the given media_type — the SSE streaming contract."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    def gen():
        yield b"a"
        yield b"b"
        yield b"c"

    @app.get("/stream")
    def stream():
        return mod.responses.StreamingResponse(gen(), media_type="text/plain")

    with TestClient(app) as client:
        resp = client.get("/stream")
        assert resp.status_code == 200
        assert resp.content == b"abc"
        assert resp.headers["content-type"].startswith("text/plain")


def test_redirect_response(depcheck):
    """RedirectResponse must emit a 3xx with a Location header (OAuth callback
    + auth flows use it)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/go")
    def go():
        return mod.responses.RedirectResponse(url="/target", status_code=mod.status.HTTP_302_FOUND)

    with TestClient(app) as client:
        resp = client.get("/go", follow_redirects=False)
        assert resp.status_code == 302
        assert resp.headers["location"] == "/target"


def test_plaintext_response(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/txt")
    def txt():
        return mod.responses.PlainTextResponse("hello")

    with TestClient(app) as client:
        resp = client.get("/txt")
        assert resp.status_code == 200
        assert resp.text == "hello"
        assert resp.headers["content-type"].startswith("text/plain")


def test_request_object_exposes_state_headers_query(depcheck):
    """Handlers take `request: Request` and read request.headers /
    request.query_params / request.url / request.app; pin that shape."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.get("/echo")
    def echo(request: mod.Request):
        return {
            "ua": request.headers.get("user-agent", ""),
            "q": request.query_params.get("x", ""),
            "path": request.url.path,
            "has_app": request.app is not None,
        }

    with TestClient(app) as client:
        resp = client.get("/echo?x=42", headers={"User-Agent": "pytest"})
        body = resp.json()
        assert body["q"] == "42"
        assert body["path"] == "/echo"
        assert body["ua"] == "pytest"
        assert body["has_app"] is True


def test_background_tasks_run_after_response(depcheck):
    """BackgroundTasks injected into a handler must run its callback after the
    response is sent (notes/channels schedule work this way)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    sink = {}

    def record(value):
        sink["value"] = value

    @app.post("/bg")
    def bg(background_tasks: mod.BackgroundTasks):
        background_tasks.add_task(record, "done")
        return {"scheduled": True}

    with TestClient(app) as client:
        resp = client.post("/bg")
        assert resp.status_code == 200
        assert resp.json() == {"scheduled": True}
    # TestClient context exit guarantees background tasks completed.
    assert sink.get("value") == "done"


def test_add_middleware_runs(depcheck):
    """app.add_middleware(...) must install an ASGI middleware that can mutate
    responses — main.py stacks several (CORS, security headers, etc.)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    class HeaderMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = message.setdefault("headers", [])
                    headers.append((b"x-mw", b"on"))
                await send(message)

            await self.app(scope, receive, send_wrapper)

    app.add_middleware(HeaderMiddleware)

    @app.get("/m")
    def m():
        return {"ok": True}

    with TestClient(app) as client:
        resp = client.get("/m")
        assert resp.status_code == 200
        assert resp.headers.get("x-mw") == "on"


def test_cors_middleware_adds_headers(depcheck):
    """CORSMiddleware (main.py + retrieval.py) must add
    access-control-allow-origin for an allowed cross-origin request."""
    mod = depcheck.load(IMPORT_NAME)
    from fastapi.middleware.cors import CORSMiddleware

    TestClient, app = _make_client(mod)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://example.com"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/c")
    def c():
        return {"ok": True}

    with TestClient(app) as client:
        resp = client.get("/c", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://example.com"


def test_httpbearer_extracts_credentials(depcheck):
    """utils/auth.py uses HTTPBearer(auto_error=False); as a dependency it must
    parse an `Authorization: Bearer <tok>` header into an object exposing
    .scheme and .credentials."""
    mod = depcheck.load(IMPORT_NAME)
    from typing import Optional

    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    TestClient, app = _make_client(mod)
    bearer = HTTPBearer(auto_error=False)

    @app.get("/auth")
    def auth(creds: Optional[HTTPAuthorizationCredentials] = mod.Depends(bearer)):
        if creds is None:
            return {"scheme": None, "token": None}
        return {"scheme": creds.scheme, "token": creds.credentials}

    with TestClient(app) as client:
        resp = client.get("/auth", headers={"Authorization": "Bearer tok123"})
        assert resp.json() == {"scheme": "Bearer", "token": "tok123"}
        # auto_error=False -> no header yields None, not a 403.
        assert client.get("/auth").json() == {"scheme": None, "token": None}


def test_run_in_threadpool_runs_sync_fn(depcheck):
    """run_in_threadpool must execute a *sync* function off the event loop and
    return its result (hot paths offload blocking work this way)."""
    mod = depcheck.load(IMPORT_NAME)
    import asyncio

    run_in_threadpool = mod.concurrency.run_in_threadpool

    def add(a, b):
        return a + b

    result = asyncio.run(run_in_threadpool(add, 2, 3))
    assert result == 5


def test_run_in_threadpool_inside_route(depcheck):
    """End-to-end: an async route awaiting run_in_threadpool returns the
    offloaded computation — mirrors evaluations/retrieval routers."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    def heavy(n):
        return n * n

    @app.get("/sq/{n}")
    async def sq(n: int):
        value = await mod.concurrency.run_in_threadpool(heavy, n)
        return {"value": value}

    with TestClient(app) as client:
        assert client.get("/sq/9").json() == {"value": 81}


def test_jsonable_encoder_serializes_pydantic_model(depcheck):
    """jsonable_encoder must turn a pydantic model into JSON-native types
    (dict of primitives) — the canonical FastAPI serialization primitive."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover - pydantic is a hard dep of fastapi
        pytest.skip(f"pydantic unavailable: {e}")

    jsonable_encoder = mod.encoders.jsonable_encoder

    class Item(BaseModel):
        id: int
        name: str
        tags: list[str] = []

    encoded = jsonable_encoder(Item(id=1, name="x", tags=["a", "b"]))
    assert encoded == {"id": 1, "name": "x", "tags": ["a", "b"]}
    assert isinstance(encoded, dict)


def test_jsonable_encoder_handles_nested_and_datetime(depcheck):
    """jsonable_encoder must recurse into nested structures and stringify
    datetimes (responses embed timestamps)."""
    mod = depcheck.load(IMPORT_NAME)
    from datetime import datetime, timezone

    jsonable_encoder = mod.encoders.jsonable_encoder
    dt = datetime(2024, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    encoded = jsonable_encoder({"when": dt, "nested": {"vals": (1, 2)}})
    assert isinstance(encoded["when"], str)
    assert "2024-01-02" in encoded["when"]
    assert encoded["nested"] == {"vals": [1, 2]}


def test_response_model_filters_output(depcheck):
    """A route declared with response_model must coerce/filter the return to
    the model's fields — used widely for typed API responses."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pydantic unavailable: {e}")

    TestClient, app = _make_client(mod)

    class Out(BaseModel):
        id: int
        name: str

    @app.get("/u", response_model=Out)
    def u():
        # Extra field must be stripped by response_model serialization.
        return {"id": 7, "name": "n", "secret": "leak"}

    with TestClient(app) as client:
        body = client.get("/u").json()
        assert body == {"id": 7, "name": "n"}
        assert "secret" not in body


def test_pydantic_body_model_parsing_and_422(depcheck):
    """A pydantic model param is parsed from the JSON body; a missing required
    field yields 422 (every router posts a `form_data: SomeForm` model)."""
    mod = depcheck.load(IMPORT_NAME)
    try:
        from pydantic import BaseModel
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pydantic unavailable: {e}")

    TestClient, app = _make_client(mod)

    class Form(BaseModel):
        name: str
        count: int = 1

    @app.post("/make")
    def make(form_data: Form):
        return {"name": form_data.name, "count": form_data.count}

    with TestClient(app) as client:
        assert client.post("/make", json={"name": "z"}).json() == {
            "name": "z",
            "count": 1,
        }
        assert client.post("/make", json={"count": 2}).status_code == 422


def test_websocket_route_roundtrip(depcheck):
    """terminals.py declares @router.websocket(...) with a WebSocket param;
    TestClient.websocket_connect must round-trip a message."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    @app.websocket("/ws")
    async def ws(websocket: mod.WebSocket):
        await websocket.accept()
        msg = await websocket.receive_text()
        await websocket.send_text(f"echo:{msg}")
        await websocket.close()

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as conn:
            conn.send_text("hi")
            assert conn.receive_text() == "echo:hi"


def test_exception_handler_registration(depcheck):
    """app.add_exception_handler / @app.exception_handler must let a custom
    handler convert an exception to a Response (used for unified error JSON)."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    class MyError(Exception):
        pass

    @app.exception_handler(MyError)
    async def handle(request, exc):
        return mod.responses.JSONResponse(
            status_code=mod.status.HTTP_400_BAD_REQUEST, content={"error": str(exc)}
        )

    @app.get("/err")
    def err():
        raise MyError("bad")

    with TestClient(app) as client:
        resp = client.get("/err")
        assert resp.status_code == 400
        assert resp.json() == {"error": "bad"}


def test_dependency_override(depcheck):
    """app.dependency_overrides — the standard test seam — must replace a
    dependency's value. Pins that the override map still exists/works."""
    mod = depcheck.load(IMPORT_NAME)
    TestClient, app = _make_client(mod)

    def get_user():
        return {"role": "user"}  # pragma: no cover - overridden below

    @app.get("/role")
    def role(user=mod.Depends(get_user)):
        return {"role": user["role"]}

    app.dependency_overrides[get_user] = lambda: {"role": "admin"}

    with TestClient(app) as client:
        assert client.get("/role").json() == {"role": "admin"}
