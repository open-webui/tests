"""External apps: the reference oix app as a real process, and the admin steps that install it.

`serving_reference_app(...)` starts `harness/reference_app/app.py` on a free port with a fresh
API key and Ed25519 key, and yields it once it answers. It runs from an environment of its own
with `oix[fastapi,redis]` 0.8 or later and `uvicorn`: `OIX_PYTHON` names that interpreter, else
`oix` importable next to the suite is used, and without either the tests that need it skip. CI
builds that environment from `open-webui/oix` at the `oix-ref` input. Open WebUI connects out to the
app over Socket.IO, so the app never calls the instance back. The app appends every event it
handles to a record file, which `app.records(...)` and `app.wait_for(...)` read.

`APPS_ENV` turns Apps and Channels on and lets the instance reach apps on the reserved ports
over plain HTTP. `install(...)` and `connect(...)` are the admin's Add app steps, `invite(...)`
is a channel manager's, and `post(...)` and `mention_text(...)` are what the channel input
sends. `RawApp` is an app endpoint the test scripts by hand, for handshakes no SDK would send.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from harness.instance import free_port

APP_SOURCE = Path(__file__).parent / "reference_app" / "app.py"
ALL_SCOPES = [
    "channels:mentions:read",
    "channels:messages:read",
    "channels:messages:write",
    "channels:read",
]
ADMIN_CONFIG = "/api/v1/auths/admin/config"
BOOT_SECONDS = 30
# the instance only reaches plain-HTTP apps it was told about at boot, so apps take these ports
APP_PORTS = [free_port() for _ in range(6)]
APPS_ENV = {
    "ENABLE_APPS": "true",
    "ENABLE_CHANNELS": "true",
    "APP_ALLOWED_PRIVATE_ORIGINS": ",".join(f"http://127.0.0.1:{port}" for port in APP_PORTS),
}


def private_key_pem() -> str:
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def public_key_pem(private_pem: str) -> str:
    key = serialization.load_pem_private_key(private_pem.encode(), password=None)
    return (
        key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )


def oix_python() -> str:
    configured = os.getenv("OIX_PYTHON")
    if configured:
        return configured
    probe = subprocess.run([sys.executable, "-c", "import oix, uvicorn"], capture_output=True)
    if probe.returncode == 0:
        return sys.executable
    pytest.skip("the oix SDK is not installed (set OIX_PYTHON to an interpreter that has it)")


@dataclass
class ReferenceApp:
    url: str
    api_key: str
    private_key: str
    manifest_id: str
    name: str
    scopes: list[str]
    record_file: Path
    log_file: Path
    process: subprocess.Popen | None = field(default=None, repr=False)

    @property
    def origin(self) -> str:
        return self.url.split("/", 3)[0] + "//" + self.url.split("/", 3)[2]

    @property
    def public_key(self) -> str:
        return public_key_pem(self.private_key)

    def manifest(self, **overrides) -> dict:
        """The manifest the app serves, for a manual install."""
        return {
            "id": self.manifest_id,
            "name": self.name,
            "description": "The regression suite's reference app.",
            "public_key": self.public_key,
            "requested_scopes": list(self.scopes),
            **overrides,
        }

    def records(self, kind: str | None = None, **match) -> list[dict]:
        if not self.record_file.exists():
            return []
        lines = self.record_file.read_text(encoding="utf-8").splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        return [
            entry
            for entry in entries
            if (kind is None or entry["kind"] == kind)
            and all(entry.get(name) == value for name, value in match.items())
        ]

    def wait_for(self, kind: str, timeout: float = 20.0, **match) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.records(kind, **match)
            if found:
                return found[0]
            time.sleep(0.1)
        raise AssertionError(f"the app recorded no {kind} {match} within {timeout}s: {self.log()}")

    def log(self) -> str:
        return self.log_file.read_text(encoding="utf-8", errors="replace")[-3000:]

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()


@contextlib.contextmanager
def serving_reference_app(
    scopes: list[str] | None = None,
    name: str = "Reference App",
    manifest_id: str | None = None,
    server: str = "builtin",
    path: str = "/events",
    manifest: bool = True,
    ignore_self: bool = True,
    api_key: str | None = None,
    private_key: str | None = None,
    port: int | None = None,
    profile_image: Path | None = None,
    redis_url: str | None = None,
    admission: bool = False,
) -> Iterator[ReferenceApp]:
    """The reference app answering on a free loopback port until the block ends."""
    python = oix_python()
    scratch = Path(tempfile.mkdtemp(prefix="owui-app-"))
    port = port or unused_app_port()
    scopes = scopes if scopes is not None else ALL_SCOPES
    reference = ReferenceApp(
        url=f"http://127.0.0.1:{port}{path}",
        api_key=api_key or secrets.token_hex(32),
        private_key=private_key or private_key_pem(),
        manifest_id=manifest_id or f"reference-{uuid.uuid4().hex[:8]}",
        name=name,
        scopes=scopes,
        record_file=scratch / "records.jsonl",
        log_file=scratch / "app.log",
    )
    key_file = scratch / "app-private-key.pem"
    key_file.write_text(reference.private_key)
    env = {
        **os.environ,
        "API_KEY": reference.api_key,
        "APP_PRIVATE_KEY_FILE": str(key_file),
        "APP_PROFILE_IMAGE": str(profile_image) if profile_image else "",
        "APP_REDIS_URL": redis_url or "",
        "APP_PORT": str(port),
        "APP_PATH": path,
        "APP_SCOPES": ",".join(scopes),
        "APP_NAME": name,
        "APP_MANIFEST_ID": reference.manifest_id,
        "APP_SERVER": server,
        "APP_MANIFEST": "true" if manifest else "false",
        "APP_IGNORE_SELF": "true" if ignore_self else "false",
        "APP_ADMISSION": "true" if admission else "false",
        "APP_RECORD_FILE": str(reference.record_file),
        "PYTHONUNBUFFERED": "1",
    }
    try:
        with open(reference.log_file, "w", encoding="utf-8") as log:
            reference.process = subprocess.Popen(
                [python, str(APP_SOURCE)], env=env, stdout=log, stderr=subprocess.STDOUT
            )
        _wait_until_listening(reference, port)
        yield reference
    finally:
        reference.stop()
        shutil.rmtree(scratch, ignore_errors=True)


def unused_app_port() -> int:
    """A port the instance accepts plain-HTTP apps on that nothing listens on yet."""
    for port in APP_PORTS:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise AssertionError(f"every app port is taken: {APP_PORTS}")


def _wait_until_listening(reference: ReferenceApp, port: int) -> None:
    deadline = time.monotonic() + BOOT_SECONDS
    while time.monotonic() < deadline:
        if reference.process.poll() is not None:
            pytest.fail(f"the reference app exited during boot:\n{reference.log()}")
        try:
            # any answer means it listens; without the key it is a 401 or a 405
            httpx.get(reference.url, timeout=2)
            return
        except httpx.HTTPError:
            time.sleep(0.2)
    pytest.fail(f"the reference app did not start within {BOOT_SECONDS}s:\n{reference.log()}")


def preview(client: httpx.Client, app: ReferenceApp, **form) -> httpx.Response:
    return client.post(
        "/api/v1/apps/manifest/preview", json={"url": app.url, "api_key": app.api_key, **form}
    )


def install(
    client: httpx.Client,
    app: ReferenceApp,
    scopes: list[str] | None = None,
    access_grants: list[dict] | None = None,
    slug: str | None = None,
    manifest: dict | None = None,
) -> httpx.Response:
    """The admin's Add app form after Continue: the discovered manifest plus the reviewed scopes."""
    form = {
        "manifest": manifest or app.manifest(),
        "url": app.url,
        "api_key": app.api_key,
        "slug": slug if slug is not None else f"app-{uuid.uuid4().hex[:8]}",
        "granted_scopes": scopes if scopes is not None else list(app.scopes),
    }
    if access_grants is not None:
        form["access_grants"] = access_grants
    return client.post("/api/v1/apps", json=form)


def connect(client: httpx.Client, app_id: str, api_key: str | None = None) -> httpx.Response:
    form = {"api_key": api_key} if api_key else {}
    return client.post(f"/api/v1/apps/{app_id}/connect", json=form)


def installed(
    client: httpx.Client,
    app: ReferenceApp,
    scopes: list[str] | None = None,
    access_grants: list[dict] | None = None,
    manifest: dict | None = None,
) -> dict:
    """Install and connect, as one press of Connect does; returns the connected app."""
    created = install(client, app, scopes=scopes, access_grants=access_grants, manifest=manifest)
    assert created.status_code == 200, f"installing the app failed: {created.text}"
    connected = connect(client, created.json()["id"])
    assert connected.status_code == 200, f"connecting the app failed: {connected.text}"
    assert connected.json()["is_active"] is True
    return connected.json()


def new_channel(client: httpx.Client, **form) -> str:
    created = client.post(
        "/api/v1/channels/create", json={"name": f"apps-{uuid.uuid4().hex[:8]}", **form}
    )
    assert created.status_code == 200, f"creating a channel failed: {created.text}"
    return created.json()["id"]


def invite(client: httpx.Client, channel_id: str, app_id: str) -> httpx.Response:
    return client.post(f"/api/v1/channels/{channel_id}/apps", json={"app_id": app_id})


def mention_text(app: dict, text: str) -> str:
    """A message that mentions the app the way the channel input writes a picked mention."""
    return f"<@A:{app['id']}|{app['name']}> {text}"


def post(client: httpx.Client, channel_id: str, content: str, **form) -> dict:
    posted = client.post(
        f"/api/v1/channels/{channel_id}/messages/post", json={"content": content, **form}
    )
    assert posted.status_code == 200, f"posting to the channel failed: {posted.text}"
    return posted.json()


def channel_messages(client: httpx.Client, channel_id: str, parent_id: str | None = None) -> list:
    path = (
        f"/api/v1/channels/{channel_id}/messages/{parent_id}/thread"
        if parent_id
        else f"/api/v1/channels/{channel_id}/messages"
    )
    listed = client.get(path)
    listed.raise_for_status()
    return listed.json()


def wait_for_message(
    client: httpx.Client,
    channel_id: str,
    content: str,
    parent_id: str | None = None,
    timeout: float = 20.0,
) -> dict:
    """The first message in the channel (or thread) whose content is `content`."""
    deadline = time.monotonic() + timeout
    seen: list = []
    while time.monotonic() < deadline:
        seen = channel_messages(client, channel_id, parent_id)
        for message in seen:
            if message.get("content") == content:
                return message
        time.sleep(0.2)
    contents = [message.get("content") for message in seen]
    raise AssertionError(f"no message {content!r} in channel {channel_id}: {contents}")


def reply_events(socket, timeout: float = 20.0, **match) -> dict:
    """The first private app reply (`channel:app`) that reached the socket and carries `match`."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in list(socket.app_events):
            if match.items() <= event.items():
                return event
        time.sleep(0.05)
    raise AssertionError(f"no private app reply {match} within {timeout}s: {socket.app_events}")


class RawApp:
    """An app endpoint the test scripts by hand: a Socket.IO server on a reserved port.

    It answers `app:connect` through `answer_connect(data)` (an honest, signed answer by
    default; `signed_proof(data, **claims)` forges single claims), acknowledges `app:ready` once
    the instance answers its ping, and records every event, answering mentions with
    `mention_answer` and button presses with `event_answer`, or both with `answer_with(event)`
    when it is set. `request(operation, **arguments)`
    calls the instance's runtime over the connection the instance opened, as an app's client does.
    """

    def __init__(self, private_pem: str):
        self.private_pem = private_pem
        # its own key: the instance keeps reconnecting earlier installs to these reused ports
        self.api_key = secrets.token_hex(32)
        self.port = unused_app_port()
        self.url = f"http://127.0.0.1:{self.port}/events"
        self.events: list[dict] = []
        self.connects: list[dict] = []
        self.installation_id = str(uuid.uuid4())
        self.sid: str | None = None
        self.ready = False
        self.answer_connect = self.signed_proof
        # what the app answers a button press with; message events always get `received`
        self.event_answer: dict = {"type": "ack"}
        # the whole socket answer to a mention, status included
        self.mention_answer: dict = {"status": 200, "data": {"received": True}}
        # the whole socket answer built from the event, for mentions and presses alike
        self.answer_with: Callable[[dict], dict] | None = None
        # the credential the instance stored at the last honest connection answer
        self.connection_token: str | None = None
        # operations sent while the instance still waits for `app:ready`, and their answers
        self.before_ready: list[str] = []
        self.before_ready_answers: list[dict] = []

    def manifest(self, scopes: list[str] | None = None) -> dict:
        return {
            "id": "raw",
            "name": "Raw App",
            "public_key": public_key_pem(self.private_pem),
            "requested_scopes": scopes if scopes is not None else ALL_SCOPES,
        }

    def signed_proof(self, data: dict, **claims) -> dict:
        """The connection answer an honest app gives; `claims` replace single signed claims."""
        key = serialization.load_pem_private_key(self.private_pem.encode(), password=None)
        token = data.get("connection_token") or jwt.encode(
            {"sub": self.installation_id, "aud": "oix:connection"},
            key,
            algorithm="EdDSA",
            headers={"typ": "oix-connection+jwt"},
        )
        now = int(time.time())
        signed = {
            "sub": self.installation_id,
            "app_id": data["app_id"],
            "manifest_id": "raw",
            "challenge": data["challenge"],
            "token_hash": hashlib.sha256(token.encode()).hexdigest(),
            "iat": now,
            "exp": now + 60,
            "aud": "oix:connect",
            **claims,
        }
        proof = jwt.encode(signed, key, algorithm="EdDSA", headers={"typ": "oix-connect+jwt"})
        self.connection_token = token
        answer = {"installation_id": self.installation_id, "connection_token": token}
        return {"status": 200, "data": {**answer, "proof": proof}}

    def request(self, operation: str, **arguments) -> dict:
        """Call the instance's runtime over its connection; returns its `{status, data}` answer."""
        call = self.server.call(
            "app:request",
            {"operation": operation, "arguments": arguments},
            to=self.sid,
            namespace="/apps",
            timeout=10,
        )
        return asyncio.run_coroutine_threadsafe(call, self.loop).result(timeout=15)

    def wait_until_ready(self, timeout: float = 20.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.ready:
            time.sleep(0.1)
        assert self.ready, "the instance never completed its connection to the raw app"


@contextlib.contextmanager
def serving_raw_app(private_pem: str | None = None) -> Iterator[RawApp]:
    import socketio
    from aiohttp import web

    raw = RawApp(private_pem or private_key_pem())
    started = threading.Event()
    stopped: list[asyncio.Future] = []

    async def serve() -> None:
        raw.loop = asyncio.get_running_loop()
        raw.server = socketio.AsyncServer(async_mode="aiohttp", transports=["websocket"])

        async def connect(sid, environ, auth):
            return environ.get("HTTP_AUTHORIZATION") == "Bearer " + raw.api_key

        async def enroll(sid, data):
            raw.connects.append(data)
            return raw.answer_connect(data)

        async def ready(sid, data):
            for operation in raw.before_ready:
                request = {"operation": operation, "arguments": {}}
                answer = await raw.server.call("app:request", request, to=sid, namespace="/apps")
                raw.before_ready_answers.append(answer)
            ping = {"operation": "ping", "arguments": {}}
            pong = await raw.server.call("app:request", ping, to=sid, namespace="/apps")
            if pong == {"status": 200, "data": {"received": True}}:
                raw.sid, raw.ready = sid, True
            return {"status": 200, "data": {"received": True}}

        async def event(sid, payload):
            raw.events.append(payload)
            if str(payload.get("type", "")).startswith("channel.message."):
                return {"status": 200, "data": {"received": True}}
            if raw.answer_with:
                return raw.answer_with(payload)
            if payload.get("type") == "channel.mention":
                return raw.mention_answer
            return {"status": 200, "data": raw.event_answer}

        for name, handler in (
            ("connect", connect),
            ("app:connect", enroll),
            ("app:ready", ready),
            ("app:event", event),
        ):
            raw.server.on(name, handler, namespace="/apps")
        application = web.Application()
        raw.server.attach(application, socketio_path="events/socket.io")
        runner = web.AppRunner(application)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", raw.port).start()
        stopped.append(raw.loop.create_future())
        started.set()
        await stopped[0]
        # the instance keeps its socket open; drop it so the runner stops at once
        for sid in list(raw.server.manager.get_participants("/apps", None)):
            await raw.server.disconnect(sid[0], namespace="/apps")
        await runner.shutdown()
        await runner.cleanup()

    thread = threading.Thread(target=asyncio.run, args=(serve(),), daemon=True)
    thread.start()
    assert started.wait(10), "the raw app did not start"
    try:
        yield raw
    finally:
        raw.loop.call_soon_threadsafe(stopped[0].set_result, None)
        thread.join(10)


def install_raw_app(client: httpx.Client, raw: RawApp, scopes: list[str] | None = None) -> dict:
    """The raw app installed with every scope it requests; not yet connected."""
    manifest = raw.manifest(scopes)
    created = client.post(
        "/api/v1/apps",
        json={
            "manifest": manifest,
            "url": raw.url,
            "api_key": raw.api_key,
            "slug": f"raw-{uuid.uuid4().hex[:8]}",
            "granted_scopes": manifest["requested_scopes"],
        },
    )
    assert created.status_code == 200, created.text
    return created.json()
