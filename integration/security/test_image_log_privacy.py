"""Regression: an image prompt sent to ComfyUI stays out of the server log.

open-webui 0.10.2 fix `64b92ff08` (#26400). `comfyui_create_image` and `comfyui_edit_image`
logged the whole workflow at INFO, the default level, and the workflow carries the user's prompt,
so every prompt landed in operator-visible logs. The fix logs it at DEBUG.

Twin of unit/security/test_image_log_privacy.py, whose source audit only caught the f-string
form; this reads the log a running instance writes, so any INFO line carrying the prompt fails.
ComfyUI is a local stand-in: the HTTP routes and the `/ws` websocket the client waits on.

Discriminates: passes on dev `bbfa876af`; with either workflow log line back at INFO (in the
`%s` form dev now uses) the prompt is in the log and the matching test fails.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import threading
import uuid
from typing import Iterator

import pytest

from harness.instance import free_port

web = pytest.importorskip("aiohttp.web")

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

IMAGES_CONFIG = ("/api/v1/images/config", "/api/v1/images/config/update")
PROMPT_ID = "prompt-1"
SAVE_NODE = "9"
WORKFLOW = {
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
    SAVE_NODE: {"class_type": "SaveImage", "inputs": {}},
}
PROMPT_NODE = {"type": "prompt", "node_ids": ["6"], "key": "text"}
IMAGE_NODE = {"type": "image", "node_ids": ["10"], "key": "image"}
# A 1x1 transparent PNG.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63f8ffff3f0005fe02fea7d6a4a50000000049454e44ae426082"
)


class FakeComfyUI:
    """Queues every prompt as done at once and serves one generated image."""

    def __init__(self) -> None:
        self.queued: list[dict] = []
        self.sockets: dict[str, web.WebSocketResponse] = {}

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        self.sockets[request.query["clientId"]] = socket
        async for _ in socket:
            pass
        return socket

    async def queue_prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.queued.append(body)
        done = {"type": "executing", "data": {"node": None, "prompt_id": PROMPT_ID}}
        await self.sockets[body["client_id"]].send_str(json.dumps(done))
        return web.json_response({"prompt_id": PROMPT_ID})

    async def history(self, request: web.Request) -> web.Response:
        image = {"filename": "out.png", "subfolder": "", "type": "output"}
        return web.json_response({PROMPT_ID: {"outputs": {SAVE_NODE: {"images": [image]}}}})

    async def view(self, request: web.Request) -> web.Response:
        return web.Response(body=PNG, content_type="image/png")

    async def upload(self, request: web.Request) -> web.Response:
        return web.json_response({"name": "input.png"})


@contextlib.contextmanager
def _serving(fake: FakeComfyUI) -> Iterator[str]:
    app = web.Application()
    app.router.add_get("/ws", fake.websocket)
    app.router.add_post("/prompt", fake.queue_prompt)
    app.router.add_get(f"/history/{PROMPT_ID}", fake.history)
    app.router.add_get("/view", fake.view)
    app.router.add_post("/api/upload/image", fake.upload)

    loop = asyncio.new_event_loop()
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    port = free_port()
    loop.run_until_complete(web.TCPSite(runner, "127.0.0.1", port).start())
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


@pytest.fixture
def comfyui(admin, preserve) -> Iterator[FakeComfyUI]:
    """ComfyUI as the image engine for generation and editing, restored afterwards."""
    fake = FakeComfyUI()
    preserve(IMAGES_CONFIG)
    with _serving(fake) as base_url, admin.client() as client:
        current = client.get(IMAGES_CONFIG[0]).json()
        engine = {
            "ENABLE_IMAGE_GENERATION": True,
            "IMAGE_GENERATION_ENGINE": "comfyui",
            "COMFYUI_BASE_URL": base_url,
            "COMFYUI_WORKFLOW": json.dumps(WORKFLOW),
            "COMFYUI_WORKFLOW_NODES": [PROMPT_NODE],
            "ENABLE_IMAGE_EDIT": True,
            "IMAGE_EDIT_ENGINE": "comfyui",
            "IMAGES_EDIT_COMFYUI_BASE_URL": base_url,
            "IMAGES_EDIT_COMFYUI_WORKFLOW": json.dumps(WORKFLOW),
            "IMAGES_EDIT_COMFYUI_WORKFLOW_NODES": [PROMPT_NODE, IMAGE_NODE],
        }
        updated = client.post(IMAGES_CONFIG[1], json={**current, **engine})
        assert updated.status_code == 200, updated.text
        yield fake


def _png_data_url() -> str:
    return f"data:image/png;base64,{base64.b64encode(PNG).decode()}"


REQUESTS = {
    "generate": lambda prompt: ("/api/v1/images/generations", {"prompt": prompt}),
    "edit": lambda prompt: ("/api/v1/images/edit", {"prompt": prompt, "image": _png_data_url()}),
}


@pytest.mark.parametrize("action", REQUESTS)
def test_the_prompt_sent_to_comfyui_stays_out_of_the_log(instance, admin, comfyui, action):
    prompt = f"a lighthouse at dusk {uuid.uuid4().hex}"
    route, payload = REQUESTS[action](prompt)
    offset = instance.log_size()

    with admin.client() as client:
        response = client.post(route, json=payload)

    assert response.status_code == 200, response.text
    assert prompt in json.dumps(comfyui.queued), "the workflow ComfyUI got lacks the prompt"
    logged = instance.log_since(offset)
    assert logged, "nothing was logged at all, so the log level hides what this checks"
    assert prompt not in logged, (
        f"the {action} prompt was written to the server log at the default level, so every "
        "image prompt is readable by whoever reads the logs (#26400)"
    )
