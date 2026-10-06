"""A ComfyUI server played by a local aiohttp app, and the image settings that point at it.

`FakeComfyUI(save_node_class)` finishes every queued prompt at once: it announces the end of the
run on the `/ws` websocket the client waits on and reports one image from the workflow's output
node, whose `class_type` is `save_node_class`, and `/object_info` lists `CHECKPOINTS`. It keeps
every queued prompt in `queued` and every uploaded input image in `uploads`.
`serving(fake)` runs it and yields its base URL, also kept as `fake.base_url`;
`comfyui_settings(base_url, workflow)` is the admin's image configuration for generation and
editing through it. Wrap a change on the shared instance in `preserve(IMAGES_CONFIG)`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from typing import Iterator

from aiohttp import web

from harness.instance import free_port

PROMPT_ID = "prompt-1"
CHECKPOINTS = ["dream.safetensors", "sketch.safetensors"]
SAVE_NODE = "9"
PROMPT_NODE = {"type": "prompt", "node_ids": ["6"], "key": "text"}
IMAGE_NODE = {"type": "image", "node_ids": ["10"], "key": "image"}
# a 1x1 transparent PNG, the image every run produces
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d4944415478da63f8ffff3f0005fe02fea7d6a4a50000000049454e44ae426082"
)


def workflow(save_node_class: str = "SaveImage") -> dict:
    """A prompt, an input image and an output node of `save_node_class`."""
    return {
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "10": {"class_type": "LoadImage", "inputs": {"image": ""}},
        SAVE_NODE: {"class_type": save_node_class, "inputs": {}},
    }


class FakeComfyUI:
    """Queues every prompt as done at once and serves one generated image."""

    def __init__(self) -> None:
        self.queued: list[dict] = []
        self.uploads: list[bytes] = []
        self.base_url = ""
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

    async def object_info(self, request: web.Request) -> web.Response:
        loader = {"input": {"required": {"ckpt_name": [CHECKPOINTS]}}}
        return web.json_response({"CheckpointLoaderSimple": loader})

    async def upload(self, request: web.Request) -> web.Response:
        form = await request.post()
        self.uploads.append(form["image"].file.read())
        return web.json_response({"name": "input.png"})


@contextlib.contextmanager
def serving(fake: FakeComfyUI) -> Iterator[str]:
    app = web.Application()
    app.router.add_get("/ws", fake.websocket)
    app.router.add_post("/prompt", fake.queue_prompt)
    app.router.add_get(f"/history/{PROMPT_ID}", fake.history)
    app.router.add_get("/view", fake.view)
    app.router.add_get("/object_info", fake.object_info)
    app.router.add_post("/api/upload/image", fake.upload)

    # the loop runs on a thread of its own, so a caller that runs a loop (Playwright) can use it
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    runner = web.AppRunner(app)
    asyncio.run_coroutine_threadsafe(runner.setup(), loop).result(timeout=10)
    port = free_port()
    site = web.TCPSite(runner, "127.0.0.1", port)
    asyncio.run_coroutine_threadsafe(site.start(), loop).result(timeout=10)
    fake.base_url = f"http://127.0.0.1:{port}"
    try:
        yield fake.base_url
    finally:
        asyncio.run_coroutine_threadsafe(runner.cleanup(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)
        loop.close()


def comfyui_settings(base_url: str, graph: dict) -> dict:
    """Image settings that generate and edit through ComfyUI at `base_url` running `graph`."""
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "IMAGE_GENERATION_ENGINE": "comfyui",
        "COMFYUI_BASE_URL": base_url,
        "COMFYUI_WORKFLOW": json.dumps(graph),
        "COMFYUI_WORKFLOW_NODES": [PROMPT_NODE],
        "ENABLE_IMAGE_EDIT": True,
        "IMAGE_EDIT_ENGINE": "comfyui",
        "IMAGES_EDIT_COMFYUI_BASE_URL": base_url,
        "IMAGES_EDIT_COMFYUI_WORKFLOW": json.dumps(graph),
        "IMAGES_EDIT_COMFYUI_WORKFLOW_NODES": [PROMPT_NODE, IMAGE_NODE],
    }
