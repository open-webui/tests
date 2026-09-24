"""Talking to a model in a channel, and an image engine for it to draw with.

`enable_channels` turns channels on (wrap the test in `preserve("admin_config")`), `mention`
posts a message that mentions a model into a new channel the way the channel input does, and
`model_reply` waits for the model's finished answer. `serve_openai_images` answers as OpenAI's
image endpoints and returns the image settings that use them (see `harness.image_engines`), and
`drawing_model` is a model on the scripted provider that generates images by default.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from typing import Iterator

import httpx

from harness.image_engines import PNG_BASE64
from harness.listener import Listener, json_answer
from harness.upstream import MOCK_MODEL_ID

ADMIN_CONFIG = "/api/v1/auths/admin/config"


def enable_channels(client: httpx.Client, reply_mode: str = "thread") -> None:
    current = client.get(ADMIN_CONFIG)
    current.raise_for_status()
    enabled = {**current.json(), "ENABLE_CHANNELS": True}
    saved = client.post(ADMIN_CONFIG, json={**enabled, "CHANNEL_MODEL_RESPONSE_MODE": reply_mode})
    assert saved.status_code == 200, f"enabling channels failed: {saved.text}"


def mention(client: httpx.Client, model_id: str, text: str) -> tuple[str, str]:
    """Post `text` mentioning `model_id` into a new channel; returns the channel and message ids."""
    channel = client.post("/api/v1/channels/create", json={"name": f"c-{uuid.uuid4().hex[:8]}"})
    assert channel.status_code == 200, f"creating a channel failed: {channel.text}"
    channel_id = channel.json()["id"]
    posted = client.post(
        f"/api/v1/channels/{channel_id}/messages/post",
        json={"content": f"<@M:{model_id}|{model_id}> {text}"},
    )
    assert posted.status_code == 200, f"posting to the channel failed: {posted.text}"
    return channel_id, posted.json()["id"]


def model_reply(client: httpx.Client, channel_id: str, timeout: float = 30.0) -> dict:
    """The model's finished answer anywhere in the channel, thread replies included."""
    deadline = time.monotonic() + timeout
    answers: list[dict] = []
    while time.monotonic() < deadline:
        listed = client.get(f"/api/v1/channels/{channel_id}/messages").json()
        threads = [
            reply
            for message in listed
            for reply in client.get(
                f"/api/v1/channels/{channel_id}/messages/{message['id']}/thread"
            ).json()
        ]
        answers = [
            message
            for message in [*listed, *threads]
            if (message.get("meta") or {}).get("model_id")
        ]
        if answers and answers[0]["meta"].get("done"):
            return answers[0]
        time.sleep(0.2)
    raise AssertionError(f"the model never finished answering in the channel: {answers}")


def stored_files(client: httpx.Client, channel_id: str, message_id: str) -> list[dict]:
    stored = client.get(f"/api/v1/channels/{channel_id}/messages/{message_id}/data")
    stored.raise_for_status()
    return (stored.json() or {}).get("files") or []


def serve_openai_images(listener: Listener) -> dict:
    """Answer as OpenAI's image generation and edit endpoints; returns the settings using them."""
    drawn = json_answer({"data": [{"b64_json": PNG_BASE64}]})
    listener.route("POST", "/images/generations", drawn)
    listener.route("POST", "/images/edits", drawn)
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "ENABLE_IMAGE_PROMPT_GENERATION": False,
        "IMAGE_GENERATION_ENGINE": "openai",
        "IMAGE_GENERATION_MODEL": "dall-e-2",
        "IMAGE_SIZE": "512x512",
        "IMAGE_STEPS": 0,
        "IMAGES_OPENAI_API_BASE_URL": listener.base_url,
        "IMAGES_OPENAI_API_KEY": "sk-images",
        "ENABLE_IMAGE_EDIT": True,
        "IMAGE_EDIT_ENGINE": "openai",
        "IMAGE_EDIT_MODEL": "dall-e-2",
        "IMAGES_EDIT_OPENAI_API_BASE_URL": listener.base_url,
        "IMAGES_EDIT_OPENAI_API_KEY": "sk-images",
    }


@contextmanager
def drawing_model(client: httpx.Client, function_calling: str = "native") -> Iterator[str]:
    """A workspace model with image generation on by default; yields its id, deletes it after."""
    model_id = f"painter-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {
            "capabilities": {"image_generation": True},
            "defaultFeatureIds": ["image_generation"],
        },
        "params": {"function_calling": function_calling},
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, f"creating {model_id} failed: {created.text}"
    try:
        client.get("/api/models").raise_for_status()  # registers the new model
        yield model_id
    finally:
        client.post("/api/v1/models/model/delete", json={"id": model_id})
