"""Image engines played by a local service, and the admin image settings that point at them.

`serve_automatic1111` routes a listener as an Automatic1111 server holding `CHECKPOINT` and
returns the image settings that use it; `save_image_settings` saves settings over the admin's
image configuration. Wrap a change on the shared instance in `preserve(IMAGES_CONFIG)`.
"""

from __future__ import annotations

import httpx

from harness.listener import Listener, json_answer

IMAGES_CONFIG = ("/api/v1/images/config", "/api/v1/images/config/update")
CHECKPOINT = "configured.safetensors"
# a 1x1 red PNG, what the stand-in engine generates
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def save_image_settings(client: httpx.Client, **changes) -> None:
    """Save `changes` over the admin's image settings, the rest as they are."""
    current = client.get(IMAGES_CONFIG[0])
    current.raise_for_status()
    saved = client.post(IMAGES_CONFIG[1], json={**current.json(), **changes})
    assert saved.status_code == 200, f"saving the image settings failed: {saved.text}"


def serve_automatic1111(listener: Listener) -> dict:
    """Answer as an Automatic1111 server on `CHECKPOINT`; returns the settings that use it."""
    listener.route("GET", "/sdapi/v1/options", json_answer({"sd_model_checkpoint": CHECKPOINT}))
    listener.route("POST", "/sdapi/v1/options", json_answer({}))
    listener.route("POST", "/sdapi/v1/txt2img", json_answer({"images": [PNG_BASE64], "info": "{}"}))
    return {
        "ENABLE_IMAGE_GENERATION": True,
        "IMAGE_GENERATION_ENGINE": "automatic1111",
        "AUTOMATIC1111_BASE_URL": listener.base_url,
        "IMAGE_GENERATION_MODEL": CHECKPOINT,
        "IMAGE_SIZE": "512x512",
        "IMAGE_STEPS": 20,
    }


def checkpoint_switches(listener: Listener) -> list[dict]:
    """Every checkpoint change the Automatic1111 stand-in was sent."""
    options = listener.requests_to("/sdapi/v1/options")
    return [call.json() for call in options if call.method == "POST"]
