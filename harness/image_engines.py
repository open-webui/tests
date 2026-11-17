"""Image engines played by a local service, and the admin image settings that point at them.

The admin's image settings are a list of image connections, each an engine at a URL with the models
it offers. `image_connection(...)` is one of them and `image_settings(...)` the settings holding
them, with generation and editing switched on; `openai_settings` are those for OpenAI.
`serve_automatic1111` routes a listener as an Automatic1111 server holding `CHECKPOINT` and returns
the image settings that use it. `serve_gemini` routes one as the Gemini API, answering Imagen's
`:predict` and Gemini's `:generateContent` for generation and edits, and refusing any other API key
the way Google does. `changed_connection` changes one connection of such settings.
`save_image_settings` saves settings over the admin's image configuration. Wrap a change on the
shared instance in `preserve(IMAGES_CONFIG)`.
"""

from __future__ import annotations

import httpx

from harness.listener import Answer, Listener, ReceivedRequest, json_answer

IMAGES_CONFIG = ("/api/v1/images/config", "/api/v1/images/config/update")
CHECKPOINT = "configured.safetensors"
IMAGEN_MODEL = "imagen-4.0-generate-001"
GEMINI_IMAGE_MODEL = "gemini-2.5-flash-image"
GEMINI_API_KEY = "gemini-key-0123456789"
# a connection without grants is the admins' alone
EVERYONE = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
# a 1x1 red PNG, what the stand-in engine generates
PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def image_connection(
    connection_id: str,
    engine: str,
    url: str,
    key: str = "",
    *,
    models: tuple[str, ...] = (),
    generates: bool = True,
    edits: bool = True,
    access_grants: list[dict] = EVERYONE,
    **config,
) -> dict:
    """`engine` at `url` offering `models` (none: the engine's own default) to everyone in
    `access_grants`, for what it `generates` and `edits`."""
    return {
        "id": connection_id,
        "name": connection_id,
        "type": engine,
        "url": url,
        "key": key,
        "config": {
            "model_ids": list(models),
            "capabilities": {"image_generation": generates, "image_edit": edits},
            "access_grants": access_grants,
            **config,
        },
    }


def image_settings(*connections: dict, generation: bool = True, edit: bool = True) -> dict:
    """Image settings running on `connections`, the first model able to do it drawing or editing."""
    return {
        "ENABLE_IMAGE_GENERATION": generation,
        "ENABLE_IMAGE_EDIT": edit,
        "IMAGE_GENERATION_DEFAULT_MODEL": "",
        "IMAGE_EDIT_DEFAULT_MODEL": "",
        "IMAGE_CONNECTIONS": list(connections),
    }


def openai_settings(base_url: str, key: str = "sk-images", model: str = "dall-e-2") -> dict:
    """Image settings that generate and edit with `model` on OpenAI's image API at `base_url`."""
    openai = image_connection("openai", "openai", base_url, key, models=(model,), size="512x512")
    return {**image_settings(openai), "ENABLE_IMAGE_PROMPT_GENERATION": False}


def changed_connection(settings: dict, connection_id: str, **changes) -> dict:
    """`settings` with one connection changed: `url`, `key` and `models` as in `image_connection`,
    anything else in its config."""
    connections = settings["IMAGE_CONNECTIONS"]
    assert connection_id in [connection["id"] for connection in connections], connections
    if "models" in changes:
        changes["model_ids"] = list(changes.pop("models"))
    top = {field: changes.pop(field) for field in ("url", "key") if field in changes}
    changed = [
        {**connection, **top, "config": {**connection["config"], **changes}}
        if connection["id"] == connection_id
        else connection
        for connection in connections
    ]
    return {**settings, "IMAGE_CONNECTIONS": changed}


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
    return automatic1111_settings(listener.base_url)


def automatic1111_settings(base_url: str) -> dict:
    """Image settings that generate on `CHECKPOINT` through Automatic1111 at `base_url`."""
    automatic1111 = image_connection(
        "automatic1111",
        "automatic1111",
        base_url,
        models=(CHECKPOINT,),
        edits=False,
        size="512x512",
        steps=20,
    )
    return image_settings(automatic1111, edit=False)


def checkpoint_switches(listener: Listener) -> list[dict]:
    """Every checkpoint change the Automatic1111 stand-in was sent."""
    options = listener.requests_to("/sdapi/v1/options")
    return [call.json() for call in options if call.method == "POST"]


def serve_gemini(listener: Listener) -> dict:
    """Answer as the Gemini API; returns the settings that generate and edit with it."""

    def answering(payload: dict):
        def answer(request: ReceivedRequest) -> Answer:
            if request.headers.get("x-goog-api-key") != GEMINI_API_KEY:
                return json_answer({"error": {"code": 403, "status": "PERMISSION_DENIED"}}, 403)
            return json_answer(payload)

        return answer

    predicted = {"predictions": [{"bytesBase64Encoded": PNG_BASE64, "mimeType": "image/png"}]}
    image_part = {"inlineData": {"mimeType": "image/png", "data": PNG_BASE64}}
    generated = {"candidates": [{"content": {"parts": [{"text": "Here it is."}, image_part]}}]}
    listener.route("POST", f"/models/{IMAGEN_MODEL}:predict", answering(predicted))
    listener.route("POST", f"/models/{GEMINI_IMAGE_MODEL}:generateContent", answering(generated))
    return gemini_settings(listener.base_url)


def gemini_settings(base_url: str) -> dict:
    """Image settings that generate with Imagen's `:predict` (connection `imagen`) and edit with
    Gemini (connection `gemini`), both at `base_url`."""
    imagen = image_connection(
        "imagen",
        "gemini",
        base_url,
        GEMINI_API_KEY,
        models=(IMAGEN_MODEL,),
        edits=False,
        endpoint_method="predict",
    )
    gemini = image_connection(
        "gemini", "gemini", base_url, GEMINI_API_KEY, models=(GEMINI_IMAGE_MODEL,), generates=False
    )
    return image_settings(imagen, gemini)


def gemini_calls(listener: Listener, model: str, method: str) -> list[ReceivedRequest]:
    """Every request the Gemini stand-in got for `model:method`."""
    return listener.requests_to(f"/models/{model}:{method}")
