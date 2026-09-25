"""Journey: feature switches and user permissions hold on the direct routes, not only in the UI.

The admin panel hides a feature that is switched off or that a user lacks the permission for,
but each feature's own routes check again: image generation and editing, speech and
transcription, channels, notes and folders. A switch that is off refuses everyone, admins
included; a missing permission refuses the user while an admin carries on. Each positive path
runs against a local fake, so an allowed call really completes, and each refusal is read on the
fake as well: a refused call never reaches the engine.

Discriminates: in a backend copy, dropping the `features.image_generation` check from
`generate_images`, the `chat.tts` check from `speech`, the `features.channels` branch from
`check_channels_access`, the admin-only standard channel check from `create_new_channel`, the
`features.notes` check from `create_new_note` and the `folders.enable` check from
`check_folders_permission` each turn their tests red (seven of 23); the positive paths stay green.
"""

from __future__ import annotations

import io
import uuid
import wave

import pytest

from harness.audio_engine import serve_audio_engine, using_audio_engine
from harness.channel_chat import serve_openai_images
from harness.channel_quotes import group_channel
from harness.image_engines import IMAGES_CONFIG, PNG_BASE64, save_image_settings

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
ADMIN_CONFIG = "/api/v1/auths/admin/config"
# channels and notes answer a missing permission with 401, the others with 403
REFUSED = {401, 403}


def _save(admin, path: str, **changes) -> None:
    """Save `changes` over the admin's settings at `path`, the rest as they are."""
    with admin.client() as client:
        current = client.get(path)
        current.raise_for_status()
        saved = client.post(path, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def switches(admin, preserve):
    """`switches(**admin_config)` changes the admin's feature switches for this test."""
    preserve("admin_config")
    return lambda **changes: _save(admin, ADMIN_CONFIG, **changes)


@pytest.fixture
def deny(admin, preserve):
    """`deny(section, flag)` takes one permission away from every user for this test."""
    preserve("permissions")

    def take_away(section: str, flag: str) -> None:
        with admin.client() as client:
            permissions = client.get(PERMISSIONS).json()
        _save(admin, PERMISSIONS, **{section: {**permissions[section], flag: False}})

    return take_away


# images


IMAGE_ROUTES = {
    "generations": (
        "/api/v1/images/generations",
        {"prompt": "a lighthouse"},
        "/images/generations",
    ),
    "edit": (
        "/api/v1/images/edit",
        {"image": f"data:image/png;base64,{PNG_BASE64}", "prompt": "make it blue"},
        "/images/edits",
    ),
}
IMAGE_SWITCHES = {"generations": "ENABLE_IMAGE_GENERATION", "edit": "ENABLE_IMAGE_EDIT"}


@pytest.fixture
def images(admin, preserve, listener):
    """`images(**overrides)` points the image settings at a local OpenAI-shaped engine."""
    preserve(IMAGES_CONFIG)
    settings = serve_openai_images(listener)

    def configure(**overrides):
        with admin.client() as client:
            save_image_settings(client, **{**settings, **overrides})

    return configure


def _draw(actor, route: str):
    path, body, _ = IMAGE_ROUTES[route]
    with actor.client() as client:
        return client.post(path, json=body)


@pytest.mark.parametrize("route", sorted(IMAGE_ROUTES))
@pytest.mark.parametrize("role", ["user", "admin"])
def test_a_switched_off_image_route_refuses_everyone(images, listener, make_user, route, role):
    images(**{IMAGE_SWITCHES[route]: False})

    refused = _draw(make_user(role=role), route)

    assert refused.status_code == 403, refused.text
    assert listener.requests_to(IMAGE_ROUTES[route][2]) == []


@pytest.mark.parametrize("route", sorted(IMAGE_ROUTES))
def test_a_user_without_image_generation_is_refused_and_an_admin_is_not(
    images, deny, listener, make_user, route
):
    images()
    deny("features", "image_generation")

    refused = _draw(make_user(), route)
    assert refused.status_code == 403, refused.text
    assert listener.requests_to(IMAGE_ROUTES[route][2]) == []

    drawn = _draw(make_user(role="admin"), route)
    assert drawn.status_code == 200, drawn.text
    assert len(listener.requests_to(IMAGE_ROUTES[route][2])) == 1


@pytest.mark.parametrize("route", sorted(IMAGE_ROUTES))
def test_a_permitted_user_draws(images, listener, make_user, route):
    images()

    drawn = _draw(make_user(), route)

    assert drawn.status_code == 200, drawn.text
    assert len(listener.requests_to(IMAGE_ROUTES[route][2])) == 1


# speech and transcription


def _recording() -> bytes:
    """A tenth of a second of silence, as a WAV file."""
    recording = io.BytesIO()
    with wave.open(recording, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(b"\x00\x00" * 1600)
    return recording.getvalue()


def _use_audio(actor, direction: str):
    with actor.client() as client:
        if direction == "tts":
            # the text is unique, so the engine is asked instead of the speech cache
            return client.post("/api/v1/audio/speech", json={"input": uuid.uuid4().hex})
        return client.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("recording.wav", _recording(), "audio/wav")},
        )


AUDIO_PATHS = {"tts": "/audio/speech", "stt": "/audio/transcriptions"}


@pytest.fixture
def audio(admin, listener):
    engine = serve_audio_engine(listener)
    with admin.client() as client, using_audio_engine(client, engine):
        yield engine


@pytest.mark.parametrize("direction", ["tts", "stt"])
def test_a_user_without_the_audio_permission_is_refused_and_an_admin_is_not(
    audio, deny, listener, make_user, direction
):
    deny("chat", direction)

    refused = _use_audio(make_user(), direction)
    assert refused.status_code == 403, refused.text
    assert listener.requests_to(AUDIO_PATHS[direction]) == []

    served = _use_audio(make_user(role="admin"), direction)
    assert served.status_code == 200, served.text
    assert len(listener.requests_to(AUDIO_PATHS[direction])) == 1


@pytest.mark.parametrize("direction", ["tts", "stt"])
def test_a_permitted_user_uses_audio(audio, listener, make_user, direction):
    served = _use_audio(make_user(), direction)

    assert served.status_code == 200, served.text
    assert len(listener.requests_to(AUDIO_PATHS[direction])) == 1


# channels


def _channel_calls(actor, channel_id: str) -> dict:
    """List, create and post as `actor`, by name; returns each status code."""
    with actor.client() as client:
        return {
            "list": client.get("/api/v1/channels/").status_code,
            "create": client.post(
                "/api/v1/channels/create", json={"name": "another", "type": "group"}
            ).status_code,
            "post": client.post(
                f"/api/v1/channels/{channel_id}/messages/post", json={"content": "hello"}
            ).status_code,
        }


@pytest.mark.parametrize("role", ["user", "admin"])
def test_switched_off_channels_refuse_everyone(switches, make_user, role):
    switches(ENABLE_CHANNELS=True)
    member = make_user(role=role)
    channel_id = group_channel(member)
    switches(ENABLE_CHANNELS=False)

    calls = _channel_calls(member, channel_id)

    assert calls == {"list": 403, "create": 403, "post": 403}, calls


def test_a_user_without_channels_is_refused_and_an_admin_is_not(switches, deny, make_user):
    switches(ENABLE_CHANNELS=True)
    member, manager = make_user(), make_user(role="admin")
    member_channel, admin_channel = group_channel(member), group_channel(manager)
    deny("features", "channels")

    refused = _channel_calls(member, member_channel)
    allowed = _channel_calls(manager, admin_channel)

    assert set(refused.values()) <= REFUSED, refused
    assert allowed == {"list": 200, "create": 200, "post": 200}, allowed


@pytest.mark.parametrize(
    ("channel_type", "role", "allowed"),
    [
        (None, "user", False),
        (None, "admin", True),
        ("group", "user", True),
        ("dm", "user", True),
    ],
)
def test_only_an_admin_creates_a_standard_channel(switches, make_user, channel_type, role, allowed):
    switches(ENABLE_CHANNELS=True)
    form = {"name": f"c-{uuid.uuid4().hex[:8]}", "type": channel_type}
    if channel_type == "dm":
        form["user_ids"] = [make_user().id]

    with make_user(role=role).client() as client:
        created = client.post("/api/v1/channels/create", json=form)

    if allowed:
        assert created.status_code == 200, created.text
        assert created.json()["type"] == channel_type
    else:
        assert created.status_code in REFUSED, created.text


# notes


def _note_calls(actor, note_id: str) -> dict:
    """Every note route a user reaches from the notes page, by name; returns each status code."""
    with actor.client() as client:
        return {
            "list": client.get("/api/v1/notes/").status_code,
            "create": client.post("/api/v1/notes/create", json={"title": "new"}).status_code,
            "read": client.get(f"/api/v1/notes/{note_id}").status_code,
            "update": client.post(
                f"/api/v1/notes/{note_id}/update", json={"title": "renamed"}
            ).status_code,
            "delete": client.delete(f"/api/v1/notes/{note_id}/delete").status_code,
        }


def _own_note(actor) -> str:
    with actor.client() as client:
        created = client.post("/api/v1/notes/create", json={"title": "mine"})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def test_a_user_without_notes_is_refused_and_an_admin_is_not(deny, make_user):
    writer, manager = make_user(), make_user(role="admin")
    writer_note, admin_note = _own_note(writer), _own_note(manager)
    deny("features", "notes")

    refused = _note_calls(writer, writer_note)
    allowed = _note_calls(manager, admin_note)

    assert set(refused.values()) <= REFUSED, refused
    assert set(allowed.values()) == {200}, allowed


# folders


def _folder_calls(actor) -> dict:
    with actor.client() as client:
        return {
            "list": client.get("/api/v1/folders/").status_code,
            "create": client.post(
                "/api/v1/folders/", json={"name": f"f-{uuid.uuid4().hex[:8]}"}
            ).status_code,
        }


@pytest.mark.parametrize("role", ["user", "admin"])
def test_switched_off_folders_refuse_everyone(switches, make_user, role):
    switches(ENABLE_FOLDERS=False)

    calls = _folder_calls(make_user(role=role))

    assert calls == {"list": 403, "create": 403}, calls


def test_a_user_without_folders_is_refused_and_an_admin_is_not(deny, make_user):
    deny("features", "folders")

    assert _folder_calls(make_user()) == {"list": 403, "create": 403}
    assert _folder_calls(make_user(role="admin")) == {"list": 200, "create": 200}
