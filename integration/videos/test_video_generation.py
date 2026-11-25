"""Journey: video generation, from the admin's video settings to the video stored on the reply.

New in v0.13.4. An admin turns on Video Generation, adds video connections (OpenAI compatible
servers such as OpenAI and LiteLLM, or OpenRouter) and may pick a default video model; a user
needs the Video Generation permission, off by default, and a chat with Video switched on. The
model then calls the built-in `generate_video` tool: Open WebUI starts a job at the provider,
waits for it, downloads the video and stores it as a file on the reply. A local stand-in plays the
provider (`harness.video_providers`), so each test reads what the provider was sent: the model,
the prompt, the connection's default size and length, the key, an attached image as the first
frame, and the multipart form OpenAI's API takes even for a text-only prompt (#34679).

The model is chosen in this order: the user's pick in the chat, the chat model's own default
video model, the admin's default, the first model the user can use. A connection shared with
some users offers its models to them alone. A failed job reaches the model as an error and stores
no video.

Discriminates: passes on dev f80a4f782. Each test fails in a backend copy with one edit: the
permission default turned on (start-off test), the connection key dropped on save (settings test),
the permission check left out of `/api/v1/videos/models` and the tool list (permission test), the
model capability check left out of the tool list (tool test), the multipart form sent only with an
image (chat test), the starting image never loaded (image test), `frame_images` never sent to
OpenRouter (OpenRouter test), LiteLLM's video labels ignored (LiteLLM test), the chat's pick and
the model's default ignored (pick order test), connection access skipped (access test) and a
failed job downloaded as if done (failure test).
"""

from __future__ import annotations

import base64
import json
import uuid

import httpx
import pytest

from harness import upstream as reply
from harness.access import grant, make_group
from harness.chat import ask
from harness.image_engines import PNG_BASE64
from harness.listener import json_answer
from harness.upstream import MOCK_MODEL_ID
from harness.video_providers import (
    FAILING_PROMPT,
    FAILURE_MESSAGE,
    VIDEO,
    VIDEO_CONTENT_TYPE,
    VIDEOS_CONFIG,
    created_jobs,
    form_fields,
    job_polls,
    model_ref,
    save_video_settings,
    serve_video_api,
    video_connection,
    video_settings,
)

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

PERMISSIONS = "/api/v1/users/default/permissions"
VIDEO_PERMISSION = {"features": {"video_generation": True}}
VIDEO_ON = {"video_generation": True}
PICTURE = base64.b64decode(PNG_BASE64)


def tag() -> str:
    return uuid.uuid4().hex[:8]


@pytest.fixture
def videos(admin, preserve, listener):
    """`videos(*connections, **settings)` saves video settings pointing at the stand-in provider."""
    preserve(VIDEOS_CONFIG)
    serve_video_api(listener)

    def configure(*connections: dict, **settings) -> dict:
        with admin.client() as client:
            return save_video_settings(client, video_settings(*connections, **settings))

    return configure


@pytest.fixture
def filmmaker(admin, make_user):
    """A user granted the Video Generation permission through a group of their own."""
    person = make_user()
    make_group(admin, [person], VIDEO_PERMISSION)
    return person


def connection_at(listener, *models: str, **config) -> dict:
    """A public OpenAI compatible connection to the stand-in, under a fresh id."""
    return video_connection(
        f"videos-{tag()}", listener.base_url, models=models, size="1280x720", seconds=8, **config
    )


def make_video(
    client: httpx.Client, upstream, prompt: str, chat_options: dict | None = None, **arguments
) -> tuple[dict, str]:
    """Ask for a video in a chat with Video on, the model calling `generate_video` with `prompt`
    and `arguments`; returns the stored reply and the tool result."""
    answer = f"Here is your video {tag()}."
    request = f"please film {prompt}"
    upstream.queue(
        reply.tool_call(
            "generate_video", {"prompt": prompt, **arguments}, match=reply.answering(request)
        ),
        reply.text(answer, match=reply.answering(request)),
    )
    options = {"features": VIDEO_ON} | (chat_options or {})
    _, stored = ask(client, request, timeout=60, **options)
    follow_ups = [body for body in upstream.chat_requests() if reply.answering(request)(body)]
    results = [
        entry["content"]
        for body in follow_ups
        for entry in reply.conversation(body)
        if entry["role"] == "tool"
    ]
    assert results, f"generate_video never ran; the reply was {stored.get('content')!r}"
    return stored, results[-1]


def videos_on(message: dict) -> list[dict]:
    return [file for file in message.get("files") or [] if file.get("type") == "video"]


def job_for(listener, prompt: str):
    jobs = [job for job in created_jobs(listener) if prompt.encode() in job.body]
    assert len(jobs) == 1, f"the provider got {len(jobs)} jobs for {prompt!r}"
    return jobs[0]


def offered_tools(client: httpx.Client, upstream, **options) -> set[str]:
    prompt = f"which tools are there? {tag()}"
    upstream.queue(reply.text("none", match=reply.answering(prompt)))
    ask(client, prompt, **options)
    sent = [body for body in upstream.chat_requests() if reply.answering(prompt)(body)]
    assert sent, f"the provider never received {prompt!r}"
    return {tool["function"]["name"] for tool in sent[0].get("tools") or []}


def upload_picture(client: httpx.Client) -> str:
    uploaded = client.post("/api/v1/files/", files={"file": ("first.png", PICTURE, "image/png")})
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


def test_video_generation_and_its_permission_start_off(admin, user):
    with admin.client() as client:
        settings = client.get(VIDEOS_CONFIG[0])
        defaults = client.get(PERMISSIONS)
    with user.client() as client:
        features = client.get("/api/config").json()["features"]

    assert settings.status_code == 200, settings.text
    assert settings.json()["ENABLE_VIDEO_GENERATION"] is False, settings.json()
    assert features.get("enable_video_generation") is False, features
    assert defaults.json()["features"].get("video_generation") is False, (
        f"the Video Generation permission is not off by default: {defaults.json()['features']}"
    )


def test_an_admin_saves_video_connections_and_reads_them_back(videos, listener, admin, user):
    connection = connection_at(
        listener, "sora-2", "sora-2-pro", key="sk-films", params={"quality": "high"}
    )
    saved = videos(connection, default_model=model_ref(connection["id"], "sora-2-pro"))

    with admin.client() as client:
        read_back = client.get(VIDEOS_CONFIG[0]).json()
        verified = client.post("/api/v1/videos/connections/verify", json=connection)
    [stored] = read_back["VIDEO_CONNECTIONS"]
    assert read_back == saved
    assert read_back["ENABLE_VIDEO_GENERATION"] is True
    assert read_back["VIDEO_GENERATION_DEFAULT_MODEL"] == f"{connection['id']}:sora-2-pro"
    assert (stored["url"], stored["key"]) == (listener.base_url, "sk-films"), stored
    config = stored["config"]
    assert config["model_ids"] == ["sora-2", "sora-2-pro"], config
    assert (config["size"], config["seconds"], config["params"]) == (
        "1280x720",
        8,
        {"quality": "high"},
    ), config
    assert verified.status_code == 200, f"Verify Connection failed: {verified.text}"
    [models_call] = listener.requests_to("/models")
    assert models_call.headers.get("Authorization") == "Bearer sk-films"

    with user.client() as client:
        assert client.get("/api/config").json()["features"]["enable_video_generation"] is True
        assert client.get(VIDEOS_CONFIG[0]).status_code in (401, 403)
        refused = client.post(VIDEOS_CONFIG[1], json=video_settings(enable=False))
    assert refused.status_code in (401, 403), "a user could change the video settings"


def test_a_user_needs_the_video_generation_permission(
    videos, listener, upstream, make_user, filmmaker
):
    connection = connection_at(listener, "sora-2")
    videos(connection)
    without = make_user()

    with without.client() as client:
        listed = client.get("/api/v1/videos/models").json()
        refused = client.post("/api/v1/videos/generations", json={"prompt": "a quiet harbour"})
        offered_without = offered_tools(client, upstream, features=VIDEO_ON)
    assert listed == [], f"a user without the permission is offered video models: {listed}"
    assert refused.status_code == 403, refused.text
    assert "generate_video" not in offered_without
    assert created_jobs(listener) == [], "a refused user still started a video job"

    for permitted in (filmmaker, make_user(role="admin")):
        with permitted.client() as client:
            listed = client.get("/api/v1/videos/models").json()
            offered = offered_tools(client, upstream, features=VIDEO_ON)
        assert [model["id"] for model in listed] == [f"{connection['id']}:sora-2"], listed
        assert "generate_video" in offered, f"{permitted.role} is not offered generate_video"


def test_the_video_tool_needs_the_switch_the_chat_toggle_and_the_model(
    videos, listener, upstream, admin, filmmaker
):
    connection = connection_at(listener, "sora-2")
    videos(connection, enable=False)
    with filmmaker.client() as client:
        assert "generate_video" not in offered_tools(client, upstream, features=VIDEO_ON), (
            "generate_video is offered with Video Generation switched off"
        )
    videos(connection)
    with filmmaker.client() as client:
        assert "generate_video" in offered_tools(client, upstream, features=VIDEO_ON)
        assert "generate_video" not in offered_tools(client, upstream, features={}), (
            "generate_video is offered without Video switched on in the chat"
        )

    for meta in (
        {"capabilities": {"video_generation": False}},
        {"builtinTools": {"video_generation": False}},
    ):
        with admin.client() as client:
            model_id = workspace_model(client, meta)
        with filmmaker.client() as client:
            offered = offered_tools(client, upstream, model=model_id, features=VIDEO_ON)
        assert "generate_video" not in offered, f"generate_video is offered to a model with {meta}"


def workspace_model(client: httpx.Client, meta: dict) -> str:
    """A public workspace model on the mock model with `meta`; returns its id."""
    model_id = f"director-{tag()}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": meta,
        "params": {"function_calling": "native"},
        "access_grants": [grant("user", "*", "read")],
    }
    created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, f"creating {model_id} failed: {created.text}"
    client.get("/api/models").raise_for_status()
    return model_id


def test_a_video_made_in_the_chat_is_stored_on_the_reply(videos, listener, upstream, filmmaker):
    connection = connection_at(listener, "sora-2", key="sk-films", params={"quality": "high"})
    videos(connection)
    prompt = f"a lighthouse at dusk {tag()}"

    with filmmaker.client() as client:
        stored, result = make_video(client, upstream, prompt)

        job = job_for(listener, prompt)
        fields = form_fields(job)
        sent = {name: value.decode() for name, (value, _) in fields.items()}
        assert job.headers.get("Authorization") == "Bearer sk-films"
        assert sent == {
            "model": "sora-2",
            "prompt": prompt,
            "seconds": "8",
            "size": "1280x720",
            "quality": "high",
        }, f"the provider was sent {sent}"
        assert job_polls(listener), "Open WebUI never checked on the job or fetched the video"

        [video] = videos_on(stored)
        downloaded = client.get(video["url"])
        assert downloaded.status_code == 200, downloaded.text
        assert downloaded.content == VIDEO, "the stored video is not the one the provider made"
        assert downloaded.headers["content-type"].startswith(VIDEO_CONTENT_TYPE)
    told = json.loads(result)
    assert told.get("status") == "success", f"the model was told {result}"
    assert video["url"] in result, "the model was not told where the video is"


def test_a_video_starts_from_an_image_in_the_chat(videos, listener, upstream, filmmaker):
    videos(connection_at(listener, "sora-2"))
    prompt = f"the boat sails off {tag()}"

    with filmmaker.client() as client:
        file_id = upload_picture(client)
        stored, _ = make_video(
            client, upstream, prompt, image_url=f"/api/v1/files/{file_id}/content"
        )

    fields = form_fields(job_for(listener, prompt))
    assert "input_reference" in fields, f"no first frame was sent: {sorted(fields)}"
    frame, content_type = fields["input_reference"]
    assert (frame, content_type) == (PICTURE, "image/png"), "the first frame is not the image"
    assert len(videos_on(stored)) == 1, stored.get("files")


def test_openrouter_lists_its_video_models_and_takes_json(videos, listener, upstream, admin):
    serve_video_api(listener, openrouter_models=("google/veo-3.1", "kwaivgi/kling-v2.1"))
    connection = connection_at(listener, provider="openrouter")
    with admin.client() as client:
        verified = client.post("/api/v1/videos/connections/verify", json=connection)
    assert verified.status_code == 200, verified.text
    assert [model["id"] for model in verified.json()] == ["google/veo-3.1", "kwaivgi/kling-v2.1"]

    videos(connection)
    person = admin
    prompt = f"a kestrel hovering {tag()}"
    with person.client() as client:
        listed = [model["model"] for model in client.get("/api/v1/videos/models").json()]
        assert listed == ["google/veo-3.1", "kwaivgi/kling-v2.1"], listed
        file_id = upload_picture(client)
        stored, _ = make_video(
            client, upstream, prompt, image_url=f"/api/v1/files/{file_id}/content"
        )
        [video] = videos_on(stored)
        assert client.get(video["url"]).content == VIDEO

    sent = job_for(listener, prompt).json()
    assert (sent["model"], sent["prompt"], sent["duration"], sent["size"]) == (
        "google/veo-3.1",
        prompt,
        8,
        "1280x720",
    ), sent
    [frame] = sent.get("frame_images") or [None]
    assert frame and frame["frame_type"] == "first_frame", f"no first frame was sent: {sent}"
    assert frame["image_url"]["url"] == f"data:image/png;base64,{PNG_BASE64}", frame


def test_a_litellm_connection_offers_only_its_video_models(videos, listener, filmmaker):
    listener.route(
        "GET",
        "/models",
        json_answer({"data": [{"id": model} for model in ("gpt-4o", "sora-2", "veo-3")]}),
    )
    listener.route(
        "GET",
        "/model/info",
        json_answer(
            {
                "data": [
                    {"model_name": "gpt-4o", "model_info": {"mode": "chat"}},
                    {"model_name": "sora-2", "model_info": {"mode": "video_generation"}},
                    {
                        "model_name": "veo-3",
                        "model_info": {"mode": "chat", "supported_endpoints": ["/v1/videos"]},
                    },
                ]
            }
        ),
    )
    connection = connection_at(listener)
    videos(connection)

    with filmmaker.client() as client:
        listed = client.get("/api/v1/videos/models").json()

    assert sorted(model["model"] for model in listed) == ["sora-2", "veo-3"], (
        f"a LiteLLM connection with no model ids offered {[model['model'] for model in listed]}"
    )


def test_the_chat_pick_beats_the_model_default_which_beats_the_admin_default(
    videos, listener, upstream, admin, filmmaker
):
    connection = connection_at(listener, "first-model", "admin-default", "model-default", "picked")
    refs = {
        model: model_ref(connection["id"], model) for model in connection["config"]["model_ids"]
    }
    videos(connection, default_model=refs["admin-default"])
    with admin.client() as client:
        director = workspace_model(client, {"videoModelId": refs["model-default"]})

    with filmmaker.client() as client:
        plain = f"no pick {tag()}"
        make_video(client, upstream, plain)
        on_model = f"model default {tag()}"
        make_video(client, upstream, on_model, chat_options={"model": director})
        picked = f"picked {tag()}"
        make_video(
            client,
            upstream,
            picked,
            chat_options={
                "model": director,
                "features": {"video_generation": {"model": refs["picked"]}},
            },
        )

    def model_sent(prompt: str) -> str:
        return form_fields(job_for(listener, prompt))["model"][0].decode()

    assert model_sent(plain) == "admin-default", "the admin's default video model was not used"
    assert model_sent(on_model) == "model-default", "the chat model's video model was not used"
    assert model_sent(picked) == "picked", "the video model picked in the chat was not used"


def test_a_connection_shared_with_some_users_offers_its_models_to_them_alone(
    videos, listener, upstream, admin, make_user
):
    insider, outsider = make_user(), make_user()
    make_group(admin, [insider, outsider], VIDEO_PERMISSION)
    public = connection_at(listener, "public-model")
    private = connection_at(
        listener, "private-model", access_grants=[grant("user", insider.id, "read")]
    )
    videos(public, private, default_model=model_ref(public["id"], "public-model"))
    private_ref = model_ref(private["id"], "private-model")

    with insider.client() as client:
        insider_models = {model["id"] for model in client.get("/api/v1/videos/models").json()}
    with outsider.client() as client:
        outsider_models = {model["id"] for model in client.get("/api/v1/videos/models").json()}
        refused = client.post(
            "/api/v1/videos/generations", json={"prompt": "a secret", "model": private_ref}
        )
        prompt = f"a private pick {tag()}"
        make_video(
            client,
            upstream,
            prompt,
            chat_options={"features": {"video_generation": {"model": private_ref}}},
        )

    assert private_ref in insider_models, insider_models
    assert outsider_models == {model_ref(public["id"], "public-model")}, outsider_models
    assert refused.status_code == 400, refused.text
    sent_models = [form_fields(job)["model"][0].decode() for job in created_jobs(listener)]
    assert "private-model" not in sent_models, "an outsider's video ran on the private connection"
    assert form_fields(job_for(listener, prompt))["model"][0] == b"public-model"


def test_a_failed_video_job_tells_the_model_and_stores_nothing(
    videos, listener, upstream, filmmaker
):
    videos(connection_at(listener, "sora-2"))

    with filmmaker.client() as client:
        stored, result = make_video(client, upstream, f"{FAILING_PROMPT} {tag()}")

    assert FAILURE_MESSAGE in result, f"the model was not told why the video failed: {result}"
    assert "error" in json.loads(result), result
    assert videos_on(stored) == [], f"a failed job left a video on the reply: {stored['files']}"
