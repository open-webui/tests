"""Journey: a realtime voice call relayed by the server between the browser and a voice provider.

With Call mode set to Realtime under Admin Settings > Audio, a call opens a WebSocket to the
instance, signs in with its first message and names the chat model; the server checks who may
call with which model, opens the provider's Realtime session with the admin's key, model, voice,
transcription model and prompt (a model's own voice replacing the admin's) and then relays: the
microphone, the earlier conversation and function results go to the provider, and its speech,
transcripts and function calls come back. Here the provider is `harness.realtime_provider`.
Every refusal and provider failure ends the call with a message the browser shows, never the
provider's own error text.

Discriminates: in a backend copy, the session voice ignoring the model's own voice turns the
model voice test red, the `bridge.result` command no longer passed on as a function output turns
the relay test red, the call permission check removed turns the permission test red, and a
provider error passed on verbatim turns the provider error tests red.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness.audio_engine import AUDIO_CONFIG
from harness.chat_history import seed_chat
from harness.instance import free_port
from harness.realtime_provider import (
    API_KEY,
    PROVIDER_ERROR_TEXT,
    TRANSCRIPTION_MODEL,
    VOICE,
    VOICE_MODEL,
    call_error,
    next_event,
    realtime_call,
    realtime_settings,
    restoring_audio_settings,
    save_realtime,
    serving_realtime_provider,
    spoken_audio,
    using_realtime,
)
from harness.upstream import MOCK_MODEL_ID

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
DEFAULT_PROMPT_START = "You are the assistant in this chat, speaking with the user."


@pytest.fixture
def provider():
    with serving_realtime_provider() as fake:
        yield fake


@pytest.fixture
def realtime(admin, provider):
    with admin.client() as client, using_realtime(client, provider):
        yield provider


@pytest.fixture
def realtime_settings_restored(admin):
    with admin.client() as client, restoring_audio_settings(client):
        yield client


def open_call(actor, model_id: str = MOCK_MODEL_ID, chat_id: str | None = None):
    return realtime_call(actor.base_url, actor.token, model_id, chat_id)


def refusal(actor, model_id: str = MOCK_MODEL_ID, chat_id: str | None = None) -> str | None:
    with open_call(actor, model_id, chat_id) as session:
        return call_error(session)


@pytest.fixture
def model_with_voice(admin):
    model_id = f"voiced-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Voiced model",
        "meta": {"voice": {"voice": "cedar"}},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield model_id
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


@pytest.fixture
def private_model(admin):
    model_id = f"private-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": "Private model",
        "meta": {},
        "params": {},
        "access_grants": [],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    yield model_id
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_saved_call_settings_are_read_back_and_published_without_the_key(
    realtime_settings_restored, provider, make_user
):
    settings = realtime_settings(provider, REALTIME_CALL_PROMPT_TEMPLATE="Speak briefly.")
    save_realtime(realtime_settings_restored, settings)

    stored = realtime_settings_restored.get(AUDIO_CONFIG[0]).json()["realtime"]
    with make_user().client() as client:
        published = client.get("/api/config").json()

    assert stored == settings
    assert published["audio"]["realtime"] == {
        "enabled": True,
        "model": VOICE_MODEL,
        "voice": VOICE,
    }
    assert API_KEY not in json.dumps(published)


def test_calls_are_off_until_the_admin_turns_them_on(admin, provider, make_user):
    with admin.client() as client:
        stored = client.get(AUDIO_CONFIG[0]).json()["realtime"]

    assert stored["ENABLED"] is False
    assert refusal(make_user()) == "Realtime calls are disabled"
    assert provider.calls == []


def test_saving_tts_and_stt_alone_keeps_the_call_settings(admin, realtime):
    with admin.client() as client:
        current = client.get(AUDIO_CONFIG[0]).json()
        saved = client.post(AUDIO_CONFIG[1], json={"tts": current["tts"], "stt": current["stt"]})
        after = client.get(AUDIO_CONFIG[0]).json()["realtime"]

    assert saved.status_code == 200, saved.text
    assert after == current["realtime"]


def test_an_empty_voice_model_is_refused(realtime_settings_restored, provider):
    current = realtime_settings_restored.get(AUDIO_CONFIG[0]).json()
    form = {**current, "realtime": realtime_settings(provider, MODEL="")}

    saved = realtime_settings_restored.post(AUDIO_CONFIG[1], json=form)

    assert saved.status_code == 422, saved.text


def test_a_call_opens_the_provider_session_the_admin_configured(realtime, make_user):
    with open_call(make_user()) as session:
        ready = next_event(session, "bridge.ready")

    assert ready == {
        "type": "bridge.ready",
        "model": VOICE_MODEL,
        "voice": VOICE,
        "sample_rate": 24000,
    }
    call = realtime.calls[0]
    assert call.path == f"/v1/realtime?model={VOICE_MODEL}"
    assert call.headers["authorization"] == f"Bearer {API_KEY}"
    session = call.session
    assert session["instructions"].startswith(DEFAULT_PROMPT_START)
    assert session["audio"]["input"]["transcription"] == {"model": TRANSCRIPTION_MODEL}
    assert session["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert session["audio"]["output"]["voice"] == VOICE
    assert [tool["name"] for tool in session["tools"]] == ["generate_chat_completion"]


def test_a_custom_prompt_template_is_the_session_instructions(admin, provider, make_user):
    prompt = "Speak like a harbour master and keep it short."
    with (
        admin.client() as client,
        using_realtime(client, provider, REALTIME_CALL_PROMPT_TEMPLATE=prompt),
    ):
        with open_call(make_user()) as session:
            next_event(session, "bridge.ready")

    assert provider.calls[0].session["instructions"] == prompt


def test_a_models_own_voice_replaces_the_admins(realtime, model_with_voice, make_user):
    with open_call(make_user(), model_with_voice) as session:
        ready = next_event(session, "bridge.ready")

    assert ready["voice"] == "cedar"
    assert realtime.calls[0].session["audio"]["output"]["voice"] == "cedar"


@pytest.mark.parametrize("voice", ["two words", ""])
def test_a_model_voice_must_be_one_word(admin, voice):
    form = {
        "id": f"badvoice-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": "Bad voice",
        "meta": {"voice": {"voice": voice}},
        "params": {},
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)

    assert created.status_code == 422, created.text


def test_the_call_relays_speech_function_calls_and_results(realtime, make_user):
    realtime.hears("when is high tide")
    with open_call(make_user()) as session:
        next_event(session, "bridge.ready")
        earlier = [
            {"role": "user", "content": "hello harbour"},
            {"role": "assistant", "content": "hello sailor"},
        ]
        session.send(json.dumps({"type": "bridge.history", "messages": earlier}))
        session.send(json.dumps({"type": "input_audio_buffer.append", "audio": spoken_audio()}))
        heard = next_event(session, "conversation.item.input_audio_transcription.completed")
        session.send(json.dumps({"type": "bridge.respond", "item_id": heard["item_id"]}))
        delegated = next_event(session, "response.output_item.done")["item"]
        result = {
            "type": "bridge.result",
            "call_id": delegated["call_id"],
            "status": "completed",
            "answer": "High tide is at six.",
        }
        session.send(json.dumps(result))
        session.send(json.dumps({"type": "bridge.respond", "call_id": delegated["call_id"]}))
        spoken = next_event(session, "response.output_audio_transcript.done")
        session.send(json.dumps({"type": "bridge.ping"}))
        next_event(session, "bridge.pong")

    assert heard["transcript"] == "when is high tide"
    assert json.loads(delegated["arguments"]) == {"request": "when is high tide"}
    assert spoken["transcript"] == "High tide is at six."
    call = realtime.calls[0]
    history = [event["item"] for event in call.received("conversation.item.create")]
    assert history[:2] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hello harbour"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hello sailor"}],
        },
    ]
    function_output = history[2]
    assert (function_output["type"], function_output["call_id"]) == (
        "function_call_output",
        delegated["call_id"],
    )
    assert json.loads(function_output["output"]) == {
        "status": "completed",
        "answer": "High tide is at six.",
    }
    responses = [event["response"] for event in call.received("response.create")]
    assert responses[0]["metadata"] == {"input_item_id": heard["item_id"]}
    assert responses[1]["metadata"] == {"call_id": delegated["call_id"]}
    assert responses[1]["tools"] == []
    assert len(call.received("input_audio_buffer.append")) == 1


def test_the_browser_only_hears_response_conversation_and_input_events(realtime, make_user):
    with open_call(make_user()) as session:
        first = json.loads(session.recv(timeout=10))
        realtime.calls[0].send({"type": "rate_limits.updated", "rate_limits": []})
        realtime.calls[0].send({"type": "session.updated", "session": {"instructions": "x"}})
        realtime.calls[0].send({"type": "input_audio_buffer.committed", "item_id": "item_1"})
        relayed = json.loads(session.recv(timeout=10))

    assert first["type"] == "bridge.ready"
    assert relayed == {"type": "input_audio_buffer.committed", "item_id": "item_1"}


@pytest.mark.parametrize(
    ("command", "message"),
    [
        ({"type": "session.update", "session": {"voice": "other"}}, "Unsupported call command"),
        (
            {"type": "bridge.respond", "item_id": "item_unheard"},
            "Unknown or already answered input",
        ),
        (
            {
                "type": "bridge.result",
                "call_id": "call_unknown",
                "status": "completed",
                "answer": "x",
            },
            "Unknown or resolved function call",
        ),
    ],
    ids=["session-update", "unheard-input", "unknown-function-call"],
)
def test_a_command_the_call_does_not_allow_ends_it(realtime, make_user, command, message):
    with open_call(make_user()) as session:
        next_event(session, "bridge.ready")
        session.send(json.dumps(command))
        ended_with = call_error(session)

    assert ended_with == message
    assert len(realtime.calls[0].received("session.update")) == 1
    realtime.wait_for(lambda: realtime.calls[0].ended.is_set(), "the call end")


def test_without_the_call_permission_a_user_cannot_call_but_an_admin_can(
    realtime, admin, make_user, preserve
):
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["call"] = False
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text

    assert refusal(make_user()) == "Call permission denied"
    with open_call(admin) as session:
        assert next_event(session, "bridge.ready")["voice"] == VOICE


def test_a_call_into_someone_elses_chat_is_refused(realtime, make_user):
    owner, stranger = make_user(), make_user()
    with owner.client() as client:
        chat_id, _ = seed_chat(client, [{"role": "user", "content": "mine"}])

    assert refusal(stranger, chat_id=chat_id) == "Chat not found"
    with open_call(owner, chat_id=chat_id) as session:
        assert next_event(session, "bridge.ready")


def test_a_model_the_user_cannot_read_is_refused(realtime, private_model, make_user):
    assert refusal(make_user(), private_model) == "Chat model access denied"


def test_an_unknown_model_is_refused(realtime, make_user):
    unknown = f"nowhere-{uuid.uuid4().hex[:6]}"
    assert refusal(make_user(), unknown) == "Bridge requires a server-configured chat model"


def test_a_call_without_a_model_is_refused(realtime, make_user):
    caller = make_user()
    with realtime_call(caller.base_url, caller.token, None) as session:
        assert call_error(session) == "Select a chat model"


def test_a_call_with_a_bad_token_is_refused(realtime, make_user):
    caller = make_user()
    with realtime_call(caller.base_url, "not-a-token", MOCK_MODEL_ID) as session:
        assert call_error(session) == "Authentication expired or invalid"
    assert realtime.calls == []


def test_a_provider_that_cannot_be_reached_ends_the_call(admin, provider, make_user):
    unreachable = f"http://127.0.0.1:{free_port()}/v1"
    with (
        admin.client() as client,
        using_realtime(client, provider, OPENAI_API_BASE_URL=unreachable),
    ):
        assert refusal(make_user()) == "Voice connection failed"


def test_a_provider_refusing_the_connection_ends_the_call(realtime, make_user):
    realtime.refuse_handshake = 401

    assert refusal(make_user()) == "Voice connection failed"


def test_a_provider_refusing_the_session_ends_the_call_without_its_error_text(realtime, make_user):
    realtime.refuse_session = True

    message = refusal(make_user())

    assert message == (
        "Provider rejected voice configuration. Check model, voice, and transcription model."
    )


def test_a_provider_error_during_the_call_ends_it_without_its_error_text(realtime, make_user):
    with open_call(make_user()) as session:
        next_event(session, "bridge.ready")
        error = {"type": "invalid_request_error", "message": PROVIDER_ERROR_TEXT}
        realtime.calls[0].send({"type": "error", "error": error})
        ended_with = call_error(session)

    assert ended_with == "Voice provider rejected a request"


def test_a_provider_dropping_the_call_ends_it(realtime, make_user):
    with open_call(make_user()) as session:
        next_event(session, "bridge.ready")
        realtime.calls[0].drop()
        ended_with = call_error(session)

    assert ended_with == "Voice provider connection closed"


@pytest.mark.parametrize(
    ("setting", "message"),
    [
        ({"OPENAI_API_BASE_URL": "ftp://127.0.0.1/v1"}, "Invalid Realtime provider URL"),
        (
            {"OPENAI_API_KEY": ""},
            "Configure the Realtime API key, model, voice, and transcription model",
        ),
    ],
    ids=["not-http", "no-key"],
)
def test_incomplete_provider_settings_end_the_call(admin, provider, make_user, setting, message):
    with admin.client() as client, using_realtime(client, provider, **setting):
        assert refusal(make_user()) == message
    assert provider.calls == []
