"""Journey: a model's voice avatar in a realtime call, in place of the voice orb.

Since d989375b4 a model can carry a VRM avatar with optional state clips and named gestures.
A realtime call on that model loads the avatar (and its clips) through the model's read access
and shows it, a WebGL canvas, where the voice orb sat; until it is ready, and for good if it
cannot load (a file gone, a rig the browser's VRM loader refuses), the orb shows instead and the
call goes on. The voice provider is offered a `play_animation` function naming the gestures; a
gesture it plays is performed by the avatar and the call tells the provider whether it started,
or that it was unavailable (no avatar on screen, reduced motion, a gesture the model does not
have), and then asks it to go on speaking. The avatar is drawn on a canvas, so a test sees it
through that canvas and the orb, and its gestures through what the provider is told. Here the
provider is `harness.realtime_provider` and the files come from `harness.voice_avatars`.

Discriminates: passes on the dev d989375b4 build. A frontend copy of it whose call overlay never
shows the avatar turns the avatar and gesture started tests red, one whose overlay never falls
back to the orb turns both fallback tests and the failed avatar case red, and a backend copy
without the avatar's files among the model's readable files turns the avatar and gesture
started tests red, the caller only reading the model. The reduced motion and unknown gesture
cases pass there as well; integration/audio/test_realtime_calls.py proves the latter.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.realtime_provider import serving_realtime_provider, using_realtime
from harness.upstream import MOCK_MODEL_ID
from harness.voice_avatars import SERVER_CHECKED_BONES, animation, avatar, upload
from utils.chat_ui import conversation, expect_reply
from utils.model_editor import open_chat_on
from utils.voice_call import TURN_TIMEOUT_MS, call_status, start_call

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
WAVE = "Wave when greeting someone."
STARTED = "The requested gesture has started and is visible to the user."


@pytest.fixture
def realtime(admin, e2e_instance):
    with serving_realtime_provider() as provider:
        with admin.client() as client, using_realtime(client, provider):
            yield provider


@pytest.fixture
def avatar_model(admin):
    """`avatar_model(content, gesture=False)`: a model everyone reads, with the admin's avatar."""
    created_ids = []

    def create(content: bytes = avatar(), gesture: bool = False) -> dict:
        settings = {"file_id": upload(admin, "pilot.vrm", content)}
        if gesture:
            settings["states"] = {"idle": {"file_id": upload(admin, "idle.vrma", animation())}}
            wave_clip = upload(admin, "wave.vrma", animation())
            settings["gestures"] = [{"name": "wave", "description": WAVE, "file_id": wave_clip}]
        model = {
            "id": f"avatar-{uuid.uuid4().hex[:8]}",
            "base_model_id": MOCK_MODEL_ID,
            "name": f"Harbour pilot {uuid.uuid4().hex[:4]}",
            "meta": {"voice_avatar": settings},
            "params": {},
            "access_grants": [EVERYONE_READS],
        }
        with admin.client() as client:
            created = client.post("/api/v1/models/create", json=model)
            assert created.status_code == 200, created.text
            client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        created_ids.append(model["id"])
        return model

    yield create
    with admin.client() as client:
        for model_id in created_ids:
            client.post("/api/v1/models/model/delete", json={"id": model_id})


def call_stage(page: Page):
    return page.get_by_role("button", name="Stop speaking")


def expect_avatar_shown(page: Page) -> None:
    # the orb shows over the avatar's canvas until the avatar is ready
    expect(call_stage(page).locator(".avatar-canvas")).to_have_count(1, timeout=TURN_TIMEOUT_MS)
    expect(call_stage(page).locator(".voice-orb")).to_have_count(0, timeout=TURN_TIMEOUT_MS)


def expect_orb_shown(page: Page) -> None:
    expect(call_stage(page).locator(".avatar-canvas")).to_have_count(0, timeout=TURN_TIMEOUT_MS)
    expect(call_stage(page).locator(".voice-orb")).to_have_count(1)


def call_on(page: Page, model: dict) -> list[str]:
    """Start a call on `model`; the URLs the page has finished loading, as they finish."""
    loaded: list[str] = []
    page.on("requestfinished", lambda request: loaded.append(request.url))
    open_chat_on(page, model)
    start_call(page)
    return loaded


def file_fetched(fetched: list[str], file_id: str) -> bool:
    return any(f"/api/v1/files/{file_id}/content" in url for url in fetched)


def gesture_statuses(realtime) -> list[str]:
    realtime.wait_for(lambda: realtime.calls and realtime.calls[-1].gesture_results, "a gesture")
    return [result["status"] for result in realtime.calls[-1].gesture_results]


def test_a_model_with_an_avatar_shows_it_in_the_call_in_place_of_the_orb(
    voice_page_for, make_user, realtime, avatar_model
):
    model = avatar_model()
    page = voice_page_for(make_user())

    fetched = call_on(page, model)

    expect_avatar_shown(page)
    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    assert file_fetched(fetched, model["meta"]["voice_avatar"]["file_id"])


def test_a_model_without_an_avatar_keeps_the_orb(voice_page_for, make_user, realtime):
    page = voice_page_for(make_user())

    start_call(page)

    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    expect_orb_shown(page)


@pytest.mark.parametrize("failure", ["rig-the-browser-refuses", "file-deleted"])
def test_an_avatar_that_cannot_load_falls_back_to_the_orb_and_the_call_goes_on(
    voice_page_for, make_user, admin, realtime, avatar_model, failure
):
    if failure == "rig-the-browser-refuses":
        model = avatar_model(avatar(bones=SERVER_CHECKED_BONES))
    else:
        model = avatar_model()
        with admin.client() as client:
            file_id = model["meta"]["voice_avatar"]["file_id"]
            client.delete(f"/api/v1/files/{file_id}").raise_for_status()
    greeting, answer = f"ahoy {uuid.uuid4().hex[:6]}", f"Ahoy, sailor {uuid.uuid4().hex[:6]}."
    page = voice_page_for(make_user())

    call_on(page, model)

    expect_orb_shown(page)
    expect(call_status(page, "Listening")).to_be_visible(timeout=TURN_TIMEOUT_MS)
    realtime.hears(greeting, answers=answer)
    expect_reply(page, answer)


def test_the_voice_model_plays_a_gesture_on_the_avatar_and_hears_it_started(
    voice_page_for, make_user, realtime, avatar_model
):
    model = avatar_model(gesture=True)
    settings = model["meta"]["voice_avatar"]
    greeting, answer = f"hello pilot {uuid.uuid4().hex[:6]}", f"Hello there {uuid.uuid4().hex[:6]}."
    page = voice_page_for(make_user())
    fetched = call_on(page, model)
    expect_avatar_shown(page)
    for clip in (settings["states"]["idle"]["file_id"], settings["gestures"][0]["file_id"]):
        realtime.wait_for(lambda: file_fetched(fetched, clip), "the avatar's clips loaded")

    realtime.hears(greeting, answers=answer, gesture="wave")

    assert gesture_statuses(realtime) == ["started"]
    realtime.wait_for(lambda: STARTED in realtime.spoken, "the call going on after the gesture")
    call = realtime.calls[-1]
    tools = {tool["name"]: tool for tool in call.session["tools"]}
    assert list(tools) == ["generate_chat_completion", "play_animation"]
    assert tools["play_animation"]["parameters"]["properties"]["name"]["enum"] == ["wave"]
    assert WAVE in tools["play_animation"]["description"]
    assert "Your avatar is your visible presence in this call." in call.session["instructions"]
    follow_up = call.received("response.create")[-1]["response"]
    assert list(follow_up["metadata"]) == ["input_item_id"]
    assert [tool["name"] for tool in follow_up["tools"]] == ["generate_chat_completion"]
    expect(conversation(page).get_by_text(answer)).to_be_visible()


@pytest.mark.parametrize(
    ("situation", "gesture"),
    [("reduced-motion", "wave"), ("unknown-gesture", "cartwheel"), ("avatar-failed", "wave")],
)
def test_a_gesture_that_cannot_play_is_reported_unavailable(
    voice_page_for, make_user, realtime, avatar_model, situation, gesture
):
    content = avatar(bones=SERVER_CHECKED_BONES) if situation == "avatar-failed" else avatar()
    model = avatar_model(content, gesture=True)
    motion = {"reduced_motion": "reduce"} if situation == "reduced-motion" else {}
    page = voice_page_for(make_user(), **motion)
    call_on(page, model)
    if situation == "avatar-failed":
        expect_orb_shown(page)
    else:
        expect_avatar_shown(page)

    realtime.hears(f"show me {uuid.uuid4().hex[:6]}", answers="Here you go.", gesture=gesture)

    assert gesture_statuses(realtime) == ["unavailable"]
    effect = json.dumps(realtime.calls[-1].gesture_results[0])
    assert "could not start" in effect
