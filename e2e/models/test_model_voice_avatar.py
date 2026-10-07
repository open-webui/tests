"""Journey: setting up a model's voice avatar in the workspace model editor.

Since d989375b4 the model editor has a Voice avatar row (Default orb or Custom avatar) while
realtime calls are on or the model already has one. Configure opens Avatar setup: Upload VRM
takes a .vrm or .glb up to 25 MiB (anything else is turned away at once) and shows it in a live
preview (Loading avatar... until it is ready, then Idle, Listening and Speaking to try), warning
when the avatar has no mouth or blink expression; a file that is no VRM, or one the browser's
VRM loader refuses, shows why and cannot be applied. State animations take a VRMA clip each for
idle, listening and speaking, and Named gestures a lowercase name, a description and a clip,
previewed on the avatar; Apply is held back until every gesture has all three. Apply and Save &
Update upload the files and store their ids under the model's `voice_avatar`; Use orb removes
the avatar on the next save and Cancel leaves it as it was. Files come from
`harness.voice_avatars`.

Discriminates: passes on the dev d989375b4 build. In a frontend copy of it whose save leaves the
avatar as loaded the upload, clips and gesture and Use orb tests turn red, and in one whose
Apply ignores what the preview found the refused avatar and lowercase name tests turn red; one
that never shows the expression warnings or accepts any gesture name turns those tests red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.realtime_provider import serving_realtime_provider, using_realtime
from harness.upstream import MOCK_MODEL_ID
from harness.voice_avatars import (
    SERVER_CHECKED_BONES,
    animation,
    avatar,
    glb,
    grown_to,
    upload,
)
from utils.model_editor import open_editor, save

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

LOAD_TIMEOUT_MS = 20_000


@pytest.fixture
def realtime_on(admin, e2e_instance):
    with serving_realtime_provider() as provider:
        with admin.client() as client, using_realtime(client, provider):
            yield


@pytest.fixture
def model(admin):
    """A model of the admin's on the scripted model; `model["meta"]` is filled in before use."""
    created = {
        "id": f"avatar-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": f"Harbour pilot {uuid.uuid4().hex[:4]}",
        "meta": {},
        "params": {},
    }
    yield created
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": created["id"]})


def create(admin, model: dict) -> dict:
    with admin.client() as client:
        response = client.post("/api/v1/models/create", json=model)
    assert response.status_code == 200, response.text
    return model


def stored_avatar(admin, model: dict):
    with admin.client() as client:
        stored = client.get("/api/v1/models/model", params={"id": model["id"]}).json()
    return stored["meta"].get("voice_avatar")


def file_content(admin, file_id: str) -> bytes:
    with admin.client() as client:
        response = client.get(f"/api/v1/files/{file_id}/content")
    assert response.status_code == 200, response.text
    return response.content


def avatar_row(editor: Locator) -> Locator:
    return editor.get_by_text("Voice avatar", exact=True).locator("xpath=../..")


def open_setup(editor: Locator) -> Locator:
    avatar_row(editor).get_by_role("button", name="Configure").click()
    setup = editor.page.get_by_role("dialog")
    expect(setup.get_by_role("heading", name="Avatar setup")).to_be_visible()
    return setup


def choose(page: Page, button: Locator, name: str, content: bytes) -> None:
    with page.expect_file_chooser() as chooser:
        button.click()
    chooser.value.set_files(
        {"name": name, "mimeType": "application/octet-stream", "buffer": content}
    )


def choose_avatar(setup: Locator, content: bytes, name: str = "pilot.vrm") -> None:
    upload_button = setup.get_by_role("button", name="Upload VRM")
    choose(setup.page, upload_button, name, content)


def expect_preview_ready(setup: Locator) -> None:
    expect(setup.get_by_label("Avatar preview").locator("canvas")).to_have_count(1)
    expect(setup.get_by_text("Loading avatar…")).to_have_count(0, timeout=LOAD_TIMEOUT_MS)
    expect(setup.get_by_role("button", name="Speaking", exact=True)).to_be_enabled()


def test_an_uploaded_avatar_previews_and_is_saved_with_the_model(
    page_for, admin, realtime_on, model
):
    create(admin, model)
    content = avatar()
    editor = open_editor(page_for(admin), model)
    expect(avatar_row(editor).get_by_text("Default orb")).to_be_visible()
    setup = open_setup(editor)

    choose_avatar(setup, content)

    expect_preview_ready(setup)
    behaviour = setup.get_by_role("group", name="Preview behavior")
    behaviour.get_by_role("button", name="Speaking").click()
    expect(behaviour.get_by_role("button", name="Speaking")).to_have_attribute(
        "aria-pressed", "true"
    )
    expect(setup.get_by_role("alert")).to_have_count(0)
    setup.get_by_role("button", name="Apply").click()
    expect(avatar_row(editor).get_by_text("Custom avatar")).to_be_visible()
    save(editor)

    stored = stored_avatar(admin, model)
    assert stored == {"file_id": stored["file_id"], "states": {}, "gestures": []}
    assert file_content(admin, stored["file_id"]) == content


def test_an_avatar_without_mouth_or_blink_is_flagged_in_the_preview(
    page_for, admin, realtime_on, model
):
    create(admin, model)
    setup = open_setup(open_editor(page_for(admin), model))

    choose_avatar(setup, avatar(expressions=()))

    expect_preview_ready(setup)
    expect(setup.get_by_text("This avatar has no mouth expression")).to_be_visible()
    expect(setup.get_by_text("This avatar has no blink expressions.")).to_be_visible()


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        (
            "plain.glb",
            glb({"asset": {"version": "2.0"}, "nodes": [{}], "buffers": [{"byteLength": 4}]}),
            "This model needs a VRM humanoid rig. A plain GLB is not enough.",
        ),
        (
            "half-rigged.vrm",
            avatar(bones=SERVER_CHECKED_BONES),
            "These humanoid bones are required but not exist",
        ),
    ],
    ids=["plain-glb", "rig-the-browser-refuses"],
)
def test_an_avatar_the_preview_cannot_load_shows_why_and_cannot_be_applied(
    page_for, admin, realtime_on, model, name, content, reason
):
    create(admin, model)
    editor = open_editor(page_for(admin), model)
    setup = open_setup(editor)

    choose_avatar(setup, content, name)

    expect(setup.get_by_role("alert")).to_contain_text(reason, timeout=LOAD_TIMEOUT_MS)
    expect(setup.get_by_role("button", name="Apply")).to_be_disabled()
    expect(setup.get_by_role("button", name="Speaking", exact=True)).to_be_disabled()


@pytest.mark.parametrize(
    ("name", "size"),
    [("pilot.png", 0), ("big.vrm", 25 * 1024 * 1024 + 4)],
    ids=["png", "over-25-mib"],
)
def test_a_file_of_the_wrong_kind_or_size_is_turned_away(
    page_for, admin, realtime_on, model, name, size
):
    create(admin, model)
    setup = open_setup(open_editor(page_for(admin), model))

    choose_avatar(setup, grown_to(avatar(), size) if size else avatar(), name)

    expect(setup.get_by_role("alert")).to_have_text("Choose a VRM file, up to 25 MiB.")
    expect(setup.get_by_label("Avatar preview")).to_have_count(0)


def test_state_clips_and_a_gesture_are_uploaded_and_saved(page_for, admin, realtime_on, model):
    create(admin, model)
    page = page_for(admin)
    editor = open_editor(page, model)
    setup = open_setup(editor)
    choose_avatar(setup, avatar())
    expect_preview_ready(setup)
    states = setup.get_by_role("region", name="State animations")
    gestures = setup.get_by_role("region", name="Named gestures")

    choose(page, states.get_by_role("button", name="Upload idle VRMA"), "drift.vrma", animation())
    expect(states.get_by_text("drift.vrma")).to_be_visible()
    gestures.get_by_role("button", name="Add gesture").click()
    expect(setup.get_by_role("button", name="Apply")).to_be_disabled()
    gestures.get_by_role("textbox", name="Gesture name").fill("wave")
    gestures.get_by_role("textbox", name="Gesture description").fill("Wave when greeting someone.")
    choose(
        page,
        gestures.get_by_role("button", name="Upload gesture 1 VRMA"),
        "wave.vrma",
        animation(2),
    )
    preview_wave = gestures.get_by_role("button", name="Preview gesture 1")
    expect(preview_wave).to_be_enabled(timeout=LOAD_TIMEOUT_MS)
    preview_wave.click()
    expect(setup.get_by_role("alert")).to_have_count(0)
    setup.get_by_role("button", name="Apply").click()
    save(editor)

    stored = stored_avatar(admin, model)
    idle_clip = stored["states"]["idle"]["file_id"]
    [wave] = stored["gestures"]
    assert (wave["name"], wave["description"]) == ("wave", "Wave when greeting someone.")
    assert len({stored["file_id"], idle_clip, wave["file_id"]}) == 3
    assert file_content(admin, idle_clip) == animation()
    assert file_content(admin, wave["file_id"]) == animation(2)


@pytest.mark.parametrize("name", ["Wave", "wave hello", "1wave"])
def test_a_gesture_needs_a_lowercase_name_before_the_setup_applies(
    page_for, admin, realtime_on, model, name
):
    create(admin, model)
    page = page_for(admin)
    setup = open_setup(open_editor(page, model))
    choose_avatar(setup, avatar())
    expect_preview_ready(setup)
    gestures = setup.get_by_role("region", name="Named gestures")
    gestures.get_by_role("button", name="Add gesture").click()
    choose(
        page, gestures.get_by_role("button", name="Upload gesture 1 VRMA"), "wave.vrma", animation()
    )
    gestures.get_by_role("textbox", name="Gesture description").fill("Wave hello.")

    gestures.get_by_role("textbox", name="Gesture name").fill(name)

    expect(gestures.get_by_text("Each gesture needs a clip, a description")).to_be_visible()
    expect(setup.get_by_role("button", name="Apply")).to_be_disabled()
    gestures.get_by_role("textbox", name="Gesture name").fill("wave")
    expect(setup.get_by_role("button", name="Apply")).to_be_enabled()


def test_use_orb_removes_the_avatar_and_cancel_keeps_it(page_for, admin, realtime_on, model):
    model["meta"] = {"voice_avatar": {"file_id": upload(admin, "pilot.vrm", avatar())}}
    create(admin, model)
    editor = open_editor(page_for(admin), model)
    expect(avatar_row(editor).get_by_text("Custom avatar")).to_be_visible()

    setup = open_setup(editor)
    expect_preview_ready(setup)
    setup.get_by_role("button", name="Use orb").click()
    setup.get_by_role("button", name="Cancel").click()
    expect(avatar_row(editor).get_by_text("Custom avatar")).to_be_visible()

    setup = open_setup(editor)
    setup.get_by_role("button", name="Use orb").click()
    setup.get_by_role("button", name="Apply").click()
    expect(avatar_row(editor).get_by_text("Default orb")).to_be_visible()
    save(editor)

    assert stored_avatar(admin, model) is None


def test_without_realtime_calls_only_a_model_with_an_avatar_shows_the_row(page_for, admin, model):
    create(admin, model)
    page = page_for(admin)

    editor = open_editor(page, model)
    expect(editor.get_by_text("Description", exact=True).first).to_be_visible()
    expect(editor.get_by_text("Voice avatar", exact=True)).to_have_count(0)

    with admin.client() as client:
        settings = {"voice_avatar": {"file_id": upload(admin, "pilot.vrm", avatar())}}
        updated = client.post("/api/v1/models/model/update", json={**model, "meta": settings})
    assert updated.status_code == 200, updated.text
    editor = open_editor(page, model)
    expect(avatar_row(editor).get_by_text("Custom avatar")).to_be_visible()
