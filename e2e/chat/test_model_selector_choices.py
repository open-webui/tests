"""Journey: the model selector keeps a default model and compares two models side by side.

Set as default saves the chosen model to the account's settings: a later new chat, and one after a
reload, opens on that model and sends its message there. With Compare turned on, choosing two
models makes both answer the next message side by side; turning Compare off leaves one model, and
the next message gets one answer. An account that saved two models opens a new chat on both only
while Multiple Models is allowed; without it the chat starts on the first, and only that model is
asked. The models are two presets on the scripted provider, told apart by their system prompts in
what the provider is sent.

Discriminates: passes on dev 30f3f6a8f; in a backend copy whose settings update drops the saved
models both tests fail, and in a frontend build whose Compare button does nothing the Compare test
fails with one model answering. In the pre-fix build of 0.12.4 (5d75c1905 reverted) the
Multiple Models test fails with the permission withheld: the new chat opens on both models.

A saved chat that two models answered shows both answers side by side in the main chat and when
dragged into a tab beside another chat (`utils/chat_surface.py`). Opened in a tab it showed one
answer (open-webui/open-webui#34030, fix PR open-webui/open-webui#34064), so the tab variants are
red on dev c15f28649.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import Actor
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import DRAGGED_TAB, MAIN_CHAT, SIDE_CHAT
from utils.chat_ui import REPLY_TIMEOUT_MS, chat_input, conversation, expect_reply, replies, send
from utils.model_selector import SELECTOR_BUTTON, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def _preset(admin: Actor, name: str, system: str) -> str:
    model_id = f"{name}-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "name": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "meta": {},
        "params": {"system": system},
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
    assert created.status_code == 200, created.text
    return model_id


def _asked_as(prompt: str, system: str):
    """A `match` for the request carrying `prompt` to the preset whose system prompt is `system`."""
    prompted = reply.answering(prompt)

    def matches(body: dict) -> bool:
        return prompted(body) and system in json.dumps(body["messages"])

    return matches


def _requests_for(upstream, prompt: str, system: str) -> list[dict]:
    return [body for body in upstream.chat_requests() if _asked_as(prompt, system)(body)]


@pytest.fixture
def two_presets(make_user) -> tuple[Actor, str, str]:
    """A fresh admin and two presets whose system prompts say which of them answered."""
    admin = make_user(role="admin")
    return (
        admin,
        _preset(admin, "first-pick", "You are the first preset."),
        _preset(admin, "second-pick", "You are the second preset."),
    )


def _selected_model(page: Page, name: str):
    return page.get_by_role("button", name=f"Selected model: {name}")


def test_set_as_default_opens_later_chats_on_that_model(page_for, two_presets, upstream):
    admin, first, second = two_presets
    page = page_for(admin)
    page.get_by_role("button", name=SELECTOR_BUTTON).wait_for()
    expect(_selected_model(page, first)).to_have_count(0)
    select_model(page, first)
    expect(_selected_model(page, first)).to_be_visible()

    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    page.get_by_role("button", name="Set as default").click()
    expect(page.get_by_text("Default model updated")).to_be_visible()
    page.keyboard.press("Escape")

    page.get_by_role("link", name="New Chat").first.click()
    expect(_selected_model(page, first)).to_be_visible()
    page.reload()
    expect(_selected_model(page, first)).to_be_visible()

    prompt = "who answers by default?"
    upstream.queue(reply.text("the default answered", match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, "the default answered")
    assert len(_requests_for(upstream, prompt, "You are the first preset.")) == 1
    assert _requests_for(upstream, prompt, "You are the second preset.") == []
    with admin.client() as client:
        stored = client.get("/api/v1/users/user/settings").json()
    assert stored["ui"]["models"] == [first]
    assert second not in stored["ui"]["models"]


def test_compare_answers_with_both_models_and_leaves_one_when_turned_off(
    page_for, two_presets, upstream
):
    admin, first, second = two_presets
    with admin.client() as client:
        saved = client.post("/api/v1/users/user/settings/update", json={"ui": {"models": [first]}})
    assert saved.status_code == 200, saved.text
    page = page_for(admin)
    expect(_selected_model(page, first)).to_be_visible()

    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    page.get_by_role("button", name="Compare", exact=True).click()
    select_model(page, second)
    page.keyboard.press("Escape")
    expect(_selected_model(page, f"{first} +1")).to_be_visible()

    both = "which of you answers?"
    upstream.queue(
        reply.text("first preset here", match=_asked_as(both, "You are the first preset.")),
        reply.text("second preset here", match=_asked_as(both, "You are the second preset.")),
    )
    send(page, both)
    expect(replies(page)).to_have_count(2, timeout=REPLY_TIMEOUT_MS)
    expect(replies(page).nth(0)).to_contain_text("first preset here", timeout=REPLY_TIMEOUT_MS)
    expect(replies(page).nth(1)).to_contain_text("second preset here", timeout=REPLY_TIMEOUT_MS)

    page.get_by_role("button", name=SELECTOR_BUTTON).click()
    compare = page.get_by_role("button", name="Compare", exact=True)
    expect(compare).to_have_attribute("aria-pressed", "true")
    compare.click()
    expect(compare).to_have_attribute("aria-pressed", "false")
    page.keyboard.press("Escape")
    expect(_selected_model(page, first)).to_be_visible()

    alone = "and now just one?"
    upstream.queue(
        reply.text("only the first", match=_asked_as(alone, "You are the first preset."))
    )
    expect(chat_input(page)).to_be_visible()
    send(page, alone)
    expect(replies(page)).to_have_count(3, timeout=REPLY_TIMEOUT_MS)
    expect_reply(page, "only the first")
    assert len(_requests_for(upstream, alone, "You are the first preset.")) == 1
    assert _requests_for(upstream, alone, "You are the second preset.") == []


def _share_with_everyone(admin: Actor, *model_ids: str) -> None:
    with admin.client() as client:
        for model_id in model_ids:
            shared = client.post(
                "/api/v1/models/model/access/update",
                json={"id": model_id, "name": model_id, "access_grants": [EVERYONE_READS]},
            )
            assert shared.status_code == 200, shared.text


@pytest.mark.parametrize("allowed", [True, False], ids=["allowed", "withheld"])
def test_a_new_chat_keeps_both_saved_models_only_with_multiple_models(
    allowed, page_for, two_presets, make_user, upstream, preserve
):
    """#32668 (fix 5d75c1905): without Multiple Models a new chat still opened on both models."""
    admin, first, second = two_presets
    _share_with_everyone(admin, first, second)
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["chat"]["multiple_models"] = allowed
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
    assert saved.status_code == 200, saved.text
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"models": [first, second]}}
        )
    assert saved.status_code == 200, saved.text

    page = page_for(account)
    expect(_selected_model(page, f"{first} +1" if allowed else first)).to_be_visible()

    question = f"who answers here? {uuid.uuid4().hex[:6]}"
    upstream.queue(
        reply.text("first preset here", match=_asked_as(question, "You are the first preset.")),
        reply.text("second preset here", match=_asked_as(question, "You are the second preset.")),
    )
    send(page, question)
    expect(replies(page).nth(0)).to_contain_text("first preset here", timeout=REPLY_TIMEOUT_MS)
    expect(replies(page)).to_have_count(2 if allowed else 1, timeout=REPLY_TIMEOUT_MS)
    asked_second = _requests_for(upstream, question, "You are the second preset.")
    assert len(asked_second) == (1 if allowed else 0), (
        f"the second saved model was asked {len(asked_second)} times"
    )


def _seed_answered_by_both(account: Actor, first: str, second: str) -> str:
    """A saved chat whose one question both presets answered; returns its id."""
    question_id, first_id, second_id = (str(uuid.uuid4()) for _ in range(3))
    stored = {"done": True, "timestamp": 1_700_000_000}
    question = {
        **stored,
        "id": question_id,
        "parentId": None,
        "childrenIds": [first_id, second_id],
        "role": "user",
        "content": "which of you answers?",
        "models": [first, second],
    }
    answers = {
        answer_id: {
            **stored,
            "id": answer_id,
            "parentId": question_id,
            "childrenIds": [],
            "role": "assistant",
            "content": f"{model} here",
            "model": model,
            "modelIdx": index,
        }
        for index, (answer_id, model) in enumerate(((first_id, first), (second_id, second)))
    }
    chat = {
        "title": f"Both presets {uuid.uuid4().hex[:8]}",
        "models": [first, second],
        "history": {"currentId": second_id, "messages": {question_id: question, **answers}},
    }
    with account.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"]


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_saved_chat_two_models_answered_shows_both_answers(chat_for, two_presets):
    """In a side chat: open-webui/open-webui#34030."""
    admin, first, second = two_presets
    chat = chat_for(admin, chat_id=_seed_answered_by_both(admin, first, second))

    expect(conversation(chat).get_by_text(f"{first} here")).to_be_visible(timeout=REPLY_TIMEOUT_MS)
    expect(
        conversation(chat).get_by_text(f"{second} here"),
        f"the {chat.place} shows one model's answer only",
    ).to_be_visible()
    expect(replies(chat)).to_have_count(2)
