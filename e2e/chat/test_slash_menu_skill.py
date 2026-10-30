"""Regression: a skill picked from the / menu in the chat input was never applied to the reply.

Fix `c402acb40` (open-webui/open-webui#31429, issue open-webui/open-webui#29978): the / menu lists
skills next to the commands, but picking one inserted it as an @ mention, so the message carried
`<@id|name>` where the $ menu writes `<$id|name>`. The server only loads skills from $ mentions,
so the skill never reached the model and its raw tag did. A skill chosen through / is now sent the
same way as through $.

Fix 14956af03 (#32690, issue #32689): a skill-only draft brought back into the input after
leaving the chat inside the app was rebuilt without the skill's name, sent as `<$id>`, and the
server's fallback for an empty message filled in the names of every skill.

Discriminates: passes on the efe63bd34 build; on a build with c402acb40 reverted the / menu test
fails (no skill in the system prompt, the raw tag in the user message). The $ menu test passes on
both, and an unrelated chat journey passes on the reverted build. In the pre-fix build of 0.12.4
(14956af03 reverted) the restored draft test fails: the user message reads both skills' names.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from utils.chat_ui import chat_input, expect_reply

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

SKILL_BODY = "Answer every question as a haiku."


@pytest.fixture
def skill(make_user):
    """A fresh admin and an active skill of theirs; yields (account, skill name)."""
    account = make_user(role="admin")
    name = f"Haiku Helper {uuid.uuid4().hex[:6]}"
    skill_id = f"haiku_{uuid.uuid4().hex[:8]}"
    with account.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={"id": skill_id, "name": name, "content": SKILL_BODY, "meta": {}},
        )
        assert created.status_code == 200, created.text
        yield account, name
        client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def _pick_skill(page: Page, trigger: str, name: str) -> None:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type(f"{trigger}{name.split()[0]}")
    page.get_by_role("button", name=re.compile(re.escape(name))).click()


def _send_after_pick(page: Page, upstream, trigger: str, name: str) -> dict:
    question = f"what is rain? {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Soft grey threads fall", match=reply.answering(question)))
    _pick_skill(page, trigger, name)
    page.keyboard.type(question)
    page.keyboard.press("Enter")
    expect_reply(page, "Soft grey threads fall")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def _system_text(request: dict) -> str:
    return "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )


def test_a_skill_picked_from_the_slash_menu_reaches_the_model(page_for, skill, upstream):
    account, name = skill
    page = page_for(account)

    request = _send_after_pick(page, upstream, "/", name)

    user_text = str(request["messages"][-1]["content"])
    assert SKILL_BODY in _system_text(request), (
        "the skill picked from the / menu was never loaded for the reply (#29978)"
    )
    assert "<@" not in user_text, f"the skill went out as a raw @ mention: {user_text!r}"


def test_a_skill_picked_from_the_dollar_menu_reaches_the_model(page_for, skill, upstream):
    account, name = skill
    page = page_for(account)

    request = _send_after_pick(page, upstream, "$", name)

    assert SKILL_BODY in _system_text(request)


@pytest.fixture
def second_skill(skill):
    """Another active skill of the same account, so "every skill" is more than the one picked."""
    account, _ = skill
    skill_id = f"sonnet_{uuid.uuid4().hex[:8]}"
    with account.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={"id": skill_id, "name": f"Sonnet Smith {skill_id[-6:]}", "content": "Rhyme."},
        )
        assert created.status_code == 200, created.text
        yield
        client.delete(f"/api/v1/skills/id/{skill_id}/delete")


def test_a_restored_skill_only_draft_sends_only_that_skill(page_for, skill, second_skill, upstream):
    """#32690 (fix 14956af03): a skill-only draft brought back sent every skill's name.

    Red on dev since b29f7781a: the draft is not brought back at all
    (open-webui/open-webui#33397, fix PR open-webui/open-webui#33398)."""
    account, name = skill
    page = page_for(account)
    _pick_skill(page, "$", name)
    # the new chat keeps its draft in the tab's session storage, written shortly after typing
    page.wait_for_function("() => (sessionStorage.getItem('chat-input') || '').includes('<$')")

    # leaving and coming back inside the app; a reload brings the draft back as plain text
    page.get_by_role("link", name="Workspace", exact=True).click()
    expect(page).to_have_url(re.compile(r"/workspace"))
    page.get_by_role("link", name="New Chat").first.click()
    expect(chat_input(page)).to_contain_text(name)
    upstream.queue(reply.text("Soft grey threads fall", match=reply.answering(name)))
    chat_input(page).press("End")
    chat_input(page).press("Enter")
    expect_reply(page, "Soft grey threads fall")

    request = next(filter(reply.answering(name), upstream.chat_requests()))
    sent = request["messages"][-1]["content"]
    assert sent == name, f"the restored draft sent {sent!r} in place of the skill's name"
    assert SKILL_BODY in _system_text(request)
