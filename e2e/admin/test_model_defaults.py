"""Journey: the Model Defaults an admin sets in Admin Settings > Models reach every model's chats.

Model Defaults holds what a model without settings of its own starts from. A prompt suggestion
added there shows on a user's new chat and sends its prompt when pressed; a parameter set there
reaches the provider with the user's chat; a capability switched off there takes that ability
from the model, so an image attached to a chat with it is refused where it attached before.
A tool picked under Tools there (#33749) stays picked after a reload and is on in a user's new
chat, so the provider is offered it; a model whose own tool list is empty does not take it, a
default tool the user may not use is left out of their chat while one they may use is kept, and
a new model in the workspace editor starts with the default tools ticked (#33787).

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose Model Defaults save sends the
stored defaults back in place of the edited ones, every test but the untouched attach fails.
The prompt suggestion and capability tests were retargeted for 8d0ff76f2, whose defaults use
the model editor's prompt rows and Capabilities switches: they pass on dev 76ad6f97c (3 of 3)
and fail in a build of it whose save drops the edited capabilities and prompt suggestions.
The default tool tests pass on dev f659c0d12; in a build of it whose Model Defaults save
leaves out the tools the picked tool test fails, and in one whose new model form starts with no
tools the new model test fails. In a build whose rows show no picked count and whose new chat
picks every default tool, the test of the default tool the user may not use fails on the count.
The Skills test marks a gap with a strict xfail: Model Defaults has no default skills.
"""

from __future__ import annotations

import contextlib
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_picked, expect_reply, send
from utils.model_editor import pick, section, selector_row
from utils.setting_rows import setting_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MODELS_CONFIG = ("/api/v1/configs/models", "/api/v1/configs/models")
# a PNG of one pixel, enough for the chat input to treat the file as an image
PIXEL_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
    "1f15c4890000000d4944415478da63f8ffff3f0005fe02fea7d6a4b40000000049454e44ae426082"
)


@pytest.fixture
def defaults_restored(admin, preserve):
    """Puts the model defaults and the default prompt suggestions back afterwards."""
    preserve(MODELS_CONFIG)
    with admin.client() as client:
        shown = client.get("/api/config").json()
    suggestions = shown.get("default_prompt_suggestions")
    localized = shown.get("default_prompt_suggestions_i18n") or {}
    yield
    with admin.client() as client:
        restored = client.post(
            "/api/v1/configs/suggestions", json={"suggestions": suggestions, "i18n": localized}
        )
    restored.raise_for_status()


def open_model_defaults(page: Page) -> Locator:
    page.goto("/admin/settings/models")
    settings = page.get_by_role("dialog")
    settings.get_by_role("button", name="Model Defaults").click()
    expect(settings.get_by_role("button", name="Model Capabilities")).to_be_visible()
    return settings


def open_section(settings: Locator, name: str) -> None:
    settings.get_by_role("button", name=name).click()


def save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Models configuration saved successfully")).to_be_visible()


def is_file_upload(response) -> bool:
    return response.request.method == "POST" and "/api/v1/files/" in response.url


def test_a_default_prompt_suggestion_shows_on_a_new_chat_and_sends(
    page_for, admin, make_user, upstream, defaults_restored
):
    title = f"Tide times {uuid.uuid4().hex[:6]}"
    content = f"When is high tide in the harbour today? {uuid.uuid4().hex[:6]}"
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Prompt Suggestions")
    settings.get_by_role("button", name="Add prompt suggestion").click()
    settings.get_by_role("textbox", name="Prompt", exact=True).last.fill(content)
    settings.get_by_text("Display text").last.click()
    settings.get_by_role("textbox", name="Title", exact=True).last.fill(title)
    save(admin_page, settings)

    page = page_for(make_user())
    upstream.queue(reply.text("At noon.", match=reply.answering(content)))
    page.get_by_role("listitem").filter(has_text=title).click()

    expect_reply(page, "At noon.")


def test_a_default_parameter_reaches_the_provider(
    page_for, admin, make_user, upstream, defaults_restored
):
    question = "How warm is the lake today?"
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Model Parameters")
    setting_button(settings, "Temperature").click()
    settings.get_by_role("spinbutton", name="Temperature").fill("0.3")
    save(admin_page, settings)

    page = page_for(make_user())
    upstream.queue(reply.text("Nineteen degrees.", match=reply.answering(question)))
    send(page, question)
    expect_reply(page, "Nineteen degrees.")

    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    assert sent["temperature"] == 0.3


def test_a_default_capability_switched_off_refuses_an_image(
    page_for, admin, make_user, defaults_restored
):
    admin_page = page_for(admin)
    settings = open_model_defaults(admin_page)
    open_section(settings, "Model Capabilities")
    vision = section(settings, "Capabilities").get_by_role("switch", name="Vision")
    expect(vision).to_have_attribute("aria-checked", "true")
    vision.click()
    expect(vision).to_have_attribute("aria-checked", "false")
    save(admin_page, settings)

    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    chooser.value.set_files({"name": "pier.png", "mimeType": "image/png", "buffer": PIXEL_PNG})

    expect(page.get_by_text("Selected model(s) do not support image inputs")).to_be_visible()
    expect(page.get_by_role("button", name="pier.png")).to_have_count(0)


def test_an_image_attaches_while_the_default_capabilities_allow_it(page_for, make_user):
    page = page_for(make_user())
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("main").get_by_label("More").click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload Files").click()
    with page.expect_response(is_file_upload) as uploaded:
        chooser.value.set_files({"name": "pier.png", "mimeType": "image/png", "buffer": PIXEL_PNG})
    file_id = uploaded.value.json()["id"]

    # the name shows only while it uploads, then the stored image shows as its thumbnail
    thumbnail = page.get_by_role("button", name="Show image preview").locator(
        f"img[src*='{file_id}']"
    )
    expect(thumbnail).to_be_visible()


# --------------------------------------------------------------------------- default tools

TIDE_TOOL = '''class Tools:
    def tide_times(self, harbour: str) -> str:
        """Tell the tide times of a harbour.

        :param harbour: Name of the harbour
        """
        return "high water at noon"
'''

LOG_TOOL = '''class Tools:
    def read_harbour_log(self) -> str:
        """Read the harbour master's log."""
        return "all quiet"
'''


@contextlib.contextmanager
def workspace_tool(admin, name: str, grants: list[dict], source: str = TIDE_TOOL) -> Iterator[str]:
    """A tool of the admin's named `name`, readable as `grants` say; deleted afterwards."""
    tool_id = f"tide_{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/tools/create",
            json={
                "id": tool_id,
                "name": name,
                "content": source,
                "meta": {"description": "Reads the harbour tides"},
                "access_grants": grants,
            },
        )
        assert created.status_code == 200, created.text
        try:
            yield tool_id
        finally:
            client.delete(f"/api/v1/tools/id/{tool_id}/delete")


def set_default_tools(admin, tool_ids: list[str]) -> None:
    with admin.client() as client:
        current = client.get(MODELS_CONFIG[0]).json()
        metadata = {**(current.get("DEFAULT_MODEL_METADATA") or {}), "toolIds": tool_ids}
        saved = client.post(MODELS_CONFIG[1], json={**current, "DEFAULT_MODEL_METADATA": metadata})
    assert saved.status_code == 200, saved.text


def offered_tools(upstream, question: str) -> set[str]:
    sent = next(filter(reply.answering(question), upstream.chat_requests()))
    return {tool["function"]["name"] for tool in sent.get("tools") or []}


def test_a_default_tool_picked_in_model_defaults_is_on_in_a_users_new_chat(
    page_for, admin, make_user, upstream, defaults_restored
):
    name = f"Tide desk {uuid.uuid4().hex[:6]}"
    with workspace_tool(admin, name, [EVERYONE_READS]):
        admin_page = page_for(admin)
        settings = open_model_defaults(admin_page)
        pick(settings, "Tools", name)
        save(admin_page, settings)

        admin_page.reload()
        settings = open_model_defaults(admin_page)
        expect(selector_row(settings, "Tools")).to_contain_text(name)

        page = page_for(make_user())
        expect_picked(page, "Tools", 1)
        question = f"When is high water? {uuid.uuid4().hex[:6]}"
        upstream.queue(reply.text("At noon.", match=reply.answering(question)))
        send(page, question)
        expect_reply(page, "At noon.")

    assert "tide_times" in offered_tools(upstream, question)


def test_a_model_with_its_own_tool_list_does_not_take_the_default_tools(
    page_for, admin, make_user, upstream, defaults_restored
):
    account = make_user()
    model_id = f"no-tools-{uuid.uuid4().hex[:8]}"
    with workspace_tool(admin, f"Tide desk {uuid.uuid4().hex[:6]}", [EVERYONE_READS]) as tool_id:
        set_default_tools(admin, [tool_id])
        with admin.client() as client:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": model_id,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": model_id,
                    # the admin unticked every tool for this model
                    "meta": {"toolIds": []},
                    "params": {},
                    "access_grants": [EVERYONE_READS],
                },
            )
            assert created.status_code == 200, created.text
        try:
            page = page_for(account, path=f"/?model={model_id}")
            expect(page.get_by_role("button", name=f"Selected model: {model_id}")).to_be_visible()
            question = f"Any tides today? {uuid.uuid4().hex[:6]}"
            upstream.queue(reply.text("No idea.", match=reply.answering(question)))
            send(page, question)
            expect_reply(page, "No idea.")
        finally:
            with admin.client() as client:
                client.post("/api/v1/models/model/delete", json={"id": model_id})

    assert "tide_times" not in offered_tools(upstream, question)


def test_a_default_tool_a_user_cannot_use_is_left_out_of_their_chat(
    page_for, admin, make_user, upstream, defaults_restored
):
    with (
        workspace_tool(admin, f"Log desk {uuid.uuid4().hex[:6]}", [], LOG_TOOL) as private_id,
        workspace_tool(admin, f"Tide desk {uuid.uuid4().hex[:6]}", [EVERYONE_READS]) as shared_id,
    ):
        set_default_tools(admin, [private_id, shared_id])
        page = page_for(make_user())
        # only the tool this account may use is picked
        expect_picked(page, "Tools", 1)
        question = f"Who keeps the harbour log? {uuid.uuid4().hex[:6]}"
        upstream.queue(reply.text("The master.", match=reply.answering(question)))
        send(page, question)
        expect_reply(page, "The master.")

    offered = offered_tools(upstream, question)
    assert "tide_times" in offered, f"the shared default tool was not offered: {sorted(offered)}"
    assert "read_harbour_log" not in offered, "the default tool the user may not use was offered"


def test_a_new_model_starts_with_the_default_tools_ticked(page_for, admin, defaults_restored):
    name = f"Tide desk {uuid.uuid4().hex[:6]}"
    with workspace_tool(admin, name, [EVERYONE_READS]) as tool_id:
        set_default_tools(admin, [tool_id])
        page = page_for(admin, path="/workspace/models/create")
        editor = page.get_by_role("main")
        expect(selector_row(editor, "Tools")).to_contain_text(name)


@pytest.mark.xfail(
    strict=True,
    reason="gap: Model Defaults sets default tools for every model but no default skills, "
    "which the model editor offers right beside its tools",
)
def test_model_defaults_offers_default_skills_beside_the_tools(page_for, admin):
    settings = open_model_defaults(page_for(admin))
    expect(selector_row(settings, "Tools")).to_be_visible()

    expect(selector_row(settings, "Skills")).to_be_visible()
