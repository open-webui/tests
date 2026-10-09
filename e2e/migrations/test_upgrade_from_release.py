"""Journey: data written by earlier releases, as its people find it in the app after the upgrade.

The browser twin of integration/migrations/test_upgrade_from_release.py, on the same data sets
(`harness.upgraded_release`): the checkout boots on what each release wrote, with the environment
the release ran with, and each account signs in with its old password. Alice's chat opens on the
branch she was last on, with her file, the tool call and the other branch one click away, and the
reply the old server answered from the handbook opens the passage it cited. Her pinned chat, her
nested folders and her archived chat are where she left them, and she can continue an old chat,
whose earlier turns reach the model and whose new turn is still there after a reload. Bob opens
the note Alice shared, starts new chats on the default model he picked and asks the knowledge
base. Group and default permissions decide the workspace and the chat menu as before, prompts
insert their text by their command, the shared tool runs, the admin's filter is still on and the
channel keeps its message, reaction and thread. Alice's old skill opens in the editor with its
instructions and saves a new version on top of the one the skill history migration gave it,
and Bob, who reads it through the group, mentions it in a new chat and the model gets it.

Discriminates: passes on dev ebc6add67 for every data set; with the last migration of a copy of it
mangling one kind of data on the v0.10.2 SQLite set, each test fails on its kind: assistant replies
blanked (the branch, cited reply and continuation tests), the regenerated branch dropped (the
branch test), stored sources dropped (the citation test), pins, folder nesting, the chat's folder
or the archive flag cleared, the note emptied, user settings reset, group and default permissions
reset, prompt commands renamed, the knowledge base's grants dropped, the tool's source broken, the
filter switched off and the channel's reactions and thread links dropped. With the skill history
migration (`d6a8c3f912ab`) of a copy setting no `version_id`, the editor test fails on every set
(the save is refused as a conflict); with it writing `SKILL.md` empty, the editor opens the skill
without its instructions.
"""

from __future__ import annotations

import re
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.instance import resolve_backend, resolve_frontend_build
from harness.upgraded_release import Upgraded, data_set_params, upgraded_release
from utils.cached_chat import attach_from_menu, calling, turn_on_tool
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, replies, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.slow,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
]

RELEASE_ENVIRONMENT = {
    # the releases ran with it, so every account could chat with the provider's model
    "BYPASS_MODEL_ACCESS_CONTROL": "true",
    "CORS_ALLOW_ORIGIN": "*",
}


@pytest.fixture(scope="module", params=data_set_params())
def upgraded(request, tmp_path_factory) -> Iterator[Upgraded]:
    backend = resolve_backend()
    build = resolve_frontend_build(backend) if backend else None
    if build is None:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    name = request.param
    settings = {**RELEASE_ENVIRONMENT, "FRONTEND_BUILD_DIR": str(build)}
    with upgraded_release(name, tmp_path_factory.mktemp(name), settings) as release:
        yield release


@pytest.fixture
def provider(upgraded):
    upgraded.provider.reset()
    return upgraded.provider


@pytest.fixture
def open_as(page_for, upgraded):
    """`open_as(who, path)` opens `path` signed in as a manifest account."""

    def open_page(who: str, path: str = "/") -> Page:
        page = page_for(upgraded.actor(who))
        if path != "/":
            page.goto(path)
        return page

    return open_page


def _sidebar(page: Page) -> Locator:
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    return page.get_by_role("navigation", name="Chat history")


def _expand(sidebar: Locator, section: str) -> None:
    toggle = sidebar.get_by_role("button", name=section, exact=True)
    expect(toggle).to_be_visible()
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()


def _folder_row(sidebar: Locator, name: str) -> Locator:
    """The folder's row, named with its unread count when it has one."""
    return sidebar.get_by_role("button", name=re.compile(rf"^{re.escape(name)}( \d+)?$"))


def _open_folder(sidebar: Locator, name: str) -> None:
    _folder_row(sidebar, name).get_by_role("button").first.click()


def _folder_with_its_chats(sidebar: Locator, folder_id: str) -> Locator:
    """The folder's row and what is listed under it, which share no accessible name."""
    row = sidebar.locator(f"#folder-{folder_id}-button")
    return row.locator("xpath=ancestor::div[@draggable][1]")


def _chat_menu(sidebar: Locator, title: str) -> Locator:
    """The menu on the row of `title`, whose trigger shows while the row has focus."""
    row = sidebar.locator("#sidebar-chat-group").filter(has_text=title)
    row.hover()
    row.get_by_role("button", name=re.compile(re.escape(title))).first.focus()
    expect(row.get_by_role("button", name=title, exact=True)).to_be_visible()
    row.get_by_role("button", name="Chat Menu").first.click()
    return sidebar.page.get_by_role("menu")


def _open_settings_tab(page: Page, tab: str) -> Locator:
    page.get_by_role("navigation", name="Chat history").get_by_label("User menu").click()
    page.get_by_role("menu").get_by_role("button", name="Settings").click()
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name=tab).click()
    return settings


def _workspace_tabs(page: Page) -> Locator:
    return page.get_by_role("main").get_by_role("navigation")


def _command_suggestions(page: Page, typed: str) -> Locator:
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.press("ControlOrMeta+A")
    page.keyboard.press("Backspace")
    page.keyboard.type(f"/{typed}")
    return page.get_by_role("tooltip")


def _the_chat_asking(question: str):
    """A `match` for the streamed chat request, which the title and query tasks are not."""
    asking = reply.answering(question)
    return lambda body: bool(body.get("stream")) and asking(body)


def _after_a_tool_result(body: dict) -> bool:
    return body["messages"][-1]["role"] == "tool"


def _turn_with(page: Page, text: str) -> Locator:
    """The whole turn holding `text`, its branch arrows and result buttons included."""
    return conversation(page).get_by_role("listitem").filter(has_text=text)


def test_the_old_chat_opens_on_its_branch_with_the_file_and_the_tool_call(open_as, upgraded):
    branched = upgraded.manifest["chats"]["branched"]
    page = open_as("alice", f"/c/{branched['id']}")
    chat = conversation(page)

    expect(last_reply(page)).to_contain_text("A light jacket.")
    expect(chat.get_by_text("What should I pack for Vienna?")).to_be_visible()
    expect(chat.get_by_role("button", name=re.compile("packing.txt"))).to_be_visible()
    regenerated = _turn_with(page, "It is sunny, so pack sunglasses.")
    expect(regenerated.get_by_text("2/2")).to_be_visible()
    regenerated.get_by_role("button", name="View Result from get_weather").click()
    expect(regenerated.get_by_text("Sunny in Vienna").first).to_be_visible()

    regenerated.get_by_role("button", name="Previous message").click()

    expect(last_reply(page)).to_contain_text("Pack an umbrella.")
    expect(_turn_with(page, "Pack an umbrella.").get_by_text("1/2")).to_be_visible()
    expect(chat.get_by_text("And for the evening?")).to_have_count(0)


def test_the_cited_reply_opens_the_passage_it_was_answered_from(open_as, upgraded):
    cited, knowledge = upgraded.manifest["chats"]["cited"], upgraded.manifest["knowledge"]
    page = open_as("alice", f"/c/{cited['id']}")

    expect(last_reply(page)).to_contain_text("Behind the blue door")
    expect(conversation(page).get_by_role("button", name="Toggle 1 source")).to_be_visible()
    marker = f"View source: {knowledge['filename']}"
    last_reply(page).get_by_role("button", name=marker).click()

    citation = page.get_by_role("dialog")
    expect(citation).to_contain_text(knowledge["text"])


def test_the_pinned_chat_sits_in_its_nested_folder_once_unpinned(open_as, upgraded):
    folders, branched = upgraded.manifest["folders"], upgraded.manifest["chats"]["branched"]
    sidebar = _sidebar(open_as("alice"))
    expect(sidebar.get_by_role("button", name="Pinned", exact=True)).to_be_visible()
    expect(sidebar.get_by_role("button", name=re.compile(branched["title"]))).to_be_visible()
    _expand(sidebar, "Folders")
    child_folder = _folder_row(sidebar, folders["child"]["name"])
    expect(_folder_row(sidebar, folders["parent"]["name"])).to_be_visible()
    expect(child_folder).to_have_count(0)
    _open_folder(sidebar, folders["parent"]["name"])
    expect(child_folder).to_be_visible()

    _chat_menu(sidebar, branched["title"]).get_by_role("button", name="Unpin").click()

    expect(sidebar.get_by_role("button", name="Pinned", exact=True)).to_have_count(0)
    _open_folder(sidebar, folders["child"]["name"])
    in_child_folder = _folder_with_its_chats(sidebar, folders["child"]["id"])
    expect(
        in_child_folder.get_by_role("button", name=re.compile(branched["title"]))
    ).to_be_visible()


def test_the_archived_chat_is_in_the_archive_and_not_the_sidebar(open_as, upgraded):
    archived = upgraded.manifest["chats"]["archived"]
    page = open_as("alice")
    sidebar = _sidebar(page)
    live_title = upgraded.manifest["chats"]["live"]["title"]
    expect(sidebar.get_by_role("button", name=re.compile(live_title))).to_be_visible()
    expect(sidebar.get_by_role("button", name=re.compile(archived["title"]))).to_have_count(0)

    settings = _open_settings_tab(page, "Archived Chats")

    expect(settings.get_by_text(archived["title"])).to_be_visible()


def test_an_old_chat_continues_and_keeps_the_new_turn(open_as, upgraded, provider):
    live = upgraded.manifest["chats"]["live"]
    page = open_as("alice", f"/c/{live['id']}")
    expect(last_reply(page)).to_contain_text("Try it with lemon.")
    question = "Is it from Vienna?"
    provider.queue(reply.text("Yes, the Wiener Schnitzel is.", match=_the_chat_asking(question)))

    send(page, question)

    expect_reply(page, "Yes, the Wiener Schnitzel is.")
    [request] = [body for body in provider.chat_requests() if _the_chat_asking(question)(body)]
    sent = [(entry["role"], entry["content"]) for entry in request["messages"]]
    assert sent[-5:] == [
        ("user", "What is a Schnitzel?"),
        ("assistant", "Schnitzel is a breaded cutlet."),
        ("user", "How do I eat it?"),
        ("assistant", "Try it with lemon."),
        ("user", question),
    ], sent
    page.reload()
    expect(last_reply(page)).to_contain_text("Yes, the Wiener Schnitzel is.")
    expect(replies(page)).to_have_count(3)
    expect(conversation(page).get_by_text("How do I eat it?")).to_be_visible()


def test_the_shared_note_opens_with_its_text_for_the_account_it_was_shared_with(open_as, upgraded):
    note = upgraded.manifest["note"]
    page = open_as("bob", "/notes")
    listed = page.get_by_role("main").get_by_role("button", name="Open note")

    listed.filter(has_text=note["title"]).click()

    expect(page.get_by_role("main").get_by_text("dark chocolate")).to_be_visible()
    expect(page.get_by_role("main").get_by_text("apples")).to_be_visible()


def test_new_chats_open_on_the_default_model_picked_before_the_upgrade(open_as, upgraded):
    model = upgraded.manifest["model"]
    page = open_as("bob")

    selected = page.get_by_role("button", name=f"Selected model: {model['name']}")

    expect(selected).to_be_visible()


def test_an_interface_setting_saved_before_the_upgrade_is_still_off(open_as):
    settings = _open_settings_tab(open_as("alice"), "Interface")

    switch = settings.get_by_role("switch", name="Chat Bubble UI")

    expect(switch).to_have_attribute("aria-checked", "false")


def test_group_and_default_permissions_shape_the_workspace_and_chat_menu(open_as, upgraded):
    member = open_as("alice", "/workspace/prompts")
    expect(_workspace_tabs(member).get_by_role("link", name=re.compile("^Prompts"))).to_be_visible()
    expect(_workspace_tabs(member).get_by_role("link", name=re.compile("^Models"))).to_be_visible()
    outsider = open_as("carol", "/workspace/prompts")
    expect(
        _workspace_tabs(outsider).get_by_role("link", name=re.compile("^Prompts"))
    ).to_be_visible()
    expect(_workspace_tabs(outsider).get_by_role("link", name=re.compile("^Models"))).to_have_count(
        0
    )

    live_title = upgraded.manifest["chats"]["live"]["title"]
    menu = _chat_menu(_sidebar(member), live_title)

    expect(menu.get_by_role("button", name="Archive")).to_be_visible()
    expect(menu.get_by_role("button", name="Delete")).to_have_count(0)


def test_prompts_insert_their_text_by_their_command_for_their_readers(open_as, upgraded):
    prompts = upgraded.manifest["prompts"]
    page = open_as("alice")
    suggestions = _command_suggestions(page, "")
    for command, prompt in prompts.items():
        expect(
            suggestions.get_by_role("button", name=f"{command} {prompt['name']}", exact=True)
        ).to_be_visible()

    _command_suggestions(page, "summ").get_by_role("button", name=re.compile("^summarize")).click()

    expect(chat_input(page)).to_have_text(prompts["summarize"]["content"])
    _command_suggestions(page, "alice").get_by_role(
        "button", name=re.compile("^alice-draft")
    ).click()
    expect(chat_input(page)).to_have_text(prompts["alice-draft"]["content"])

    outsider = _command_suggestions(open_as("carol"), "")
    expect(outsider.get_by_role("button", name=re.compile("^Settings"))).to_be_visible()
    expect(outsider.get_by_role("button", name=re.compile("^summarize"))).to_have_count(0)


def test_the_knowledge_base_answers_a_new_chat_from_its_old_file(open_as, upgraded, provider):
    knowledge = upgraded.manifest["knowledge"]
    page = open_as("bob")
    attach_from_menu(page, "Knowledge", knowledge["name"])
    question = "Which door hides the key?"
    provider.queue(reply.text("The blue one.", match=_the_chat_asking(question)))

    send(page, question)

    expect_reply(page, "The blue one.")
    [request] = [body for body in provider.chat_requests() if _the_chat_asking(question)(body)]
    sent = "\n".join(str(entry["content"]) for entry in request["messages"])
    assert knowledge["text"] in sent, "the old file's text did not reach the model"


def test_the_shared_tool_runs_in_a_new_chat(open_as, upgraded, provider):
    tool = upgraded.manifest["tool"]
    page = open_as("alice")
    turn_on_tool(page, tool["name"])
    question = "What is the weather in Graz?"
    provider.queue(
        calling("get_weather", {"city": "Graz"}, match=_the_chat_asking(question)),
        reply.text("Pack sunglasses for Graz.", match=_after_a_tool_result),
    )

    send(page, question)

    expect_reply(page, "Pack sunglasses for Graz.")
    [answered] = [body for body in provider.chat_requests() if _after_a_tool_result(body)]
    results = [entry["content"] for entry in answered["messages"] if entry["role"] == "tool"]
    assert any("Sunny in Graz" in str(result) for result in results), results


def test_the_filter_function_is_still_on(open_as, upgraded):
    function = upgraded.manifest["function"]
    page = open_as("admin", "/admin/functions")

    row = page.get_by_role("main").get_by_role("button", name=re.compile(function["name"]))

    expect(row.get_by_role("switch")).to_be_checked()


def test_the_channel_keeps_its_message_reaction_and_thread(open_as, upgraded):
    channel = upgraded.manifest["channel"]
    page = open_as("alice")
    sidebar = _sidebar(page)
    _expand(sidebar, "Channels")
    sidebar.get_by_role("link", name=channel["name"]).click()
    main = page.get_by_role("main")

    expect(main.get_by_text(channel["message"]["content"])).to_be_visible()
    expect(main.get_by_role("button", name=f"{channel['reaction']} 1")).to_be_visible()
    main.get_by_role("button", name=re.compile("^1 Replies")).click()

    expect(page.get_by_text(channel["reply"]["content"])).to_be_visible()


def test_an_old_skill_opens_in_the_editor_and_saves_a_new_version(open_as, upgraded):
    skill = upgraded.manifest["skills"]["trip-planner"]
    page = open_as("alice", "/workspace/skills/edit?id=trip-planner")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Skill Name")).to_have_value(skill["name"])
    expect(editor.get_by_text("Prefer night trains over flights.")).to_be_visible()

    editor.get_by_placeholder("Skill Description").fill("Plans trips by train and ferry.")
    editor.get_by_role("textbox", name="Commit message").fill("Mention ferries")
    editor.get_by_role("button", name="Save", exact=True).click()

    expect(page.get_by_text("Skill updated successfully")).to_be_visible()
    saved = upgraded.get("alice", "/api/v1/skills/id/trip-planner").json()
    assert (saved["description"], saved["content"]) == (
        "Plans trips by train and ferry.",
        skill["content"],
    )
    history = upgraded.get("alice", "/api/v1/skills/id/trip-planner/history").json()
    assert [entry["commit_message"] for entry in history] == ["Mention ferries", None]


def test_a_group_reader_mentions_an_old_skill_in_a_new_chat(open_as, upgraded, provider):
    skill = upgraded.manifest["skills"]["trip-planner"]
    page = open_as("bob")
    expect(chat_input(page)).to_be_visible()
    chat_input(page).click()
    page.keyboard.type("$Trip")
    page.get_by_role("button").filter(has_text=skill["name"]).click()
    question = "Which trains for Ljubljana?"
    provider.queue(reply.text("The night train.", match=_the_chat_asking(question)))

    page.keyboard.type(question)
    page.keyboard.press("Enter")

    expect_reply(page, "The night train.")
    [request] = [body for body in provider.chat_requests() if _the_chat_asking(question)(body)]
    system = "\n".join(
        str(entry["content"]) for entry in request["messages"] if entry["role"] == "system"
    )
    assert "Prefer night trains over flights." in system, system
