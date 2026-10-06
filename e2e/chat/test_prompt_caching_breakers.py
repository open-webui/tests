"""Journey: each breaker the Prompt Caching page names rewrites the prefix of a browser chat.

The page lists what breaks the provider's cached prefix and what does not. A person drives chats
on the page's setup in the web client and applies exactly one breaker per test: a model with File
Context on and a text file attached, the same file switched to Using Entire Document, Citations
on with a knowledge file read by a tool, a memory added in Settings > Personalization while the
memory system context is on, an AGENTS.md written into the Open Terminal's home between turns, a
skill saved there between turns, a builtin tool category unticked on the model by an admin in the
model editor, web search or a workspace tool switched on from the Integrations menu, an earlier
question edited and sent again, and a skill mentioned with `$` in a later turn, which is how a
chat attaches one (its whole text joins the system message). The page also calls some changes
harmless, and those tests stay green: a file or a whole knowledge base switched to Using Entire
Document with File Context off, memories in the system context that do not change, the first file
attached in a chat adding the Files tools once with every request after it appending again, and
an AGENTS.md or a skill that was in the terminal from the first turn. Each control checks where
the break falls.

Three more controls show breakers the page does not name, each rewriting the system message:
a chat moved into a folder that has its own system prompt, a knowledge base attached to the chat
after its first turn (its `<attached_knowledge>` list is in the system message, not in the
message that carried it) and a skill shared with the person mid-chat (every readable skill is
listed in the system message, whether or not it was attached).

Discriminates: passes on dev 30f3f6a8f. In a backend copy with a clock value added to the
model's system prompt every positive test failed, and so did the controls whose break falls after
the system message, since the clock moved the first break into it; the other controls passed.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.knowledge_bases import knowledge_base
from harness.prompt_caching import (
    ADMIN_CONFIG,
    assert_append_only,
    cache_optimal_model,
    first_break,
    turn_off_memory_system_context,
)
from harness.python_tools import EVERYONE_READS, python_tool
from harness.terminal_server import TERMINAL_SERVERS_CONFIG, configure_terminals, read_grant
from harness.web_retrieval import RETRIEVAL_CONFIG, save_web_settings
from utils.cached_chat import (
    ask,
    attach,
    attach_from_menu,
    calling,
    chat_requests,
    offered_tools,
    pick_terminal,
    tool_results,
    turn_on_tool,
)
from utils.chat_ui import chat_input, conversation, expect_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")
SKILL_NAME = "tide-tables"
SKILL = (
    f"---\nname: {SKILL_NAME}\ndescription: Reading tide tables\n---\n"
    "Read the high tide column first.\n"
)
TIDE_TOOL = """
class Tools:
    def tide_time(self, harbour: str) -> str:
        \"\"\"
        The next high tide in a harbour.
        :param harbour: The harbour's name
        \"\"\"
        return f"High tide in {harbour} is at noon."
"""
AGENTS_MD = "Always greet the harbour master by name.\n"
FIRST_MEMORY = "lives by the harbour"
SECOND_MEMORY = "takes the early ferry"


@pytest.fixture
def model_with(admin, preserve):
    """`model_with(**capabilities)` is the page's model with those capabilities changed."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with contextlib.ExitStack() as models:
        yield lambda **capabilities: models.enter_context(
            cache_optimal_model(admin, **capabilities)
        )


@pytest.fixture
def cached_setup(model_with):
    return model_with()


@pytest.fixture
def terminal_chat(page_for, cached_setup, admin, make_user, preserve, open_terminal):
    """A page on the cached model with the Open Terminal picked for the chat; the page's first
    message is still to come."""
    preserve(TERMINAL_SERVERS_CONFIG)
    person = make_user()
    connection = open_terminal.connection(config={"access_grants": [read_grant(person.id)]})
    with admin.client() as client:
        configure_terminals(client, connection)
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")
    pick_terminal(page, connection["name"])
    return page


@pytest.fixture
def agents_md(open_terminal):
    path = open_terminal.home / "AGENTS.md"
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture
def skill_dir(open_terminal):
    path = open_terminal.home / ".agents" / "skills" / SKILL_NAME
    yield path
    shutil.rmtree(path, ignore_errors=True)


def save_skill(directory) -> None:
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(SKILL)


# --- File Context and Using Entire Document -----------------------------------------------


def test_file_context_on_rewrites_the_prefix(model_with, page_for, make_user, upstream):
    page = page_for(make_user())
    page.goto(f"/?models={model_with(file_context=True).id}")

    attach(page, *NOTES)
    ask(page, upstream, "which berth is free?", reply.text("Berth 5 is free."))
    ask(page, upstream, "and for the pilot boat?", reply.text("Berth 3."))

    broken = first_break(chat_requests(upstream))
    assert broken is not None, "File Context on left the prefix append-only"
    assert "messages[1] (user)" in broken, broken


def use_entire_document(page: Page, name: str) -> None:
    page.get_by_role("button", name=re.compile(re.escape(name))).click()
    dialog = page.get_by_role("dialog")
    expect(dialog.get_by_text("Using Focused Retrieval")).to_be_visible()
    dialog.get_by_role("switch").click()
    expect(dialog.get_by_text("Using Entire Document")).to_be_visible()
    page.keyboard.press("Escape")
    expect(dialog).to_have_count(0)


def test_using_the_entire_document_with_file_context_on_rewrites_the_prefix(
    model_with, page_for, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={model_with(file_context=True).id}")

    attach(page, *NOTES)
    use_entire_document(page, NOTES[0])
    ask(page, upstream, "which berth is free?", reply.text("Berth 5 is free."))
    ask(page, upstream, "and for the pilot boat?", reply.text("Berth 3."))

    requests = chat_requests(upstream)
    assert "Berth 3 is reserved for the pilot boat." in str(requests[0]["messages"])
    broken = first_break(requests)
    assert broken is not None, "Using Entire Document with File Context on kept the prefix"
    assert "messages[1] (user)" in broken, broken


def test_using_the_entire_document_with_file_context_off_does_nothing(
    cached_setup, page_for, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    attach(page, *NOTES)
    use_entire_document(page, NOTES[0])
    ask(page, upstream, "which berth is free?", reply.text("Berth 5 is free."))
    ask(page, upstream, "and for the pilot boat?", reply.text("Berth 3."))

    requests = chat_requests(upstream)
    assert "Berth 3 is reserved for the pilot boat." not in str(requests[0]["messages"])
    assert_append_only(requests)


def test_a_whole_knowledge_base_with_file_context_off_does_nothing(
    cached_setup, page_for, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    attach_from_menu(page, "Attach Knowledge", "Harbour handbook")
    use_entire_document(page, "Harbour handbook collection")
    ask(page, upstream, "when does the ferry leave?", reply.text("At 06:40."))
    ask(page, upstream, "and from which pier?", reply.text("Pier 7."))

    requests = chat_requests(upstream)
    assert "The ferry leaves pier 7" not in str(requests[0]["messages"])
    assert_append_only(requests)


# --- Citations ------------------------------------------------------------------------------


def test_citations_on_rewrite_the_prefix(model_with, page_for, make_user, upstream):
    model = model_with(citations=True)
    page = page_for(make_user())
    page.goto(f"/?models={model.id}")

    ask(
        page,
        upstream,
        "read the handbook",
        calling("view_knowledge_file", {"file_id": model.handbook_file_id}),
        reply.text("The ferry leaves at 06:40."),
    )

    requests = chat_requests(upstream)
    assert "06:40" in tool_results(requests[-1]), tool_results(requests[-1])
    broken = first_break(requests)
    assert broken is not None, "Citations on left the prefix append-only after a tool round"
    assert "messages[1] (user)" in broken, broken


# --- Memory ---------------------------------------------------------------------------------


def add_memory_in_settings(page: Page, text: str) -> None:
    page.goto("/?settings=personalization")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Personalization")).to_be_visible()
    settings.get_by_role("button", name="Actions").first.click()
    page.get_by_role("menu").get_by_role("button", name="Add Memory").click()
    adding = page.get_by_role("dialog").filter(has_text="Add Memory")
    adding.get_by_role("textbox", name="Add a preference, fact, or instruction about you").fill(
        text
    )
    adding.get_by_role("button", name="Add", exact=True).click()
    expect(settings.get_by_text(text)).to_be_visible()


def test_a_memory_added_mid_chat_with_the_system_context_on_rewrites_the_prefix(
    cached_setup, admin, page_for, make_user, upstream
):
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, "ENABLE_MEMORY_SYSTEM_CONTEXT": True})
    assert saved.status_code == 200, saved.text
    person = make_user()
    with person.client() as client:
        client.post("/api/v1/memories/add", json={"content": FIRST_MEMORY}).raise_for_status()
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    ask(page, upstream, "good morning", reply.text("Morning."))
    add_memory_in_settings(page_for(person), SECOND_MEMORY)
    ask(page, upstream, "what is the weather?", reply.text("Clear."))

    requests = chat_requests(upstream)
    assert FIRST_MEMORY in requests[0]["messages"][0]["content"]
    assert SECOND_MEMORY in requests[1]["messages"][0]["content"]
    broken = first_break(requests)
    assert broken is not None, "a memory added mid-chat left the prefix append-only"
    assert "messages[0] (system)" in broken, broken


def test_unchanged_memories_with_the_system_context_on_only_append(
    cached_setup, admin, page_for, make_user, upstream
):
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, "ENABLE_MEMORY_SYSTEM_CONTEXT": True})
    assert saved.status_code == 200, saved.text
    person = make_user()
    with person.client() as client:
        for memory in (FIRST_MEMORY, SECOND_MEMORY):
            client.post("/api/v1/memories/add", json={"content": memory}).raise_for_status()
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(page, upstream, "what is the weather?", reply.text("Clear."))
    page.reload()
    ask(page, upstream, "and the tides?", reply.text("High at noon."))

    requests = chat_requests(upstream)
    assert SECOND_MEMORY in requests[0]["messages"][0]["content"]
    assert_append_only(requests)


# --- The first file attached to a chat ------------------------------------------------------


def test_the_first_file_attached_adds_the_file_tools_once(
    cached_setup, page_for, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    ask(page, upstream, "good morning", reply.text("Morning."))
    attach(page, *NOTES)
    ask(page, upstream, "which berth is free?", reply.text("Berth 5 is free."))
    ask(page, upstream, "and for the pilot boat?", reply.text("Berth 3."))
    page.reload()
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "list_chat_files" not in offered_tools(requests[0])
    assert "list_chat_files" in offered_tools(requests[1])
    broken = first_break(requests)
    assert broken is not None and "in tools" in broken, broken
    assert first_break(requests[:1]) is None
    assert_append_only(requests[1:])


# --- Open Terminal AGENTS.md and skills -----------------------------------------------------


def test_an_agents_file_there_from_the_first_turn_only_appends(terminal_chat, agents_md, upstream):
    page = terminal_chat
    agents_md.write_text(AGENTS_MD)
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(page, upstream, "what are my instructions?", reply.text("Greet the harbour master."))
    page.reload()
    ask(page, upstream, "anything else?", reply.text("Nothing else."))
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert AGENTS_MD.strip() in str(requests[0]["messages"])
    assert_append_only(requests)


def test_an_agents_file_written_between_turns_rewrites_the_prefix(
    terminal_chat, agents_md, upstream
):
    page = terminal_chat
    ask(page, upstream, "good morning", reply.text("Morning."))
    agents_md.write_text(AGENTS_MD)
    ask(page, upstream, "what are my instructions?", reply.text("Greet the harbour master."))

    requests = chat_requests(upstream)
    assert AGENTS_MD.strip() in str(requests[1]["messages"])
    broken = first_break(requests)
    assert broken is not None, "an AGENTS.md written between turns left the prefix append-only"
    assert "messages[1] (user)" in broken, broken


def test_a_skill_there_from_the_first_turn_only_appends(terminal_chat, skill_dir, upstream):
    page = terminal_chat
    save_skill(skill_dir)
    ask(page, upstream, "good morning", reply.text("Morning."))
    ask(
        page,
        upstream,
        "when is high tide?",
        calling("view_skill", {"id": f"terminal:{SKILL_NAME}"}),
        reply.text("Read the high tide column."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert "view_skill" in offered_tools(requests[0])
    assert "Read the high tide column first." in tool_results(requests[-1])
    assert_append_only(requests)


def test_a_skill_saved_between_turns_rewrites_the_prefix(terminal_chat, skill_dir, upstream):
    page = terminal_chat
    ask(page, upstream, "good morning", reply.text("Morning."))
    save_skill(skill_dir)
    ask(page, upstream, "when is high tide?", reply.text("At noon."))

    requests = chat_requests(upstream)
    assert "view_skill" not in offered_tools(requests[0])
    assert "view_skill" in offered_tools(requests[1])
    broken = first_break(requests)
    assert broken is not None and "in tools" in broken, broken


# --- The model's tools and the Integrations menu --------------------------------------------


def untick_builtin_tool(page: Page, model_id: str, label: str) -> None:
    page.goto(f"/workspace/models/edit?id={model_id}")
    editor = page.get_by_role("main")
    checkbox = editor.get_by_role("checkbox", name=label, exact=True)
    expect(checkbox).to_have_attribute("aria-checked", "true")
    checkbox.click()
    expect(checkbox).to_have_attribute("aria-checked", "false")
    editor.get_by_role("button", name="Save & Update").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))


def test_unticking_a_builtin_tool_category_mid_chat_rewrites_the_prefix(
    cached_setup, page_for, admin, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))

    untick_builtin_tool(page_for(admin), cached_setup.id, "Notes")
    page.reload()
    ask(page, upstream, "anything new?", reply.text("Nothing new."))

    requests = chat_requests(upstream)
    assert "search_notes" in offered_tools(requests[0])
    assert "search_notes" not in offered_tools(requests[1])
    broken = first_break(requests)
    assert broken is not None and "in tools" in broken, broken


def test_switching_web_search_on_mid_chat_rewrites_the_prefix(
    cached_setup, admin, preserve, page_for, make_user, upstream
):
    preserve(RETRIEVAL_CONFIG)
    with admin.client() as client:
        save_web_settings(client, ENABLE_WEB_SEARCH=True)
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    ask(page, upstream, "good morning", reply.text("Morning."))
    page.get_by_label("Integrations").click()
    page.get_by_role("menu").get_by_role("button", name="Web Search").click()
    page.keyboard.press("Escape")
    ask(page, upstream, "look up the tides", reply.text("High at noon."))

    requests = chat_requests(upstream)
    assert "search_web" not in offered_tools(requests[0])
    assert "search_web" in offered_tools(requests[1])
    broken = first_break(requests)
    assert broken is not None and "in tools" in broken, broken


def test_a_skill_mentioned_mid_chat_rewrites_the_prefix(
    cached_setup, admin, page_for, make_user, upstream
):
    skill_id = f"tide-tables-{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": "Tide tables",
                "description": "Reading tide tables",
                "content": "Read the high tide column first.",
                "meta": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        try:
            page = page_for(make_user())
            page.goto(f"/?models={cached_setup.id}")
            ask(page, upstream, "good morning", reply.text("Morning."))
            chat_input(page).click()
            page.keyboard.type("$Tide")
            page.get_by_role("button", name=re.compile("Tide tables")).click()
            ask(page, upstream, "when is high tide?", reply.text("At noon."))
        finally:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")

    requests = chat_requests(upstream)
    assert "Read the high tide column first." in requests[1]["messages"][0]["content"]
    assert first_break(requests) is not None, "a skill mentioned mid-chat left the prefix alone"


def test_a_workspace_tool_switched_on_mid_chat_rewrites_the_prefix(
    cached_setup, admin, page_for, make_user, upstream
):
    with python_tool(admin, TIDE_TOOL, name="Tide clock"):
        page = page_for(make_user())
        page.goto(f"/?models={cached_setup.id}")
        ask(page, upstream, "good morning", reply.text("Morning."))
        turn_on_tool(page, "Tide clock")
        ask(page, upstream, "when is high tide?", reply.text("At noon."))

    requests = chat_requests(upstream)
    assert "tide_time" in offered_tools(requests[1])
    broken = first_break(requests)
    assert broken is not None and "in tools" in broken, broken


# --- Editing an earlier question ------------------------------------------------------------


def test_an_earlier_question_edited_and_sent_again_rewrites_the_prefix(
    cached_setup, page_for, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    ask(page, upstream, "when does the ferry leave?", reply.text("At 06:40."))
    ask(page, upstream, "and the bus?", reply.text("At 07:15."))
    question = conversation(page).locator(".chat-user").first
    question.hover()
    question.get_by_role("button", name="Edit").click()
    question.locator("textarea").fill("when does the night ferry leave?")
    upstream.queue(reply.text("At 23:10.", match=reply.answering("night ferry")))
    question.get_by_role("button", name="Send").click()
    expect_reply(page, "At 23:10.")

    requests = chat_requests(upstream)
    assert "night ferry" in str(requests[-1]["messages"])
    broken = first_break(requests)
    assert broken is not None, "an edited question left the prefix append-only"


# --- breakers the page does not name --------------------------------------------------------


def test_moving_a_chat_into_a_folder_with_a_prompt_rewrites_the_prefix(
    page_for, cached_setup, make_user, upstream
):
    person = make_user()
    with person.client() as client:
        folder = client.post(
            "/api/v1/folders/",
            json={"name": "Harbour", "data": {"system_prompt": "Answer as the harbour master."}},
        )
    assert folder.status_code == 200, folder.text
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    chat_id = page.url.rsplit("/", 1)[-1]
    with person.client() as client:
        moved = client.post(
            f"/api/v1/chats/{chat_id}/folder", json={"folder_id": folder.json()["id"]}
        )
    assert moved.status_code == 200, moved.text
    ask(page, upstream, "anything new?", reply.text("Nothing new."))

    broken = first_break(chat_requests(upstream))
    assert broken is not None and "messages[0] (system)" in broken, broken


def test_a_knowledge_base_attached_after_the_first_turn_rewrites_the_prefix(
    page_for, cached_setup, admin, make_user, upstream
):
    person = make_user()
    with admin.client() as client, knowledge_base(client, "Timetables", [EVERYONE_READS]):
        page = page_for(person)
        page.goto(f"/?models={cached_setup.id}")
        ask(page, upstream, "good morning", reply.text("Morning."))
        attach_from_menu(page, "Attach Knowledge", "Timetables")
        ask(page, upstream, "when does the bus leave?", reply.text("At 07:15."))

    broken = first_break(chat_requests(upstream))
    assert broken is not None and "messages[0] (system)" in broken, broken


def test_a_skill_shared_with_the_person_mid_chat_rewrites_the_prefix(
    page_for, cached_setup, admin, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    ask(page, upstream, "good morning", reply.text("Morning."))
    skill_id = f"tide-tables-{uuid.uuid4().hex[:6]}"
    with admin.client() as client:
        created = client.post(
            "/api/v1/skills/create",
            json={
                "id": skill_id,
                "name": "Tide tables",
                "description": "Reading tide tables",
                "content": "Read the high tide column first.",
                "meta": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        try:
            ask(page, upstream, "anything new?", reply.text("Nothing new."))
        finally:
            client.delete(f"/api/v1/skills/id/{skill_id}/delete")

    requests = chat_requests(upstream)
    assert "Tide tables" in requests[1]["messages"][0]["content"]
    broken = first_break(requests)
    assert broken is not None, "a new skill left the prefix alone"
