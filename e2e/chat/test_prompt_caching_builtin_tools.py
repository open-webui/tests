"""Journey: every builtin tool family, called in the browser on the Prompt Caching page's setup.

The Prompt Caching docs page tells a person to keep retrieval agentic: Builtin Tools on with
every category left enabled, File Context and Citations off, native function calling, so that
each tool result is appended at the end and the start of the request never changes. Here a person
chats through the web client on a model set up as its checklist says, and the scripted model
calls every builtin tool the setup offers at least once, over several turns: the chat file tools
on a file attached in the chat input (alone with reasoning, two in parallel and two rounds in a
row), the knowledge tools on the model's knowledge, the discovery tools on a model without any,
a knowledge base, a note and a chat attached through the chat input's menu, the note and chat
history tools, every memory tool, the time, task list and question tools (the question answered
on its card), the channel, automation and calendar tools, image generation and editing, and web
search, page fetch and a notification. Every consecutive pair of the provider's requests is
checked: each one repeats the tool list and every message of the one before it byte for byte and
only adds to the end.

`test_the_time_task_list_and_question_tools_only_append` failed on dev 1c010b438 in CI: the reply
stayed blank on the page; since de73bb830 the reply in a new chat sometimes stays blank until a
reload although the server saved it whole (open-webui/open-webui#32091).

Discriminates: passes on dev 30f3f6a8f. In a backend copy with a clock value added to the
model's system prompt every test fails.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import admin_of, create_user
from harness.channel_chat import enable_channels, serve_openai_images
from harness.image_engines import IMAGES_CONFIG, save_image_settings
from harness.listener import json_answer, text_answer
from harness.prompt_caching import (
    ADMIN_CONFIG,
    assert_append_only,
    cache_optimal_model,
    turn_off_memory_system_context,
)
from harness.web_retrieval import (
    LOCAL_WEB_FETCH,
    RETRIEVAL_CONFIG,
    save_web_settings,
    serve_search_results,
)
from utils.cached_chat import (
    ask,
    attach,
    attach_from_menu,
    called_tools,
    calling,
    calling_together,
    chat_requests,
    offered_tools,
    tool_results,
)
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

NOTES = ("berths.txt", "Berth 3 is reserved for the pilot boat.\nBerth 5 is free on Mondays.\n")
SCHEDULE = "FREQ=DAILY;BYHOUR=7;BYMINUTE=0"
PLAN_QUESTION = {
    "id": "crossing",
    "header": "Crossing",
    "question": "Which crossing suits you?",
    "options": [
        {"label": "Morning", "description": "The 06:40 ferry"},
        {"label": "Evening", "description": "The 18:10 ferry"},
    ],
}


@pytest.fixture
def cached_setup(admin, preserve):
    """The page's setup: its model, and the memory system context switched off."""
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin) as model:
        yield model


def last_result(upstream) -> dict:
    """What the latest tool call answered, as the provider was sent it."""
    return json.loads(chat_requests(upstream)[-1]["messages"][-1]["content"])


def attached_file_id(request: dict) -> str:
    """The id the chat's `<attached_files>` block gives the first attached file."""
    found = re.search(r'<file type="file" id="([^"]+)"', str(request["messages"]))
    assert found, "the request names no attached file"
    return found.group(1)


def assert_called_every(requests: list[dict], tools: set[str]) -> None:
    offered = offered_tools(requests[-1])
    assert tools <= offered, f"not offered: {sorted(tools - offered)}"
    called = called_tools(requests[-1])
    assert tools <= called, f"never called: {sorted(tools - called)}"


def test_the_file_tools_on_a_file_attached_in_the_chat_only_append(
    page_for, cached_setup, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    attach(page, *NOTES)
    ask(
        page,
        upstream,
        "which files do I have?",
        calling("list_chat_files", {}, reasoning="I should look first"),
        reply.text("One.", reasoning="the list has one file"),
    )
    file_id = attached_file_id(chat_requests(upstream)[0])
    ask(
        page,
        upstream,
        "which berth is free and which is reserved?",
        calling_together(
            ("query_chat_files", {"query": "free berth"}),
            ("grep_chat_files", {"pattern": "reserved"}),
        ),
        reply.text("Berth 5 is free, berth 3 reserved (berths.txt)."),
    )
    ask(
        page,
        upstream,
        "read me the first line, then the second",
        calling("view_file", {"file_id": file_id, "start_line": 1, "end_line": 1}),
        calling("view_file", {"file_id": file_id, "start_line": 2, "end_line": 2}),
        reply.text("Both lines read (berths.txt)."),
    )
    page.reload()
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(requests, {"list_chat_files", "query_chat_files", "grep_chat_files"})
    assert "view_file" in called_tools(requests[-1])
    assert tool_results(requests[-1]).count("Berth 5 is free") >= 2, tool_results(requests[-1])
    assert_append_only(requests)


def test_the_knowledge_tools_on_the_models_knowledge_only_append(
    page_for, cached_setup, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    handbook = cached_setup.handbook_file_id

    ask(page, upstream, "what do you know?", calling("list_knowledge", {}), reply.text("A book."))
    ask(
        page,
        upstream,
        "find the handbook",
        calling("search_knowledge_files", {"query": "handbook"}),
        calling("query_knowledge_files", {"query": "ferry"}),
        reply.text("It says 06:40 (handbook.txt)."),
    )
    ask(
        page,
        upstream,
        "show me the pier lines",
        calling_together(
            ("grep_knowledge_files", {"pattern": "pier"}),
            ("view_knowledge_file", {"file_id": handbook}),
            ("view_file", {"file_id": handbook}),
        ),
        reply.text("Pier 7 (handbook.txt)."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {
            "list_knowledge",
            "search_knowledge_files",
            "query_knowledge_files",
            "grep_knowledge_files",
            "view_knowledge_file",
            "view_file",
        },
    )
    assert tool_results(requests[-1]).count("pier 7") >= 3, tool_results(requests[-1])
    assert_append_only(requests)


def test_the_knowledge_discovery_tools_only_append(admin, preserve, page_for, make_user, upstream):
    preserve("admin_config")
    turn_off_memory_system_context(admin)
    with cache_optimal_model(admin, with_knowledge=False) as model:
        page = page_for(make_user())
        page.goto(f"/?models={model.id}")
        ask(
            page,
            upstream,
            "which knowledge bases are there?",
            calling("list_knowledge_bases", {}),
            reply.text("A handbook."),
        )
        ask(
            page,
            upstream,
            "find the harbour one",
            calling_together(
                ("search_knowledge_bases", {"query": "Harbour"}),
                ("query_knowledge_bases", {"query": "harbour handbook"}),
            ),
            reply.text("Harbour handbook."),
        )
        ask(
            page,
            upstream,
            "read it",
            calling("search_knowledge_files", {"query": "handbook"}),
            calling("view_knowledge_file", {"file_id": model.handbook_file_id}),
            reply.text("06:40 from pier 7 (handbook.txt)."),
        )
        ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {
            "list_knowledge_bases",
            "search_knowledge_bases",
            "query_knowledge_bases",
            "search_knowledge_files",
            "view_knowledge_file",
        },
    )
    assert "06:40" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_a_knowledge_base_note_and_chat_attached_in_the_chat_only_append(
    page_for, cached_setup, make_user, upstream
):
    person = make_user()
    with person.client() as client:
        note = client.post(
            "/api/v1/notes/create",
            json={"title": "Packing list", "data": {"content": {"md": "rain jacket, tickets"}}},
        )
        assert note.status_code == 200, note.text
        earlier = client.post(
            "/api/v1/chats/new",
            json={"chat": {"title": "An earlier trip", "messages": [], "history": {}}},
        )
        assert earlier.status_code == 200, earlier.text
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    attach_from_menu(page, "Attach Knowledge", "Harbour handbook")
    attach_from_menu(page, "Attach Notes", "Packing list")
    attach_from_menu(page, "Reference Chats", "An earlier trip")
    ask(
        page,
        upstream,
        "when does the ferry leave?",
        calling("query_knowledge_files", {"query": "ferry"}),
        reply.text("06:40 (handbook.txt)."),
    )
    ask(
        page,
        upstream,
        "what should I pack?",
        calling("view_note", {"note_id": note.json()["id"]}),
        reply.text("A rain jacket."),
    )
    ask(
        page,
        upstream,
        "what did we say last time?",
        calling("view_chat", {"chat_id": earlier.json()["id"]}),
        reply.text("We talked about a trip."),
    )
    page.reload()
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    results = tool_results(requests[-1])
    assert "06:40" in results and "rain jacket" in results and "earlier trip" in results, results
    assert_append_only(requests)


def test_the_note_and_chat_history_tools_only_append(page_for, cached_setup, make_user, upstream):
    person = make_user()
    with person.client() as client:
        earlier = client.post(
            "/api/v1/chats/new",
            json={"chat": {"title": "Tide tables for May", "messages": [], "history": {}}},
        )
        assert earlier.status_code == 200, earlier.text
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    ask(
        page,
        upstream,
        "write down the ferry time",
        calling("write_note", {"title": "Ferry", "content": "06:40 from pier 7"}),
        reply.text("Noted."),
    )
    note_id = last_result(upstream)["id"]
    ask(
        page,
        upstream,
        "find that note and change it",
        calling("search_notes", {"query": "Ferry"}),
        calling("replace_note_content", {"note_id": note_id, "content": "06:50 from pier 7"}),
        calling("view_note", {"note_id": note_id}),
        reply.text("Changed to 06:50."),
    )
    ask(
        page,
        upstream,
        "what did I ask about tides?",
        calling("search_chats", {"query": "Tide tables"}),
        calling("view_chat", {"chat_id": earlier.json()["id"]}),
        reply.text("The May tide tables."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {"write_note", "search_notes", "replace_note_content", "view_note", "search_chats"},
    )
    assert "view_chat" in called_tools(requests[-1])
    assert "06:50 from pier 7" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_every_memory_tool_only_appends(page_for, cached_setup, make_user, upstream):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    ask(
        page,
        upstream,
        "remember that I take the early ferry",
        calling("add_memory", {"content": "takes the early ferry", "path": "travel"}),
        reply.text("Noted."),
    )
    memory_id = last_result(upstream)["id"]
    ask(
        page,
        upstream,
        "what do you know about my travel?",
        calling_together(
            ("search_memories", {"query": "ferry"}),
            ("list_memories", {}),
            ("list_memory_paths", {}),
            ("read_memory_path", {"path": "travel"}),
        ),
        reply.text("You take the early ferry."),
    )
    ask(
        page,
        upstream,
        "it is the late ferry now, and I like tea",
        calling(
            "replace_memory_content", {"memory_id": memory_id, "content": "takes the late ferry"}
        ),
        calling(
            "update_memory",
            {"operations": [{"action": "add", "content": "likes tea", "path": "food"}]},
        ),
        reply.text("Updated."),
    )
    ask(
        page,
        upstream,
        "forget the ferry",
        calling("delete_memory", {"memory_id": memory_id}),
        reply.text("Forgotten."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {
            "add_memory",
            "search_memories",
            "list_memories",
            "list_memory_paths",
            "read_memory_path",
            "replace_memory_content",
            "update_memory",
            "delete_memory",
        },
    )
    assert "takes the late ferry" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_the_time_task_list_and_question_tools_only_append(
    page_for, cached_setup, make_user, upstream
):
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")

    ask(
        page,
        upstream,
        "what is the date now and a week ago?",
        calling_together(
            ("get_current_timestamp", {}),
            ("calculate_timestamp", {"weeks_ago": 1}),
        ),
        reply.text("Both dates read."),
    )
    ask(
        page,
        upstream,
        "plan my trip",
        calling(
            "create_tasks",
            {
                "tasks": [
                    {"id": "ticket", "content": "Buy a ticket"},
                    {"id": "pack", "content": "Pack a rain jacket"},
                ]
            },
        ),
        calling("update_task", {"id": "ticket", "status": "completed"}),
        reply.text("The ticket is done."),
    )
    question = "which crossing should I take?"
    upstream.queue(
        calling("ask_user", {"questions": [PLAN_QUESTION]}, match=reply.answering(question)),
        reply.text("The morning ferry it is.", match=reply.answering(question)),
    )
    send(page, question)
    expect(page.get_by_text("Which crossing suits you?")).to_be_visible(timeout=30_000)
    page.get_by_role("button", name=re.compile("Morning")).click()  # the only question submits
    expect_reply(page, "The morning ferry it is.")
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {"get_current_timestamp", "calculate_timestamp", "create_tasks", "update_task", "ask_user"},
    )
    assert "Morning" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_the_channel_tools_only_append(page_for, cached_setup, admin, make_user, upstream):
    with admin.client() as client:
        enable_channels(client)
    person = make_user()
    tag = uuid.uuid4().hex[:6]
    with person.client() as client:
        channel = client.post(
            "/api/v1/channels/create",
            json={"name": f"harbour-{tag}", "description": "harbour news", "type": "group"},
        )
        assert channel.status_code == 200, channel.text
        channel_id = channel.json()["id"]
        posted = client.post(
            f"/api/v1/channels/{channel_id}/messages/post",
            json={"content": f"the ferry is late today {tag}"},
        )
        assert posted.status_code == 200, posted.text
        message_id = posted.json()["id"]
        answered = client.post(
            f"/api/v1/channels/{channel_id}/messages/post",
            json={"content": "by twenty minutes", "parent_id": message_id},
        )
        assert answered.status_code == 200, answered.text
    page = page_for(person)
    page.goto(f"/?models={cached_setup.id}")

    ask(
        page,
        upstream,
        "any harbour news?",
        calling_together(
            ("search_channels", {"query": "harbour"}),
            ("search_channel_messages", {"query": "ferry"}),
        ),
        reply.text("The ferry is late."),
    )
    ask(
        page,
        upstream,
        "how late?",
        calling("view_channel_message", {"message_id": message_id}),
        calling("view_channel_thread", {"parent_message_id": message_id}),
        reply.text("Twenty minutes."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {
            "search_channels",
            "search_channel_messages",
            "view_channel_message",
            "view_channel_thread",
        },
    )
    assert "by twenty minutes" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def test_the_automation_and_calendar_tools_only_append(page_for, cached_setup, make_user, upstream):
    page = page_for(make_user(role="admin"))  # automations need a permission users lack
    page.goto(f"/?models={cached_setup.id}")

    ask(
        page,
        upstream,
        "send me the ferry times every morning",
        calling("create_automation", {"name": "Ferry", "prompt": "ferry times", "rrule": SCHEDULE}),
        reply.text("Scheduled."),
    )
    automation_id = last_result(upstream)["id"]
    ask(
        page,
        upstream,
        "make it the bus times and pause it",
        calling("list_automations", {}),
        calling_together(
            ("update_automation", {"automation_id": automation_id, "name": "Bus"}),
            ("toggle_automation", {"automation_id": automation_id}),
        ),
        reply.text("Renamed and paused."),
    )
    ask(
        page,
        upstream,
        "drop it and book the crossing",
        calling("delete_automation", {"automation_id": automation_id}),
        calling("create_calendar_event", {"title": "Crossing", "start": "2026-11-03 06:40"}),
        reply.text("Booked."),
    )
    event_id = last_result(upstream)["id"]
    ask(
        page,
        upstream,
        "move it and then cancel it",
        calling("search_calendar_events", {"query": "Crossing"}),
        calling("update_calendar_event", {"event_id": event_id, "start": "2026-11-03 18:10"}),
        calling("delete_calendar_event", {"event_id": event_id}),
        reply.text("Moved and cancelled."),
    )
    ask(page, upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(upstream)
    assert_called_every(
        requests,
        {
            "create_automation",
            "list_automations",
            "update_automation",
            "toggle_automation",
            "delete_automation",
            "create_calendar_event",
            "search_calendar_events",
            "update_calendar_event",
            "delete_calendar_event",
        },
    )
    assert "18:10" in tool_results(requests[-1]), tool_results(requests[-1])
    assert_append_only(requests)


def turn_on_integration(page: Page, name: str) -> None:
    """Switch a feature on for this chat from the Integrations menu, before the first message."""
    expect(chat_input(page)).to_be_visible()
    page.get_by_label("Integrations").click()
    page.get_by_role("menu").get_by_role("button", name=name, exact=True).click()
    page.keyboard.press("Escape")


def test_the_image_tools_only_append(
    page_for, cached_setup, admin, make_user, preserve, listener, upstream
):
    preserve(IMAGES_CONFIG)
    with admin.client() as client:
        save_image_settings(client, **serve_openai_images(listener))
    page = page_for(make_user())
    page.goto(f"/?models={cached_setup.id}")
    turn_on_integration(page, "Image")

    ask(
        page,
        upstream,
        "draw the harbour",
        calling("generate_image", {"prompt": "a harbour at dawn"}),
        reply.text("Here is the harbour."),
    )
    drawn = re.search(r"/api/v1/files/[^\s\"')\]]+", tool_results(chat_requests(upstream)[-1]))
    assert drawn, tool_results(chat_requests(upstream)[-1])
    ask(
        page,
        upstream,
        "make it evening",
        calling("edit_image", {"prompt": "the same at dusk", "image_urls": [drawn.group(0)]}),
        reply.text("Here it is at dusk."),
    )
    ask(page, upstream, "thanks", reply.text("Enjoy the view."))

    requests = chat_requests(upstream)
    assert_called_every(requests, {"generate_image", "edit_image"})
    assert "image_url" in json.dumps(requests[-1]["messages"]), "no picture reached the model"
    assert_append_only(requests)


@pytest.mark.slow
def test_web_search_fetch_and_a_notification_only_append(
    instance_with, preserve, page_for, listener
):
    fetching = instance_with(LOCAL_WEB_FETCH)
    preserve(RETRIEVAL_CONFIG, "admin_config", "permissions", on=fetching)
    tides = f"{listener.base_url}/tides"
    listener.route(
        "GET", "/tides", text_answer("<html><body><p>High tide at noon.</p></body></html>")
    )
    listener.route("POST", "/hook", json_answer({}))
    fetching_admin = admin_of(fetching)
    with fetching_admin.client() as client:
        save_web_settings(client, **serve_search_results(listener, [tides]))
        config = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**config, "ENABLE_USER_WEBHOOKS": True})
        assert saved.status_code == 200, saved.text
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["webhooks"] = True
        client.post("/api/v1/users/default/permissions", json=permissions).raise_for_status()
    turn_off_memory_system_context(fetching_admin)
    person = create_user(fetching)
    with person.client() as client:
        target = {"id": "phone", "config": {"url": f"{listener.base_url}/hook"}}
        client.post("/api/v1/notifications/targets", json=target).raise_for_status()

    with cache_optimal_model(fetching_admin) as model:
        page = page_for(person)
        page.goto(f"/?models={model.id}")
        turn_on_integration(page, "Web Search")
        ask(
            page,
            fetching.upstream,
            "search the tides",
            calling("search_web", {"query": "tides"}),
            reply.text("Found a page."),
        )
        ask(
            page,
            fetching.upstream,
            "read that page and tell my phone",
            calling("fetch_url", {"url": tides}),
            calling("notify", {"message": "High tide at noon", "title": "Tides"}),
            reply.text("High tide is at noon."),
        )
        ask(page, fetching.upstream, "thanks", reply.text("Safe travels."))

    requests = chat_requests(fetching.upstream)
    assert_called_every(requests, {"search_web", "fetch_url", "notify"})
    assert "High tide at noon." in tool_results(requests[-1]), tool_results(requests[-1])
    assert len(listener.requests_to("/hook")) == 1, "the notification never arrived"
    assert_append_only(requests)
