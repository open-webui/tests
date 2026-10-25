"""Journey: the admin Session Explorer counts and traces what three seeded chats hold.

Two fresh users own three chats in a group of their own: one with two token-bearing responses,
one whose single response saved a tool call and its result, one whose single response failed.
Scoped to that group, the explorer lists the three with their owners, responses, tokens and
errors, its summary counts the sessions and the error rate, a token sort reorders them and a
search narrows to one. Opening the tool chat shows its stats, usage chart, timeline and the tool
call in the trace; clicking anywhere on the failed chat's row opens it, naming its owner and
showing the response as an error. A plain user is sent back to the chat by the analytics page,
cannot see the Sessions tab and is refused the sessions API.

The chats are stored directly (as an import leaves them), so the saved output items, usage and
error that the live pipeline rarely writes together are present and the counts are the test's own.

Discriminates: passes on dev 8889e8d6a against its own build; in a frontend build of it whose rows
open only from the title and name no owner, the listing and failed-session tests fail and the
rest pass; in a backend copy, with the sessions query reporting `errors` as 0 the listing test
fails (the failed chat shows 0 errors, rechecked on 8889e8d6a), with the trace never marking an
assistant response `failed` the error-trace test fails (the response is not shown as an error),
with the trace dropping `function_call` items the tool-trace test fails (the tool call never
appears), with `list_sessions` ignoring `order_by`/`direction` the sort test fails (the token
sort does not reorder), and with session access granted to verified users the guard test fails
(the sessions API answers 200); those four were proven on dev 7b303cf49.
"""

from __future__ import annotations

import time
import uuid
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.upstream import MOCK_MODEL_ID
from utils.analytics import group_picker
from utils.chat_ui import chat_input
from utils.session_analytics import (
    detail_stat,
    open_session,
    search_box,
    session_cell,
    session_row,
    session_rows,
    sessions_tab,
    signal,
    sort_header,
    trace_filter,
    visualization,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PUBLIC_READ = [{"principal_type": "user", "principal_id": "*", "permission": "read"}]
TOOL_NAME = "lookup_weather"


def _usage(input_tokens: int, output_tokens: int) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def _seed_session(client, title: str, model: str, messages: list[dict]) -> str:
    """Store `messages` as one branch under a chat named `title`, as an import leaves it.

    A title is given so the explorer and its search have something the test owns to match on.
    """
    history: dict[str, dict] = {}
    parent_id = None
    for message in messages:
        message_id = str(uuid.uuid4())
        history[message_id] = {
            "model": model,
            "timestamp": int(time.time()),
            "done": True,
            **message,
            "id": message_id,
            "parentId": parent_id,
            "childrenIds": [],
        }
        if parent_id:
            history[parent_id]["childrenIds"].append(message_id)
        parent_id = message_id
    chat = {
        "title": title,
        "models": [model],
        "history": {"currentId": parent_id, "messages": history},
    }
    created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, f"seeding {title!r} failed: {created.text}"
    return created.json()["id"]


@pytest.fixture
def presets(admin) -> Iterator[tuple[str, str]]:
    """Two presets only this module records against, so the model names resolve in the explorer."""
    names = tuple(f"session-{word}-{uuid.uuid4().hex[:6]}" for word in ("aster", "birch"))
    with admin.client() as client:
        for name in names:
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": name,
                    "base_model_id": MOCK_MODEL_ID,
                    "name": name,
                    "meta": {},
                    "params": {},
                    "access_grants": PUBLIC_READ,
                },
            )
            assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        yield names
        for name in names:
            client.post("/api/v1/models/model/delete", json={"id": name})


class Sessions:
    """The three seeded chats, the group holding just their owners, each chat's title and owner."""

    def __init__(self, group_name: str, tag: str, titles: dict[str, str], owners: dict[str, str]):
        self.group_name = group_name
        self.tag = tag
        self.titles = titles
        self.owners = owners


@pytest.fixture
def sessions(admin, make_user, presets) -> Sessions:
    aster, birch = presets
    first = make_user(name=f"Session First {uuid.uuid4().hex[:6]}")
    second = make_user(name=f"Session Second {uuid.uuid4().hex[:6]}")
    group_id = make_group(admin, [first, second])
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]

    tag = uuid.uuid4().hex[:8]
    titles = {
        "tokens": f"Alpha tokens {tag}",
        "tool": f"Bravo tool {tag}",
        "error": f"Charlie error {tag}",
    }
    tool_call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": TOOL_NAME,
        "arguments": '{"city": "Berlin"}',
        "status": "completed",
    }
    tool_result = {
        "type": "function_call_output",
        "id": "fc_1_out",
        "call_id": "call_1",
        "output": [{"type": "input_text", "text": '{"temp_c": 21}'}],
        "status": "completed",
    }
    with first.client() as client:
        _seed_session(
            client,
            titles["tokens"],
            aster,
            [
                {"role": "user", "content": "alpha ask one"},
                {"role": "assistant", "content": "alpha reply one", "usage": _usage(100, 20)},
                {"role": "user", "content": "alpha ask two"},
                {"role": "assistant", "content": "alpha reply two", "usage": _usage(80, 50)},
            ],
        )
        _seed_session(
            client,
            titles["tool"],
            aster,
            [
                {"role": "user", "content": "bravo ask"},
                {
                    "role": "assistant",
                    "content": "bravo reply",
                    "usage": _usage(50, 10),
                    "output": [tool_call, tool_result],
                },
            ],
        )
    with second.client() as client:
        _seed_session(
            client,
            titles["error"],
            birch,
            [
                {"role": "user", "content": "charlie ask"},
                {
                    "role": "assistant",
                    "content": "charlie reply",
                    "error": {"content": "the model provider is unavailable"},
                },
            ],
        )
    owners = {"tokens": first.name, "tool": first.name, "error": second.name}
    return Sessions(group_name, tag, titles, owners)


@pytest.fixture
def explorer_view(page_for, make_user, sessions) -> tuple[Page, Locator]:
    """A fresh admin on the Sessions view, scoped to the seeded group."""
    page = page_for(make_user(role="admin"), path="/admin/analytics")
    dialog = page.get_by_role("dialog")
    expect(sessions_tab(dialog)).to_be_visible()
    sessions_tab(dialog).click()
    expect(group_picker(dialog)).to_be_visible()
    group_picker(dialog).select_option(label=sessions.group_name)
    expect(sessions_tab(dialog, selected=True)).to_be_visible()
    return page, dialog


def _expect_order(dialog: Locator, titles: list[str]) -> None:
    """Each row's title cell, in table order; the cell also carries the model subtitle."""
    rows = session_rows(dialog)
    expect(rows).to_have_count(len(titles))
    for index, title in enumerate(titles):
        expect(rows.nth(index).get_by_role("cell").first).to_contain_text(title)


def test_the_explorer_lists_each_seeded_sessions_responses_tokens_and_errors(
    explorer_view, sessions
):
    _, dialog = explorer_view
    expect(session_rows(dialog)).to_have_count(3)

    alpha = session_row(dialog, sessions.titles["tokens"])
    expect(session_cell(alpha, "responses")).to_have_text("2")
    expect(session_cell(alpha, "tokens")).to_have_text("250")
    expect(session_cell(alpha, "errors")).to_have_text("0")

    bravo = session_row(dialog, sessions.titles["tool"])
    expect(session_cell(bravo, "responses")).to_have_text("1")
    expect(session_cell(bravo, "tokens")).to_have_text("60")
    expect(session_cell(bravo, "errors")).to_have_text("0")

    charlie = session_row(dialog, sessions.titles["error"])
    expect(session_cell(charlie, "user")).to_have_text(sessions.owners["error"])
    expect(session_cell(alpha, "user")).to_have_text(sessions.owners["tokens"])
    expect(session_cell(charlie, "responses")).to_have_text("1")
    expect(session_cell(charlie, "tokens")).to_have_text("-")
    expect(session_cell(charlie, "errors")).to_have_text("1")

    # the summary signals above the table count the scope as a whole
    expect(signal(dialog, "Sessions")).to_have_text("3")
    expect(signal(dialog, "Response errors")).to_have_text("25.0%")


def test_sorting_by_tokens_reorders_the_sessions(explorer_view, sessions):
    _, dialog = explorer_view
    alpha, bravo, charlie = (sessions.titles[k] for k in ("tokens", "tool", "error"))
    # loads sorted by tokens descending; the unrecorded chat has no tokens and sorts last
    _expect_order(dialog, [alpha, bravo, charlie])

    sort_header(dialog, "Tokens").click()  # toggles to ascending
    _expect_order(dialog, [bravo, alpha, charlie])


def test_searching_narrows_the_explorer_to_the_matching_session(explorer_view, sessions):
    _, dialog = explorer_view
    search_box(dialog).fill("Alpha")
    # the narrowed count must settle before reading the remaining row
    expect(session_rows(dialog)).to_have_count(1)
    expect(session_rows(dialog).first.get_by_role("cell").first).to_contain_text(
        sessions.titles["tokens"]
    )


def test_opening_the_tool_session_shows_its_usage_and_the_tool_call_in_the_trace(
    explorer_view, sessions
):
    _, dialog = explorer_view
    detail = open_session(dialog, sessions.titles["tool"])
    expect(detail.get_by_role("heading", name=sessions.titles["tool"])).to_be_visible()
    expect(detail_stat(detail, "Responses")).to_have_text("1")
    expect(detail_stat(detail, "Tokens")).to_have_text("60")

    # the recorded response drives the token chart shown by default
    expect(detail.get_by_role("figure", name="Recorded tokens by response")).to_be_visible()

    trace_filter(detail, "Tools").click()
    expect(detail.get_by_text(TOOL_NAME)).to_be_visible()
    expect(detail.get_by_text("No saved items match this filter.")).to_have_count(0)

    # the timeline draws the saved sequence of the same items
    visualization(detail, "Timeline").click()
    expect(detail.get_by_role("figure", name="Execution timeline")).to_be_visible()


def test_opening_the_failed_session_shows_the_response_as_an_error(explorer_view, sessions):
    _, dialog = explorer_view
    # the whole row opens its session, not only the title
    session_cell(session_row(dialog, sessions.titles["error"]), "errors").click()
    detail = dialog.page.get_by_role("dialog", name="Session details")
    expect(detail.get_by_role("heading", name=sessions.titles["error"])).to_be_visible()
    # the detail names the owner beside the session's times
    expect(detail.get_by_text(sessions.owners["error"], exact=True)).to_be_visible()

    trace_filter(detail, "Errors").click()
    # the failed response survives the errors filter and is marked as an error
    expect(
        detail.get_by_role("button").filter(has_text="Errors").filter(has_text="birch")
    ).to_be_visible()
    expect(detail.get_by_text("No saved items match this filter.")).to_have_count(0)


def test_a_user_is_kept_out_of_the_session_explorer_and_refused_its_api(page_for, make_user):
    user = make_user()
    page = page_for(user, path="/admin/analytics")
    expect(chat_input(page)).to_be_visible()
    expect(page).to_have_url(f"{user.base_url}/")
    expect(page.get_by_role("dialog")).to_have_count(0)

    page.goto("/?settings=admin%3Aanalytics")
    expect(page.get_by_role("dialog").get_by_role("tab", selected=True)).to_be_visible()
    expect(sessions_tab(page.get_by_role("dialog"))).to_have_count(0)

    now = int(time.time())
    with user.client() as client:
        refused = client.get(
            "/api/v1/analytics/sessions", params={"start_date": now - 3600, "end_date": now}
        )
    assert refused.status_code == 401, refused.text
