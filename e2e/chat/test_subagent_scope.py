"""Journey: what a sub-agent is given and who sees it, from the chat page.

The tool a person switches on under Integrations is the tool the sub-agent gets, and one left
off stays out of its reach. The sub-agent's own chat never shows in the sidebar or its search,
not for the account that delegated and not for another. A chat shared by link shows the reader
the delegation row and the answer it holds.

Twin of integration/tools/test_subagent_scope.py.

`test_a_chat_shared_by_link_shows_the_reader_the_delegation_and_its_answer` and
`test_a_tool_switched_on_in_the_chat_reaches_the_subagent_and_one_left_off_does_not` are red on dev
62f70a844: since de73bb830 a chat request whose reply message is already stored in the chat, the way
automations, sub-agents and timers prepare their reply, is refused with 409 and the reply is never
written (open-webui/open-webui#32066, fixed by 641c82df1).

Discriminates: passes on dev 176d31d1d. In backend copies each test turns red with its edit: the
sub-agent's tool ids dropped, the sub-agent's chat left listed, and the sub-agent's answer left
out of a shared copy of the chat.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.python_tools import python_tool
from utils.cached_chat import turn_on_tool
from utils.chat_ui import conversation, expect_reply, last_reply, send
from utils.member_picker import pick_member
from utils.search_dialog import no_matches, open_search_from_sidebar, result, search

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
TOOL_SOURCE = '''
class Tools:
    def {name}(self, text: str) -> str:
        """Shout the text."""
        return text.upper()
'''


@pytest.fixture
def subagents_on(admin, preserve):
    preserve(SUBAGENTS)
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()


def unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:8]}"


def offered(request: dict) -> set[str]:
    return {tool["function"]["name"] for tool in request.get("tools") or []}


def sub_requests(upstream, task: str) -> list[dict]:
    return [r for r in upstream.chat_requests() if reply.answering(task)(r)]


def delegation(prompt: str, task: str, *sub_steps: reply.Reply) -> list[reply.Reply]:
    return [
        reply.tool_call("delegate_task", {"task": task}, match=reply.answering(prompt)),
        *sub_steps,
        reply.text("The helper is done.", match=reply.answering(prompt)),
    ]


def delegated_row(page: Page, task: str) -> Locator:
    # a narrow chat column shows the short label, without "View Result from"
    name = re.escape(f'Sub-agent: "{task}"')
    return last_reply(page).get_by_role("button", name=re.compile(name))


def test_a_tool_switched_on_in_the_chat_reaches_the_subagent_and_one_left_off_does_not(
    subagents_on, admin, page_for, make_user, upstream
):
    used, unused = f"shout_{uuid.uuid4().hex[:6]}", f"whisper_{uuid.uuid4().hex[:6]}"
    used_name, unused_name = unique("Loud tools"), unique("Quiet tools")
    prompt, task = unique("hand this over"), unique("shout the word")
    upstream.queue(
        *delegation(
            prompt,
            task,
            reply.tool_call(used, {"text": "ahoy"}, "sub_call", match=reply.answering(task)),
            reply.text("Shouted.", match=reply.answering(task)),
        )
    )
    with (
        python_tool(admin, TOOL_SOURCE.format(name=used), used_name),
        python_tool(admin, TOOL_SOURCE.format(name=unused), unused_name),
    ):
        page = page_for(make_user())
        turn_on_tool(page, used_name)
        send(page, prompt)
        expect_reply(page, "The helper is done.")

    first, *_, last = sub_requests(upstream, task)
    assert used in offered(first)
    assert unused not in offered(first)
    assert last["messages"][-1] == {"role": "tool", "tool_call_id": "sub_call", "content": "AHOY"}
    expect(delegated_row(page, task)).to_be_visible()


def sidebar_chats(page: Page) -> Locator:
    return page.locator('a[href^="/c/"]')


def test_the_subagents_chat_shows_in_no_ones_sidebar_or_search(
    subagents_on, page_for, make_user, upstream
):
    prompt, task = unique("hand this over"), unique("count the boats")
    upstream.queue(
        *delegation(prompt, task, reply.text("Three boats.", match=reply.answering(task)))
    )
    other_prompt = unique("a chat of my own")
    upstream.queue(reply.text("Noted.", match=reply.answering(other_prompt)))
    owner_page, other_page = page_for(make_user()), page_for(make_user())

    send(owner_page, prompt)
    expect_reply(owner_page, "The helper is done.")
    send(other_page, other_prompt)
    expect_reply(other_page, "Noted.")

    expect(sidebar_chats(owner_page)).to_have_count(1)
    owner_page.reload()
    expect(sidebar_chats(owner_page)).to_have_count(1)
    owner_search = open_search_from_sidebar(owner_page)
    search(owner_search, prompt)
    expect(result(owner_search, prompt)).to_be_visible()
    search(owner_search, task)
    expect(no_matches(owner_search)).to_be_visible()
    expect(sidebar_chats(other_page)).to_have_count(1)
    other_search = open_search_from_sidebar(other_page)
    search(other_search, task)
    expect(no_matches(other_search)).to_be_visible()


def share_with(page: Page, reader) -> str:
    """Create the chat's share link, grant `reader` read access and return the link."""
    page.get_by_role("button", name="Chat actions").first.click()
    page.get_by_role("menu").get_by_role("menuitem", name="Share").click()
    dialog = page.get_by_role("dialog").filter(has_text="Share Chat")
    dialog.get_by_role("button", name="Copy Link").click()
    link = dialog.get_by_role("link", name="You have shared this chat before")
    expect(link).to_be_visible()
    dialog.get_by_role("button", name="Add Access").click()
    picker = page.get_by_role("dialog").filter(has_text="Add Access").last
    # the picker lists one page of users; searching reaches any account
    picker.get_by_placeholder("Search").fill(reader.name)
    pick_member(picker, reader.name)
    picker.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Access updated")).to_be_visible()
    return link.get_attribute("href")


def test_a_chat_shared_by_link_shows_the_reader_the_delegation_and_its_answer(
    subagents_on, page_for, make_user, upstream
):
    prompt, task, answer = unique("hand this over"), unique("name the capital"), unique("Vienna")
    upstream.queue(*delegation(prompt, task, reply.text(answer, match=reply.answering(task))))
    owner_page = page_for(make_user())
    send(owner_page, prompt)
    expect_reply(owner_page, "The helper is done.")
    reader = make_user()
    path = share_with(owner_page, reader)

    reader_page = page_for(reader, path=path)

    expect(conversation(reader_page).get_by_text("The helper is done.")).to_be_visible()
    delegated_row(reader_page, task).click()
    expect(last_reply(reader_page).get_by_text(answer, exact=True)).to_be_visible()
