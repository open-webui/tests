"""Journey: a model in a chat looks up what was said in the person's channels.

A member asks in an ordinary chat; the model searches the channel messages and the chat shows that
it did, and the search finds a thread reply in a channel the member belongs to. Someone who is not
in the channel asks the same and the search finds nothing.

Discriminates: passes on dev ebc6add67; in a backend copy, a message search that leaves out thread
replies turns the member test red and one that ignores membership turns the outsider test red.
"""

from __future__ import annotations

import json
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.channel_chat import enable_channels
from harness.channel_quotes import group_channel, post_message
from utils.chat_ui import expect_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


@pytest.fixture
def herons(admin, preserve, make_user):
    """A group channel with a thread about herons, its member and an outsider; and a unique tag."""
    preserve("admin_config")
    with admin.client() as client:
        enable_channels(client)
    member, other_member, outsider = make_user(), make_user(), make_user()
    tag = uuid.uuid4().hex[:8]
    channel_id = group_channel(member, other_member)
    parent_id = post_message(member, channel_id, f"birds at the jetty {tag}")
    post_message(other_member, channel_id, f"two grey herons {tag}", parent_id=parent_id)
    return member, outsider, tag


def _ask_about(page_for, account, upstream, tag: str) -> list:
    """Ask in a new chat, with the model searching the channels; returns what the search found."""
    question = f"what birds did the crew see? {tag}"
    upstream.queue(
        reply.tool_call("search_channel_messages", {"query": tag}, match=reply.answering(question)),
        reply.text(f"Here is what I found about {tag}", match=reply.answering(question)),
    )
    page = page_for(account)
    send(page, question)
    expect_reply(page, f"Here is what I found about {tag}")
    expect(
        page.get_by_text("search_channel_messages").locator("visible=true").first
    ).to_be_visible()
    [follow_up] = [
        request
        for request in upstream.chat_requests()
        if question in str(request)
        and any(entry["role"] == "tool" for entry in request["messages"])
    ]
    [found] = [entry["content"] for entry in follow_up["messages"] if entry["role"] == "tool"]
    return json.loads(found)


def test_the_model_finds_a_thread_reply_in_the_members_channel(herons, page_for, upstream):
    member, _, tag = herons

    found = _ask_about(page_for, member, upstream, tag)

    assert f"two grey herons {tag}" in json.dumps(found)


def test_the_model_finds_nothing_for_someone_outside_the_channel(herons, page_for, upstream):
    _, outsider, tag = herons

    found = _ask_about(page_for, outsider, upstream, tag)

    assert found == []
