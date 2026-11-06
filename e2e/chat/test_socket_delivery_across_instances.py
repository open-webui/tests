"""A page whose socket and requests land on different instances shows all of it, on or off.

PR open-webui/open-webui#28818 (0.11.5) publishes each socket emit to one room on a Redis channel
of that room's own instead of the shared one; `WEBSOCKET_REDIS_ROOM_CHANNELS=false` puts every
emit back on the shared channel. Behind a load balancer with several workers a browser's socket
and its HTTP requests can reach different instances, so what a page shows crosses Redis even for
a single person.

Four instances share one database and one real Redis, two with the switch on and two with it off.
Per switch value one address splits its pair: the socket goes to the first instance and every
other request to the second. A chat sent in a page on that address streams its reply into the
page, and a tool asking the user through an event call shows its dialog there and gets the typed
answer back. Channel messages and note edits are seen live by a second person's page on the other
instance of the pair, each page on one instance directly. Every test runs in both modes and
expects the same thing on the screen.

Twin of integration/chat/test_socket_delivery_across_instances.py.

Discriminates: passes on dev 176d31d1d; with the room-channel listener deaf to room channels every
switched-on case fails and every switched-off case passes; with the stock manager deaf to other
instances every switched-off case fails and every switched-on case passes.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
from typing import Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import backends
from harness import upstream as reply
from harness.access import grant
from harness.actors import Actor, admin_of, create_user
from harness.channel_quotes import enable_channels, group_channel
from harness.instance import LaunchedInstance
from harness.python_tools import python_tool
from harness.socket_client import connected
from harness.split_proxy import splitting
from utils.cached_chat import turn_on_tool
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

# the instance an account's page uses and the one another person's page uses, per switch value
PAIRS = {"on": ("on-a", "on-b"), "off": ("off-a", "off-b")}
MODES = list(PAIRS)
ADMIN_CONFIG = "/api/v1/auths/admin/config"


@pytest.fixture(scope="module")
def shared_redis() -> Iterator[backends.RedisProcess]:
    server = backends.RedisProcess()
    server.start()
    yield server
    server.close()


@pytest.fixture(scope="module")
def fleet(shared_redis, instance_with) -> dict[str, LaunchedInstance]:
    """Four instances on one database and one Redis, by name: on-a, on-b, off-a and off-b."""
    shared = {"REDIS_URL": shared_redis.url, "WEBSOCKET_MANAGER": "redis"}
    first = instance_with({**shared, "WEBUI_NAME": "on-a"})  # the switch at its default
    joined = {**shared, "DATABASE_URL": first.database_url}
    instances = {
        "on-a": first,
        "on-b": instance_with(
            {**joined, "WEBUI_NAME": "on-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "true"}
        ),
        "off-a": instance_with(
            {**joined, "WEBUI_NAME": "off-a", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
        "off-b": instance_with(
            {**joined, "WEBUI_NAME": "off-b", "WEBSOCKET_REDIS_ROOM_CHANNELS": "false"}
        ),
    }
    # an instance listens on Redis only once a socket has connected; event calls need that
    for instance in instances.values():
        with connected(admin_of(instance)):
            pass
    return instances


@pytest.fixture(scope="module")
def split_addresses(fleet) -> Iterator[dict[str, str]]:
    """Per switch value, the base URL whose socket is on `<mode>-a` and requests on `<mode>-b`."""
    with contextlib.ExitStack() as opened:
        yield {
            mode: opened.enter_context(splitting(fleet[near], fleet[far]))
            for mode, (near, far) in PAIRS.items()
        }


@pytest.fixture(scope="module")
def channels_enabled(fleet) -> Iterator[None]:
    """Channels switched on through the admin; the admin config is put back after the module."""
    admin = admin_of(fleet["on-a"])
    with admin.client() as client:
        before = client.get(ADMIN_CONFIG)
        before.raise_for_status()
    enable_channels(admin)
    yield
    with admin.client() as client:
        client.post(ADMIN_CONFIG, json=before.json()).raise_for_status()


@pytest.fixture
def provider(fleet):
    """The provider every instance asks: the connection settings live in the shared database."""
    fleet["on-a"].upstream.reset()
    return fleet["on-a"].upstream


def at(account: Actor, base_url: str) -> Actor:
    """The same account reached at another address: they share the secret key and the database."""
    return dataclasses.replace(account, base_url=base_url)


def _tool_results(provider) -> list[str]:
    return [
        entry["content"]
        for sent in provider.chat_requests()
        for entry in reply.conversation(sent)
        if entry["role"] == "tool"
    ]


@pytest.mark.parametrize("mode", MODES)
def test_a_reply_streams_into_a_page_whose_socket_is_on_another_instance(
    fleet, split_addresses, provider, page_for, mode
):
    prompt = f"Where is Vienna? ({mode})"
    pieces = ["Vienna ", "lies ", "on the ", "Danube."]
    provider.queue(reply.text(pieces, match=reply.answering(prompt)))
    page = page_for(at(create_user(fleet[PAIRS[mode][0]]), split_addresses[mode]))

    send(page, prompt)

    expect_reply(page, "".join(pieces))


ASK_THE_PAGE = '''
import json


class Tools:
    async def ask_the_page(self, __event_call__=None) -> str:
        """
        Ask the user for their name.
        """
        answer = await __event_call__(
            {"type": "input", "data": {"title": "Your name?", "message": "For the greeting."}}
        )
        return json.dumps(answer)
'''


@pytest.mark.parametrize("mode", MODES)
def test_a_tool_asks_the_page_on_the_split_address_and_gets_the_typed_answer(
    fleet, split_addresses, provider, page_for, mode
):
    near, _far = PAIRS[mode]
    prompt = f"who am I? ({mode})"
    provider.queue(
        reply.tool_call("ask_the_page", {}, match=reply.answering(prompt)),
        reply.text("Nice to meet you, Ada.", match=reply.answering(prompt)),
    )
    with python_tool(admin_of(fleet[near]), ASK_THE_PAGE, name="Ask the page"):
        page = page_for(at(create_user(fleet[near]), split_addresses[mode]))
        turn_on_tool(page, "Ask the page")

        send(page, prompt)
        dialog = page.get_by_role("dialog", name="Your name?")
        dialog.get_by_label("Enter your message").fill("Ada")
        dialog.get_by_role("button", name="Confirm").click()
        expect_reply(page, "Nice to meet you, Ada.")
    assert _tool_results(provider) == [json.dumps("Ada")]


def _message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


@pytest.mark.parametrize("mode", MODES)
def test_a_channel_message_shows_live_in_a_page_on_the_other_instance(
    fleet, channels_enabled, page_for, mode
):
    near, far = PAIRS[mode]
    owner, member = create_user(fleet[near]), create_user(fleet[near])
    channel_id = group_channel(owner, member)
    pages = page_for(owner), page_for(at(member, fleet[far].base_url))
    for page in pages:
        page.goto(f"/channels/{channel_id}")
        expect(chat_input(page)).to_be_visible()
    owner_page, member_page = pages

    send(owner_page, f"the ferry leaves at nine ({mode})")

    expect(_message(member_page, f"the ferry leaves at nine ({mode})")).to_be_visible()


def _note_editor(page: Page) -> Locator:
    return page.get_by_role("main").get_by_label("Write something...")


@pytest.mark.parametrize("mode", MODES)
def test_text_typed_in_a_note_shows_live_in_the_page_on_the_other_instance(fleet, page_for, mode):
    near, far = PAIRS[mode]
    owner, writer = create_user(fleet[near]), create_user(fleet[near])
    with owner.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={
                "title": f"Packing list ({mode})",
                "data": {"content": {"md": "packing list"}},
                "access_grants": [
                    grant("user", writer.id, "read"),
                    grant("user", writer.id, "write"),
                ],
            },
        )
    assert created.status_code == 200, created.text
    note_url = f"/notes/{created.json()['id']}"
    owner_page, writer_page = page_for(owner), page_for(at(writer, fleet[far].base_url))
    for page in (owner_page, writer_page):
        page.goto(note_url)
        expect(_note_editor(page)).to_contain_text("packing list")

    _note_editor(owner_page).click()
    owner_page.keyboard.press("End")
    owner_page.keyboard.type(" with sunscreen")
    expect(_note_editor(writer_page)).to_contain_text("packing list with sunscreen")

    _note_editor(writer_page).click()
    writer_page.keyboard.press("End")
    writer_page.keyboard.type(" and a hat")
    expect(_note_editor(owner_page)).to_contain_text("packing list with sunscreen and a hat")
