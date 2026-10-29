"""Journey: two people hold a temporary end-to-end encrypted chat inside their direct message.

An admin turns on "Encrypted channels" in Admin Settings > General and the Encrypted Channels
feature permission in the default permissions; until both are on, a user's conversation menu has
no Encrypted Chat. One person picks Encrypted Chat from the direct message's options menu, the
other sees "Encrypted chat in progress" with a Join link in the conversation, and once both are in
what each writes shows on both screens, while the conversation itself keeps no copy. Each person's
verification code in Participants reads the same on both screens. A message over 16 KiB cannot be
sent. Reloading the page clears that device's chat and leaves the other person in it; when the
host leaves, the chat ends and the Join link goes away. In a private group of three, started from
the group's Channel options, all three see "3 participants" and everyone's messages.

Discriminates: passes on dev f5a26f53a. One frontend build with three edits turns three tests red
while the exchange and reload tests pass on it: a menu check that skips the permission (admin
test), a 32 KiB message limit (size test) and every code shown from the first participant's key
(verification test). In a backend copy, a relay that forwards no messages or saves each one as a
channel message turns the exchange test red, and one that keeps the session when the host leaves
turns the reload and leave test red. The three people test passes on dev 3b8205505; a relay that
sends a guest's messages only to the host turns it red while the exchange test passes.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.access import make_group
from harness.channel_quotes import enable_channels

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
DEFAULT_PERMISSIONS = "/api/v1/users/default/permissions"
PERMITTED = {"features": {"encrypted_channels": True}}
LIMIT_BYTES = 16 * 1024
ACTIVE = "Messages are end-to-end encrypted and aren’t saved."


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config", "permissions")
    enable_channels(admin)


@pytest.fixture
def pair(admin, make_user, channels_on):
    """Two people with the permission, encrypted chats on, and their direct message's id."""
    _switch_encrypted_chats(admin, turn_on=True)
    first, second = make_user(), make_user()
    make_group(admin, [first, second], PERMITTED)
    return first, second, _direct_message(first, second)


def _switch_encrypted_chats(admin, turn_on: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG)
        assert current.status_code == 200, current.text
        switched = {**current.json(), "ENABLE_ENCRYPTED_CHANNELS": turn_on}
        saved = client.post(ADMIN_CONFIG, json=switched)
    assert saved.status_code == 200, saved.text


def _withhold_permission_by_default(admin) -> None:
    with admin.client() as client:
        current = client.get(DEFAULT_PERMISSIONS)
        assert current.status_code == 200, current.text
        permissions = current.json()
        features = {**permissions["features"], "encrypted_channels": False}
        saved = client.post(DEFAULT_PERMISSIONS, json={**permissions, "features": features})
    assert saved.status_code == 200, saved.text


def _direct_message(actor, other) -> str:
    with actor.client() as client:
        opened = client.get(f"/api/v1/channels/users/{other.id}")
    assert opened.status_code == 200, opened.text
    return opened.json()["id"]


def _open_conversation(page: Page, channel_id: str) -> None:
    page.goto(f"/channels/{channel_id}")
    expect(page.locator("#chat-input")).to_be_visible()


def _conversation_menu(page: Page, name: str = "Conversation options") -> Locator:
    """The conversation's options menu in its header, opened ("Channel options" in a group)."""
    # the menu's trigger wraps its button, and both carry the name
    page.get_by_role("button", name=name).first.click()
    menu = page.get_by_role("menu")
    expect(menu.get_by_text("Notifications")).to_be_visible()
    return menu


def _start(page: Page, channel_id: str, menu: str = "Conversation options") -> None:
    _open_conversation(page, channel_id)
    _conversation_menu(page, menu).get_by_role("menuitem", name="Encrypted Chat").click()
    expect(page).to_have_url(re.compile(f"/channels/{channel_id}/encrypted$"))
    expect(page.get_by_text(ACTIVE)).to_be_visible()


def _join(page: Page, channel_id: str) -> None:
    _open_conversation(page, channel_id)
    page.get_by_role("button", name="Join encrypted chat").click()
    expect(page).to_have_url(re.compile(f"/channels/{channel_id}/encrypted$"))


def _messages(page: Page) -> Locator:
    return page.get_by_role("log", name="Encrypted messages")


def _write(page: Page, text: str) -> None:
    composer = page.locator("#chat-input")
    expect(composer).to_be_editable()
    composer.click()
    page.keyboard.insert_text(text)


def _send(page: Page, text: str) -> None:
    _write(page, text)
    page.get_by_role("button", name="Send message").click()


def _both_in(pair, page_for) -> tuple[Page, Page]:
    host, guest, channel_id = pair
    host_page, guest_page = page_for(host), page_for(guest)
    _start(host_page, channel_id)
    _join(guest_page, channel_id)
    expect(_messages(host_page).get_by_text(f"{guest.name} joined")).to_be_visible()
    expect(host_page.get_by_role("button", name="2 participants")).to_be_visible()
    expect(guest_page.get_by_role("button", name="2 participants")).to_be_visible()
    return host_page, guest_page


def _private_group(owner, *members) -> str:
    with owner.client() as client:
        created = client.post(
            "/api/v1/channels/create",
            json={
                "name": f"group-{uuid.uuid4().hex[:8]}",
                "type": "group",
                "is_private": True,
                "user_ids": [member.id for member in members],
            },
        )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _code_of(page: Page, name: str) -> str:
    """The verification code the Participants dialog shows for this person."""
    page.get_by_role("button", name=re.compile(r"\d+ participants")).click()
    dialog = page.get_by_role("dialog", name="Participants")
    member = dialog.get_by_role("group").filter(has_text=name)
    member.get_by_text(name, exact=True).click()
    code = member.get_by_role("code")
    expect(code).to_have_text(re.compile(r"\S{4}( \S{1,4})+"))
    text = code.inner_text()
    dialog.get_by_role("button", name="Close").click()
    expect(dialog).to_have_count(0)
    return text


def test_encrypted_chat_is_offered_once_the_admin_turns_it_on_and_grants_it(
    page_for, admin, make_user, channels_on
):
    _switch_encrypted_chats(admin, turn_on=False)
    _withhold_permission_by_default(admin)
    person, other = make_user(), make_user()
    channel_id = _direct_message(person, other)
    admin_page = page_for(admin, path="/admin/settings/general")

    settings = admin_page.get_by_role("dialog")
    switch = settings.get_by_role("switch", name="Encrypted channels", exact=True)
    switch.click()
    expect(switch).to_have_attribute("aria-checked", "true")
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text("Settings saved successfully!")).to_be_visible()
    page = page_for(person)
    _open_conversation(page, channel_id)
    expect(_conversation_menu(page).get_by_role("menuitem", name="Encrypted Chat")).to_have_count(0)

    admin_page.goto("/admin/users/groups")
    admin_page.get_by_role("button", name="Default permissions").click()
    dialog = admin_page.get_by_role("dialog")
    permission = dialog.get_by_role("switch", name="Encrypted Channels", exact=True)
    permission.click()
    expect(permission).to_have_attribute("aria-checked", "true")
    dialog.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Default permissions updated successfully")).to_be_visible()

    page = page_for(person)
    _start(page, channel_id)


def test_two_people_exchange_encrypted_messages_that_the_conversation_does_not_keep(pair, page_for):
    host, guest, channel_id = pair
    host_page, guest_page = _both_in(pair, page_for)

    _send(host_page, "the boat leaves at seven")
    expect(_messages(guest_page).get_by_text("the boat leaves at seven")).to_be_visible()
    _send(guest_page, "I will bring the oars")
    expect(_messages(host_page).get_by_text("I will bring the oars")).to_be_visible()
    expect(_messages(host_page).get_by_text("the boat leaves at seven")).to_be_visible()
    expect(_messages(guest_page).get_by_text("I will bring the oars")).to_be_visible()

    with host.client() as client:
        stored = client.get(f"/api/v1/channels/{channel_id}/messages")
    assert stored.status_code == 200, stored.text
    assert stored.json() == [], (
        f"the encrypted chat was saved into the conversation: {stored.json()}"
    )


def test_three_people_in_a_private_group_all_read_each_other(
    admin, make_user, channels_on, page_for
):
    _switch_encrypted_chats(admin, turn_on=True)
    host, first_guest, second_guest = make_user(), make_user(), make_user()
    make_group(admin, [host, first_guest, second_guest], PERMITTED)
    channel_id = _private_group(host, first_guest, second_guest)
    host_page = page_for(host)
    _start(host_page, channel_id, menu="Channel options")
    first_page = page_for(first_guest)
    _join(first_page, channel_id)
    expect(first_page.get_by_role("button", name="2 participants")).to_be_visible()
    second_page = page_for(second_guest)
    _join(second_page, channel_id)
    pages = (host_page, first_page, second_page)
    for page in pages:
        expect(page.get_by_role("button", name="3 participants")).to_be_visible()
    expect(_messages(first_page).get_by_text(f"{second_guest.name} joined")).to_be_visible()

    _send(host_page, "we meet at the bridge")
    _send(second_page, "I will be late")
    _send(first_page, "save me a seat")

    for page in pages:
        for line in ("we meet at the bridge", "I will be late", "save me a seat"):
            expect(_messages(page).get_by_text(line)).to_be_visible()
    with host.client() as client:
        stored = client.get(f"/api/v1/channels/{channel_id}/messages")
    assert stored.status_code == 200, stored.text
    assert stored.json() == [], f"the encrypted chat was saved into the group: {stored.json()}"


def test_each_persons_verification_code_reads_the_same_on_both_screens(pair, page_for):
    host, guest, _ = pair
    host_page, guest_page = _both_in(pair, page_for)

    host_seen_by_host, host_seen_by_guest = (
        _code_of(host_page, host.name),
        _code_of(guest_page, host.name),
    )
    guest_seen_by_host, guest_seen_by_guest = (
        _code_of(host_page, guest.name),
        _code_of(guest_page, guest.name),
    )

    assert host_seen_by_host == host_seen_by_guest, (host_seen_by_host, host_seen_by_guest)
    assert guest_seen_by_host == guest_seen_by_guest, (guest_seen_by_host, guest_seen_by_guest)
    assert host_seen_by_host != guest_seen_by_host, "both people were shown the same code"


def test_a_message_over_16_kib_cannot_be_sent(pair, page_for):
    host, _, channel_id = pair
    page = page_for(host)
    _start(page, channel_id)
    send_button = page.get_by_role("button", name="Send message")

    _write(page, "a" * LIMIT_BYTES)
    expect(send_button).to_be_enabled()
    expect(page.get_by_text("Message exceeds the 16 KiB limit.")).to_have_count(0)
    page.keyboard.insert_text("a")

    expect(page.get_by_text("Message exceeds the 16 KiB limit.")).to_be_visible()
    expect(send_button).to_be_disabled()


def test_a_reload_clears_that_device_and_the_host_leaving_ends_the_chat(pair, page_for):
    host, guest, channel_id = pair
    host_page, guest_page = _both_in(pair, page_for)
    _send(host_page, "before the reload")
    expect(_messages(guest_page).get_by_text("before the reload")).to_be_visible()

    guest_page.reload()

    expect(guest_page.get_by_text("Encrypted chat ended")).to_be_visible()
    expect(guest_page.get_by_text("before the reload")).to_have_count(0)
    expect(host_page.get_by_role("button", name="1 participants")).to_be_visible()
    guest_page.get_by_role("button", name="Return to conversation").click()
    expect(guest_page).to_have_url(re.compile(f"/channels/{channel_id}$"))
    join = guest_page.get_by_role("button", name="Join encrypted chat")
    expect(join).to_be_visible()

    host_page.get_by_role("button", name="Leave", exact=True).click()

    expect(host_page).to_have_url(re.compile(f"/channels/{channel_id}$"))
    expect(join).to_have_count(0)
