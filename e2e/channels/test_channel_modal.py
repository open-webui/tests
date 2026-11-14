"""Journey: people create, rename and delete channels and give them webhooks from the sidebar.

The Channels section's plus button opens the Create Channel modal. A group channel made there is
private, lists in the owner's sidebar, opens at once and takes messages, and someone who was not let
in is sent home when they open it. A direct message made there with a person picked by name reaches
that person in their own sidebar. "Edit channel" in an owned channel's options menu opens Edit
Channel: a new name shows in the sidebar and the navbar for the owner and, after a reload, for
another member. Delete asks for a confirmation, then the channel leaves the owner's sidebar and a
member who opens its address is sent home. The Webhooks modal makes a webhook, copies its
address, and posting to that address shows the message to a member under the webhook's name;
deleting the webhook makes the address refuse further posts. A post reaches a member who has
the channel open without a reload; renaming the webhook in the modal shows its earlier posts
under the new name, and once it is deleted they read "Deleted Webhook". A webhook rename typed
but not yet saved stays in the open Webhooks window when a message arrives in the channel.

Discriminates: passes on dev 30f3f6a8f; in a frontend copy, a create form that sends a fixed name
turns the group test red, a direct message created without its picked person turns the direct
message test red, an update that sends the old name turns the rename test red, a delete that skips
its request turns the delete test red, and a webhook delete that skips its request turns the webhook
test red; in a backend copy, listing every webhook post under a fixed name turns the renaming test
red (checked on dev ebc6add67). The unsaved rename test passes with the fix for
open-webui/open-webui#33775 and fails on dev c165ee14f, where the arriving message resets the
window; the webhook test fails there too, as the webhook's own post closes the selected webhook.

Retargeted for b704d0613: the gear became an options menu that also holds the mute choice.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

import httpx
import pytest
from playwright.sync_api import Locator, Page, expect

from harness.channel_quotes import enable_channels, group_channel, post_message
from utils.chat_ui import chat_input, send
from utils.clipboard import read_text
from utils.member_picker import pick_member
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CLIPBOARD = ["clipboard-read", "clipboard-write"]


@pytest.fixture
def channels_on(admin, preserve):
    preserve("admin_config")
    enable_channels(admin)


def _sidebar(page: Page) -> Locator:
    sidebar = page.get_by_role("navigation", name="Chat history")
    expect(sidebar).to_be_visible()
    open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
    if open_sidebar.is_visible():
        open_sidebar.click()
    return sidebar


def _entry(page: Page, name: str) -> Locator:
    toggle = _sidebar(page).get_by_role("button", name="Channels", exact=True)
    if toggle.get_attribute("aria-expanded") == "false":
        toggle.click()
    return (
        _sidebar(page)
        .locator("#sidebar-channels-content #sidebar-channel-item")
        .filter(has_text=name)
    )


def _open_channel(page: Page, channel_id: str) -> None:
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()


def _message(page: Page, text: str) -> Locator:
    return page.locator("[id^='message-']").filter(has_text=text).first


def _open_create_modal(page: Page) -> Locator:
    expect(chat_input(page)).to_be_visible()
    create = _sidebar(page).get_by_role("button", name="Create Channel")
    modal = page.get_by_role("dialog").filter(has_text="Create Channel")
    # a click while the sidebar is still opening can be lost; the next one opens the dialog
    for _ in range(3):
        create.click()
        try:
            expect(modal).to_be_visible(timeout=2_000)
            return modal
        except AssertionError:
            continue
    expect(modal).to_be_visible()
    return modal


def _open_edit_modal(page: Page, name: str) -> Locator:
    entry = _entry(page, name)
    entry.hover()
    # the menu's trigger wraps its button, and both carry the name
    entry.get_by_role("button", name="Channel options").first.click()
    page.get_by_role("menuitem", name="Edit channel").click()
    modal = page.get_by_role("dialog").filter(has_text="Edit Channel")
    expect(modal).to_be_visible()
    return modal


def _channel_id_named(actor, name: str) -> str:
    with actor.client() as client:
        listed = client.get("/api/v1/channels/")
    listed.raise_for_status()
    [channel] = [channel for channel in listed.json() if channel["name"] == name]
    return channel["id"]


def _name_of(actor, channel_id: str) -> str:
    with actor.client() as client:
        listed = client.get("/api/v1/channels/")
    listed.raise_for_status()
    [channel] = [channel for channel in listed.json() if channel["id"] == channel_id]
    return channel["name"]


def _sent_home(page: Page) -> None:
    expect(chat_input(page)).to_be_visible()
    expect(page).not_to_have_url(re.compile("/channels/"))


def test_a_group_channel_created_in_the_modal_opens_and_stays_closed_to_outsiders(
    channels_on, make_user, page_for
):
    owner, outsider = make_user(), make_user()
    letting_in = group_channel(owner, outsider)
    page = page_for(owner)
    modal = _open_create_modal(page)

    modal.get_by_placeholder("new-channel").fill("harbour watch")
    expect(modal.get_by_text("Only invited users can access")).to_be_visible()
    modal.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile("/channels/"))
    expect(_entry(page, "harbour-watch")).to_be_visible()
    send(page, "the lamp is lit")
    expect(_message(page, "the lamp is lit")).to_be_visible()
    outsider_page = page_for(outsider)
    outsider_page.goto(f"/channels/{_channel_id_named(owner, 'harbour-watch')}")
    _sent_home(outsider_page)
    expect(_entry(outsider_page, _name_of(owner, letting_in))).to_be_visible()
    expect(_entry(outsider_page, "harbour-watch")).to_have_count(0)


def test_a_direct_message_created_in_the_modal_reaches_the_person_picked(
    channels_on, make_user, page_for
):
    starter, other = make_user(), make_user()
    page = page_for(starter)
    modal = _open_create_modal(page)

    modal.get_by_role("combobox").filter(has_text="Direct Message").select_option(
        label="Direct Message"
    )
    modal.get_by_placeholder("Search").fill(other.name)
    pick_member(modal, other.name)
    modal.get_by_role("button", name="Create", exact=True).click()

    expect(page).to_have_url(re.compile("/channels/"))
    send(page, "are you coming to the quay?")
    expect(_message(page, "are you coming to the quay?")).to_be_visible()
    other_page = page_for(other)
    entry = _entry(other_page, starter.name)
    expect(entry).to_be_visible()
    entry.click()
    expect(_message(other_page, "are you coming to the quay?")).to_be_visible()


def test_a_renamed_channel_shows_its_new_name_to_the_owner_and_to_a_member(
    channels_on, make_user, page_for
):
    owner, member = make_user(), make_user()
    channel_id = group_channel(owner, member)
    page = page_for(owner)
    _open_channel(page, channel_id)
    entry = _entry(page, "channel-")
    expect(entry).to_have_count(1)
    modal = _open_edit_modal(page, "channel-")

    modal.get_by_placeholder("new-channel").fill("lighthouse crew")
    modal.get_by_role("button", name="Update").click()

    expect(page.get_by_text("Channel updated successfully")).to_be_visible()
    expect(_entry(page, "lighthouse-crew")).to_be_visible()
    page.reload()
    expect(page.locator("nav").filter(has_text="lighthouse-crew")).to_be_visible()
    member_page = page_for(member)
    _open_channel(member_page, channel_id)
    expect(_entry(member_page, "lighthouse-crew")).to_be_visible()
    expect(member_page.locator("nav").filter(has_text="lighthouse-crew")).to_be_visible()


def test_a_deleted_channel_leaves_the_sidebar_and_sends_a_member_home(
    channels_on, make_user, page_for
):
    owner, member = make_user(), make_user()
    channel_id = group_channel(owner, member)
    page = page_for(owner)
    _open_channel(page, channel_id)
    modal = _open_edit_modal(page, "channel-")

    modal.get_by_role("button", name="Delete", exact=True).click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Delete", exact=True
    ).click()

    expect(page.get_by_text("Channel deleted successfully")).to_be_visible()
    _sent_home(page)
    expect(_entry(page, "channel-")).to_have_count(0)
    member_page = page_for(member, path=f"/channels/{channel_id}")
    _sent_home(member_page)
    expect(_entry(member_page, "channel-")).to_have_count(0)


def test_a_webhook_made_in_the_modal_posts_under_its_name_until_it_is_deleted(
    channels_on, make_user, page_for
):
    """Caught open-webui/open-webui#33775: the webhook's own post collapsed the open window."""
    owner, member = make_user(), make_user()
    channel_id = group_channel(owner, member)
    page = page_for(owner, permissions=CLIPBOARD)
    _open_channel(page, channel_id)
    edit = _open_edit_modal(page, "channel-")
    edit.get_by_role("button", name="Manage").click()
    webhooks = page.get_by_role("dialog").filter(has_text="New Webhook")

    webhooks.get_by_role("button", name="New Webhook").click()
    webhooks.get_by_placeholder("Webhook Name").fill("Tide Bot")
    webhooks.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()
    expect(webhooks.get_by_role("button", name="Tide Bot")).to_be_visible()
    tooltip_button(webhooks, "Copy URL").click()
    url = urljoin(owner.base_url, read_text(page))

    assert "/channels/webhooks/" in url
    posted = httpx.post(url, json={"content": "high tide at six"}, timeout=30)
    assert posted.status_code == 200, posted.text
    member_page = page_for(member)
    _open_channel(member_page, channel_id)
    message = _message(member_page, "high tide at six")
    expect(message).to_be_visible()
    expect(message).to_contain_text("Tide Bot")

    tooltip_button(webhooks, "Delete").click()
    page.get_by_role("dialog", name="Confirm your action").get_by_role(
        "button", name="Confirm", exact=True
    ).click()

    expect(page.get_by_text("Deleted", exact=True)).to_be_visible()
    refused = httpx.post(url, json={"content": "high tide again"}, timeout=30)
    assert refused.status_code == 401, refused.text
    post_message(owner, channel_id, "back to people")
    expect(_message(member_page, "back to people")).to_be_visible()
    expect(member_page.get_by_text("high tide again")).to_have_count(0)


def test_webhook_posts_arrive_live_and_follow_the_webhooks_name(channels_on, make_user, page_for):
    owner, member = make_user(), make_user()
    channel_id = group_channel(owner, member)
    with owner.client() as client:
        made = client.post(
            f"/api/v1/channels/{channel_id}/webhooks/create", json={"name": "Tide Bot"}
        )
    assert made.status_code == 200, made.text
    webhook = made.json()
    url = f"{owner.base_url}/api/v1/channels/webhooks/{webhook['id']}/{webhook['token']}"
    member_page = page_for(member)
    _open_channel(member_page, channel_id)

    assert httpx.post(url, json={"content": "low tide at noon"}, timeout=30).status_code == 200

    expect(_message(member_page, "low tide at noon")).to_contain_text("Tide Bot")
    page = page_for(owner)
    _open_channel(page, channel_id)
    _open_edit_modal(page, "channel-").get_by_role("button", name="Manage").click()
    webhooks = page.get_by_role("dialog").filter(has_text="New Webhook")
    webhooks.get_by_role("button", name="Tide Bot").click()
    webhooks.get_by_placeholder("Webhook Name").fill("Harbour Bot")
    webhooks.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()
    member_page.reload()
    expect(_message(member_page, "low tide at noon")).to_contain_text("Harbour Bot")

    with owner.client() as client:
        deleted = client.delete(f"/api/v1/channels/{channel_id}/webhooks/{webhook['id']}/delete")
    assert deleted.status_code == 200, deleted.text
    member_page.reload()
    expect(_message(member_page, "low tide at noon")).to_contain_text("Deleted Webhook")


def test_an_unsaved_webhook_rename_survives_a_message_arriving_in_the_channel(
    channels_on, make_user, page_for
):
    """Regression for open-webui/open-webui#33775.

    Every message arriving in the open channel reset the Webhooks window, so the selected webhook
    closed and a name typed but not yet saved was lost. Now the window keeps both until it closes.
    """
    owner, member = make_user(), make_user()
    channel_id = group_channel(owner, member)
    with owner.client() as client:
        made = client.post(
            f"/api/v1/channels/{channel_id}/webhooks/create", json={"name": "Tide Bot"}
        )
    assert made.status_code == 200, made.text
    page = page_for(owner)
    _open_channel(page, channel_id)
    _open_edit_modal(page, "channel-").get_by_role("button", name="Manage").click()
    webhooks = page.get_by_role("dialog").filter(has_text="New Webhook")
    webhooks.get_by_role("button", name="Tide Bot").click()
    webhooks.get_by_placeholder("Webhook Name").fill("Harbour Bot")

    post_message(member, channel_id, "ebb tide at nine")

    expect(_message(page, "ebb tide at nine")).to_be_attached()
    expect(webhooks.get_by_placeholder("Webhook Name")).to_have_value("Harbour Bot")
    webhooks.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Saved", exact=True)).to_be_visible()
    with owner.client() as client:
        listed = client.get(f"/api/v1/channels/{channel_id}/webhooks")
    assert [webhook["name"] for webhook in listed.json()] == ["Harbour Bot"], listed.text
