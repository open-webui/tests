"""Journey: what an admin saves in Admin Settings > General changes what every user gets.

The Features switches there decide whether Notes and Calendar are in a user's menu, whether the
sidebar has Channels and Folders and whether Settings has the Personalization tab of Memories.
Default Interface Settings sets the interface every account starts from, while an account's own
choice still wins. Memory System Context switched off keeps a person's saved memories out of what
the model is sent. A Response Watermark is appended to every reply a user copies. The Model
Response Mode picked for channels decides whether a model mentioned there answers in a thread or in
the channel itself.

Discriminates: passes on dev 30f3f6a8f; in a frontend build whose General form sends the stored
settings back in place of the edited ones, every test but the "own setting" one fails; in a backend
copy whose settings read lets the defaults win over the account's own, the "own setting" test
fails.
"""

from __future__ import annotations

import json
from typing import Callable

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.channel_chat import enable_channels, mention, model_reply
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
CLIPBOARD = ["clipboard-read", "clipboard-write"]
QUESTION = "Is the harbour light on?"
ANSWER = "The harbour light is on."


def open_general_settings(page: Page) -> Locator:
    page.goto("/admin/settings/general")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Community Sharing")).to_be_visible()
    return settings


def set_switch(scope: Locator, name: str, turn_on: bool) -> None:
    switch = scope.get_by_role("switch", name=name, exact=True)
    if (switch.get_attribute("aria-checked") == "true") != turn_on:
        switch.click()
    expect(switch).to_have_attribute("aria-checked", "true" if turn_on else "false")


def save(settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(settings.page.get_by_text("Settings saved successfully!")).to_be_visible()


def set_features(admin, **switches: bool) -> None:
    with admin.client() as client:
        current = client.get(ADMIN_CONFIG).json()
        saved = client.post(ADMIN_CONFIG, json={**current, **switches})
    saved.raise_for_status()


def user_menu_link(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        expect(chat_input(page)).to_be_visible()
        page.get_by_role("button", name="User menu").first.click()
        menu = page.get_by_role("menu")
        expect(menu.get_by_role("button", name="Settings")).to_be_visible()
        return menu.get_by_role("link", name=name, exact=True)

    return locate


def sidebar_section(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        expect(chat_input(page)).to_be_visible()
        open_sidebar = page.get_by_role("button", name="Open Sidebar", exact=True)
        if open_sidebar.is_visible():
            open_sidebar.click()
        sidebar = page.get_by_role("navigation", name="Chat history")
        expect(sidebar.get_by_role("button", name="Chats", exact=True)).to_be_visible()
        return sidebar.get_by_role("button", name=name, exact=True)

    return locate


def settings_tab(name: str) -> Callable[[Page], Locator]:
    def locate(page: Page) -> Locator:
        page.goto("/?settings=general")
        dialog = page.get_by_role("dialog")
        expect(dialog.get_by_role("tab").first).to_be_visible()
        return dialog.get_by_role("tab", name=name, exact=True)

    return locate


FEATURES = {
    "Notes": ("ENABLE_NOTES", user_menu_link("Notes")),
    "Calendar": ("ENABLE_CALENDAR", user_menu_link("Calendar")),
    "Channels": ("ENABLE_CHANNELS", sidebar_section("Channels")),
    "Folders": ("ENABLE_FOLDERS", sidebar_section("Folders")),
    "Memories": ("ENABLE_MEMORIES", settings_tab("Personalization")),
}


@pytest.mark.parametrize("feature", FEATURES)
def test_a_feature_switched_on_is_offered_to_a_user(feature, page_for, admin, make_user, preserve):
    preserve("admin_config")
    setting, entry = FEATURES[feature]
    set_features(admin, **{setting: False})
    settings = open_general_settings(page_for(admin))
    set_switch(settings, feature, turn_on=True)
    save(settings)

    page = page_for(make_user())

    expect(entry(page)).to_be_visible()


@pytest.mark.parametrize("feature", FEATURES)
def test_a_feature_switched_off_is_withdrawn_from_a_user(
    feature, page_for, admin, make_user, preserve
):
    preserve("admin_config")
    setting, entry = FEATURES[feature]
    set_features(admin, **{setting: True})
    settings = open_general_settings(page_for(admin))
    set_switch(settings, feature, turn_on=False)
    save(settings)

    page = page_for(make_user())

    expect(entry(page)).to_have_count(0)


def own_message_label(page: Page, text: str) -> Locator:
    return conversation(page).locator(".user-message").get_by_text(text, exact=True)


def chat_once(page: Page, upstream) -> None:
    upstream.queue(reply.text(ANSWER, match=reply.answering(QUESTION)))
    send(page, QUESTION)
    expect_reply(page, ANSWER)


def save_bubble_default(admin_page: Page, turn_on: bool) -> None:
    settings = open_general_settings(admin_page)
    settings.get_by_role("button", name="Default Interface Settings").click()
    set_switch(settings, "Chat Bubble UI", turn_on=turn_on)
    expect(settings.get_by_text("1 settings configured")).to_be_visible()
    save(settings)


def test_a_default_interface_setting_reaches_a_new_account(
    page_for, admin, make_user, upstream, preserve
):
    preserve("admin_config")
    save_bubble_default(page_for(admin), turn_on=False)

    page = page_for(make_user())
    chat_once(page, upstream)

    expect(own_message_label(page, "You")).to_be_visible()


def test_an_accounts_own_interface_setting_wins_over_the_default(
    page_for, admin, make_user, upstream, preserve
):
    preserve("admin_config")
    save_bubble_default(page_for(admin), turn_on=False)
    account = make_user()
    with account.client() as client:
        kept = client.post("/api/v1/users/user/settings/update", json={"ui": {"chatBubble": True}})
    kept.raise_for_status()

    page = page_for(account)
    chat_once(page, upstream)

    expect(conversation(page).locator(".user-message")).to_contain_text(QUESTION)
    expect(own_message_label(page, "You")).to_have_count(0)


def test_a_response_watermark_is_appended_to_a_copied_reply(
    page_for, admin, make_user, upstream, preserve
):
    preserve("admin_config")
    watermark = "Checked by the harbour office."
    settings = open_general_settings(page_for(admin))
    settings.get_by_placeholder("Enter a watermark for the response. Leave empty for none.").fill(
        watermark
    )
    save(settings)
    page = page_for(make_user(), permissions=CLIPBOARD)
    chat_once(page, upstream)

    reply_message = page.locator("[id^='message-']:has(.chat-assistant)").last
    reply_message.hover()
    reply_message.get_by_role("button", name="Copy", exact=True).click()

    expect(page.get_by_text("Copying to clipboard was successful!")).to_be_visible()
    expect(last_reply(page)).not_to_contain_text(watermark)
    copied = page.evaluate("navigator.clipboard.readText()")
    assert copied.startswith(ANSWER), copied
    assert copied.rstrip().endswith(watermark), copied


@pytest.mark.parametrize(("mode", "other"), [("Thread", "channel"), ("Channel", "thread")])
def test_the_model_response_mode_decides_where_a_model_answers_in_a_channel(
    mode, other, page_for, admin, make_user, upstream, preserve
):
    preserve("admin_config")
    with admin.client() as client:
        enable_channels(client, reply_mode=other)
    settings = open_general_settings(page_for(admin))
    settings.get_by_role("combobox", name="Model Response Mode").select_option(label=mode)
    save(settings)
    question = "How far is it to the island?"
    upstream.queue(reply.text("About six nautical miles.", match=reply.answering(question)))

    with make_user(role="admin").client() as client:
        channel_id, message_id = mention(client, reply.MOCK_MODEL_ID, question)
        answer = model_reply(client, channel_id)

    assert answer["content"] == "About six nautical miles."
    assert answer.get("parent_id") == (message_id if mode == "Thread" else None)


def test_memory_system_context_switched_off_keeps_memories_out_of_the_chat(
    page_for, admin, make_user, upstream, preserve
):
    preserve("admin_config")
    set_features(admin, ENABLE_MEMORIES=True, ENABLE_MEMORY_SYSTEM_CONTEXT=True)
    memory = "Keeps two rowing boats at the north quay."
    account = make_user()
    with account.client() as client:
        added = client.post("/api/v1/memories/add", json={"content": memory, "type": "user"})
    added.raise_for_status()
    page = page_for(account)
    chat_once(page, upstream)
    sent = next(filter(reply.answering(QUESTION), upstream.chat_requests()))
    assert memory in json.dumps(sent["messages"])

    settings = open_general_settings(page_for(admin))
    set_switch(settings, "Memory System Context", turn_on=False)
    save(settings)
    upstream.reset()
    page.goto("/")
    chat_once(page, upstream)

    sent = next(filter(reply.answering(QUESTION), upstream.chat_requests()))
    assert memory not in json.dumps(sent["messages"])
