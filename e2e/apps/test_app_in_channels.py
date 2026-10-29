"""Journey: people use an app inside a channel, from inviting it to pressing its buttons.

A channel manager invites the app through Add Member in the channel's member list. People then
mention it, picked from the `@` list or typed by hand as "@<name>"; the app answers as an ordinary
message everyone sees under its name. When a mention cannot be delivered, the app's "App error:"
reply says why for everyone; only when the app cannot post (it was disabled) does the reason appear
under the asker's message marked "Only visible to you" for them alone. Buttons on an app message
call the app: a private answer, a download or an error shows only to whoever pressed it, an
acknowledgement shows nothing, and the result of longer work arrives later as an ordinary app
message.

An app can name a person as `<@U:id>`; the channel shows them by name, or as Unknown when the
account is gone.

Discriminates: passes on dev 4ca8fdc3f; in a frontend copy that drops both acknowledgement
checks (the button handler's and the reply list's), the private reply row the acknowledgement
then gets turns test_a_button_the_app_acknowledges_shows_nothing red. Either check alone holds.
Retargeted for dev c7193760b with oix 0.8, where a mention only gets a delivery receipt; passes
there except the `bare` cases of the two person mention tests: a mention written without a name
shows as `@U:<id>`, while `<@U:id|id>` resolves (#32362). Retargeted for dev b704d0613, where a
failed delivery becomes the app's public error reply; every test passes there, the `bare` cases
since 403f21ae7 (#32371) resolves a mention written without a name. With the backend's private
fallback sent to every channel member, test_a_mention_the_app_cannot_answer_tells_only_the_asker
turns red; with the public error reply never posted,
test_a_mention_that_cannot_be_delivered_shows_the_app_error_to_everyone turns red. On dev 226dca18c
(frontend built from 948432b43), a frontend copy that sends without resolving typed names turns the
`typed` case of test_a_mention_is_answered_for_everyone red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.apps import (
    connect,
    install_raw_app,
    installed,
    invite,
    new_channel,
    post,
    serving_raw_app,
    wait_for_message,
)
from utils.chat_ui import chat_input
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ANSWER_TIMEOUT_MS = 20_000


def _open_channel(page: Page, channel_id: str) -> None:
    page.goto(f"/channels/{channel_id}")
    expect(chat_input(page)).to_be_visible()


def _type_mention(
    page: Page, app_name: str, text: str, send: bool = True, picked: bool = True
) -> None:
    chat_input(page).click()
    if picked:
        page.keyboard.type("@" + app_name.split()[0])
        page.get_by_role("button", name=app_name).first.click()
        expect(chat_input(page)).to_contain_text("@" + app_name)
        page.keyboard.type(" " + text)
    else:
        page.keyboard.type(f"@{app_name} {text}")
    expect(chat_input(page)).to_contain_text(text)
    if send:
        page.keyboard.press("Enter")


def _private_reply(page: Page, app_id: str):
    return page.locator(f"[data-app-reply='{app_id}']")


def _channel_with_app(admin, app, **form) -> str:
    with admin.client() as client:
        channel_id = new_channel(client, **form)
        assert invite(client, channel_id, app["id"]).status_code == 200
    return channel_id


def _open_member_list(page: Page):
    page.get_by_role("button", name="User Count").click()
    members = page.get_by_role("dialog").filter(has_text="Members")
    expect(members.first).to_be_visible()
    return members.first


def test_a_manager_invites_and_removes_an_app_from_the_member_list(
    page_for, apps_admin, installed_app
):
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"name": "Invitable App"})
        channel_id = new_channel(client)
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    members = _open_member_list(page)
    members.get_by_role("button", name="Add Member").click()
    adding = page.get_by_role("dialog").filter(has_text="Add Members")
    adding.get_by_placeholder("Search").fill("Invitable")
    adding.get_by_label("Invitable App").click()
    adding.get_by_role("button", name="Add", exact=True).click()
    expect(page.get_by_text("Members added successfully")).to_be_visible()
    with apps_admin.client() as client:
        joined = client.get(f"/api/v1/channels/{channel_id}/apps").json()
    assert [app["id"] for app in joined] == [installed_app["id"]]
    expect(members.get_by_text("Invitable App")).to_be_visible()
    members.get_by_role("button", name="Remove Invitable App").click()
    expect(members.get_by_text("Invitable App")).to_have_count(0)
    with apps_admin.client() as client:
        assert client.get(f"/api/v1/channels/{channel_id}/apps").json() == []


def test_a_plain_member_sees_the_app_but_cannot_manage_it(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    page = page_for(member)
    _open_channel(page, channel_id)
    members = _open_member_list(page)
    expect(members.get_by_text(installed_app["name"])).to_be_visible()
    remove = members.get_by_role("button", name=f"Remove {installed_app['name']}")
    expect(remove).to_have_count(0)
    expect(members.get_by_role("button", name="Add Member")).to_have_count(0)


def test_the_app_answers_a_command_for_everyone_under_its_name(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    asker, watcher = page_for(apps_admin), page_for(member)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    chat_input(asker).click()
    asker.keyboard.type("!echo seen by all")
    asker.keyboard.press("Enter")
    for page in (asker, watcher):
        answer = page.get_by_text("echo: seen by all")
        expect(answer).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
        expect(page.get_by_text(installed_app["name"]).last).to_be_visible()


@pytest.mark.parametrize("picked", [True, False], ids=["picked", "typed"])
def test_a_mention_is_answered_for_everyone(
    page_for, apps_admin, make_apps_user, installed_app, reference_app, picked
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    asker, watcher = page_for(apps_admin), page_for(member)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    _type_mention(asker, installed_app["name"], "hello", picked=picked)
    for page in (asker, watcher):
        expect(page.get_by_text("hello").last).to_be_visible()
        answer = page.locator("[id^='message-']").filter(has_text="pong:").last
        expect(answer).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
        expect(answer).to_contain_text(installed_app["name"])
    reference_app.wait_for("mention", app_id=installed_app["id"], channel_id=channel_id)
    expect(_private_reply(asker, installed_app["id"])).to_have_count(0)


def test_a_mention_the_app_cannot_answer_tells_only_the_asker(
    page_for, apps_admin, make_apps_user, installed_app
):
    asker_account, watcher_account = make_apps_user(), make_apps_user()
    channel_id = _channel_with_app(
        apps_admin, installed_app, type="group", user_ids=[asker_account.id, watcher_account.id]
    )
    asker, watcher = page_for(asker_account), page_for(watcher_account)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    _type_mention(asker, installed_app["name"], "secret question", send=False)
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"is_active": False})
    asker.keyboard.press("Enter")
    reply = _private_reply(asker, installed_app["id"])
    expect(reply).to_contain_text("App is inactive", timeout=ANSWER_TIMEOUT_MS)
    expect(reply).to_contain_text("Only visible to you")
    expect(reply).not_to_contain_text("could not complete")
    expect(watcher.get_by_text("secret question")).to_be_visible()
    expect(_private_reply(watcher, installed_app["id"])).to_have_count(0)
    reply.get_by_role("button", name="Dismiss").click()
    expect(reply).to_have_count(0)


def test_a_mention_that_cannot_be_delivered_shows_the_app_error_to_everyone(
    page_for, apps_admin, make_apps_user, installed_app
):
    asker_account, watcher_account = make_apps_user(), make_apps_user()
    channel_id = _channel_with_app(
        apps_admin, installed_app, type="group", user_ids=[asker_account.id, watcher_account.id]
    )
    asker, watcher = page_for(asker_account), page_for(watcher_account)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    _type_mention(asker, installed_app["name"], "open question", send=False)
    with apps_admin.client() as client:
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"access_grants": []})
    asker.keyboard.press("Enter")
    for page in (asker, watcher):
        error = page.locator("[id^='message-']").filter(has_text="App invocation denied").last
        expect(error).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
        expect(error).to_contain_text("App error:")
        expect(error).to_contain_text(installed_app["name"])
    expect(_private_reply(asker, installed_app["id"])).to_have_count(0)


def test_a_public_mention_answer_shows_to_everyone(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    asker, watcher = page_for(apps_admin), page_for(member)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    _type_mention(asker, installed_app["name"], "public")
    expected = f"public answer for {apps_admin.id}"
    expect(asker.get_by_text(expected)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(watcher.get_by_text(expected)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)


def test_pressing_a_button_updates_the_message_for_everyone(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    with apps_admin.client() as client:
        post(client, channel_id, "!buttons")
        wait_for_message(client, channel_id, "Pick one")
    presser, watcher = page_for(member), page_for(apps_admin)
    _open_channel(presser, channel_id)
    _open_channel(watcher, channel_id)
    presser.get_by_role("button", name="Approve").click()
    expected = f"Approved by {member.id}"
    expect(presser.get_by_text(expected)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(watcher.get_by_text(expected)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(watcher.get_by_text("Pick one")).to_have_count(0)


def test_a_button_with_a_private_answer_tells_only_the_presser(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    with apps_admin.client() as client:
        post(client, channel_id, "!buttons")
        wait_for_message(client, channel_id, "Pick one")
    presser, watcher = page_for(member), page_for(apps_admin)
    _open_channel(presser, channel_id)
    _open_channel(watcher, channel_id)
    presser.get_by_role("button", name="Who am I").click()
    reply = _private_reply(presser, installed_app["id"])
    expect(reply).to_contain_text(f"you are {member.id} admin=False", timeout=ANSWER_TIMEOUT_MS)
    expect(_private_reply(watcher, installed_app["id"])).to_have_count(0)


def test_acknowledged_work_shows_nothing_until_its_result_arrives(
    page_for, apps_admin, make_apps_user, installed_app
):
    member = make_apps_user()
    channel_id = _channel_with_app(apps_admin, installed_app, type="group", user_ids=[member.id])
    asker, watcher = page_for(member), page_for(apps_admin)
    _open_channel(asker, channel_id)
    _open_channel(watcher, channel_id)
    _type_mention(asker, installed_app["name"], "later")
    result = f"Job finished for {member.name}"
    expect(asker.get_by_text(result)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(watcher.get_by_text(result)).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(_private_reply(asker, installed_app["id"])).to_have_count(0)


def test_a_button_the_app_acknowledges_shows_nothing(
    page_for, apps_admin, installed_app, reference_app
):
    channel_id = _channel_with_app(apps_admin, installed_app)
    with apps_admin.client() as client:
        post(client, channel_id, "!buttons")
        wait_for_message(client, channel_id, "Pick one")
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    page.get_by_role("button", name="Noted").click()
    reference_app.wait_for("action.noted", app_id=installed_app["id"], channel_id=channel_id)
    page.wait_for_timeout(1000)
    expect(_private_reply(page, installed_app["id"])).to_have_count(0)
    expect(page.get_by_role("alert")).to_have_count(0)


def _tools(page: Page, admin, channel_id: str) -> None:
    with admin.client() as client:
        post(client, channel_id, "!tools")
        wait_for_message(client, channel_id, "Tools")
    _open_channel(page, channel_id)


def test_a_download_answer_offers_its_file(page_for, apps_admin, installed_app):
    channel_id = _channel_with_app(apps_admin, installed_app)
    page = page_for(apps_admin)
    _tools(page, apps_admin, channel_id)
    page.get_by_role("button", name="Report", exact=True).click()
    reply = _private_reply(page, installed_app["id"])
    button = reply.get_by_role("button", name="Download weekly-report.md")
    expect(button).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    with page.expect_download() as download:
        button.click()
    assert download.value.suggested_filename == "weekly-report.md"


def test_an_error_answer_tells_the_presser_why(page_for, apps_admin, installed_app):
    channel_id = _channel_with_app(apps_admin, installed_app)
    page = page_for(apps_admin)
    _tools(page, apps_admin, channel_id)
    page.get_by_role("button", name="Fail", exact=True).click()
    reply = _private_reply(page, installed_app["id"])
    expect(reply).to_contain_text("It broke", timeout=ANSWER_TIMEOUT_MS)
    expect(reply).not_to_contain_text("could not complete")


def test_a_button_that_times_out_shows_an_error(page_for, apps_admin, installed_app):
    channel_id = _channel_with_app(apps_admin, installed_app)
    page = page_for(apps_admin)
    _tools(page, apps_admin, channel_id)
    page.get_by_role("button", name="Slow", exact=True).click()
    expect(page.get_by_role("alert").first).to_be_visible(timeout=ANSWER_TIMEOUT_MS)


def test_the_picker_offers_only_apps_in_this_channel(
    page_for, apps_admin, installed_app, reference_app
):
    with apps_admin.client() as client:
        outsider = installed(client, reference_app)
        client.patch(f"/api/v1/apps/{outsider['id']}", json={"name": "Outsider App"})
        client.patch(f"/api/v1/apps/{installed_app['id']}", json={"name": "Insider App"})
    channel_id = _channel_with_app(apps_admin, installed_app)
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    chat_input(page).click()
    page.keyboard.type("@")
    expect(page.get_by_role("button", name="Insider App")).to_be_visible()
    expect(page.get_by_role("button", name="Outsider App")).to_have_count(0)


def test_the_app_starts_a_thread_people_can_open(page_for, apps_admin, installed_app):
    channel_id = _channel_with_app(apps_admin, installed_app)
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    chat_input(page).click()
    page.keyboard.type("!thread written in a thread")
    page.keyboard.press("Enter")
    with apps_admin.client() as client:
        root = wait_for_message(client, channel_id, "!thread written in a thread")
        wait_for_message(client, channel_id, "written in a thread", parent_id=root["id"])
    parent = page.locator(f"[id='message-{root['id']}']").first
    parent.hover()
    tooltip_button(parent, "Reply in Thread").click()
    in_thread = page.locator(f"[id^='message-{root['id']}-']").filter(
        has_text="written in a thread"
    )
    expect(in_thread.first).to_be_visible()


@pytest.fixture
def raw_app_channel(apps_admin):
    with serving_raw_app() as raw:
        with apps_admin.client() as client:
            raw.installed = install_raw_app(client, raw)
            assert connect(client, raw.installed["id"]).status_code == 200
            raw.wait_until_ready()
            channel_id = new_channel(client)
            invite(client, channel_id, raw.installed["id"])
        yield raw, channel_id


@pytest.mark.parametrize("form", ["<@U:{id}>", "<@U:{id}|{id}>"], ids=["bare", "id-as-label"])
def test_a_person_an_app_mentions_by_id_is_shown_by_name(
    page_for, apps_admin, make_apps_user, raw_app_channel, form
):
    raw, channel_id = raw_app_channel
    mentioned = make_apps_user()
    content = f"over to {form.format(id=mentioned.id)} now"
    created = raw.request("create_message", channel_id=channel_id, content=content)
    assert created["status"] == 200, created
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    expect(page.get_by_text("over to")).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(page.locator(".mention").first).to_have_text(
        f"@{mentioned.name}", timeout=ANSWER_TIMEOUT_MS
    )


@pytest.mark.parametrize("form", ["<@U:{id}>", "<@U:{id}|{id}>"], ids=["bare", "id-as-label"])
def test_a_person_who_no_longer_exists_is_shown_as_unknown(
    page_for, apps_admin, raw_app_channel, form
):
    raw, channel_id = raw_app_channel
    missing = "00000000-0000-4000-8000-000000000000"
    content = f"ask {form.format(id=missing)}"
    created = raw.request("create_message", channel_id=channel_id, content=content)
    assert created["status"] == 200, created
    page = page_for(apps_admin)
    _open_channel(page, channel_id)
    expect(page.get_by_text("ask")).to_be_visible(timeout=ANSWER_TIMEOUT_MS)
    expect(page.locator(".mention").first).to_have_text("@Unknown", timeout=ANSWER_TIMEOUT_MS)
