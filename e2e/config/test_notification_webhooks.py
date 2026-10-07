"""Journey: a user's notification targets and the admin's event webhooks, set up in the browser.

With the admin's User Webhooks switch on (and the permission, which admins always hold) a user adds
a target under Settings > Notifications: a URL, the events it receives and whether it fires always
or only when the user is away. The list shows the URL masked, Send Test calls the URL, a finished
reply calls every enabled target that subscribes to it (an away target stays quiet while the user
is on the page, a switched-off one stays quiet always) and with the admin switch off the section is
not offered and nothing is called. An admin adds an event webhook under Admin Settings > General,
and a new account then reaches its URL as a `user.created` event until it is switched off. The chat
link a finished reply sends must open that chat; it was rewritten from `/c/<id>` to `/<id>`, which
the frontend has no route for (404), until PR #31572 (open-webui/open-webui#31565). Loopback URLs
are only fetchable on an instance booted with local fetching allowed. Editing a target's URL
sends the next reply to the new one, a removed target leaves the list and is not called, and
Make Default moves the model's `notify` tool to that target. A target for failed chats is called
when a reply fails, as the notifications docs promise; on dev a provider error that ends a reply
announces nothing, so that test stays red until the failure is published
(open-webui/open-webui#32003).

Discriminates: passes on dev a5bc78300, and the chat link test fails on dev 176d31d1d, before PR
#31572; in frontend copies of 176d31d1d, with the target save sending no URL the save test fails,
with the Send Test call removed the test-button test fails, with the row switch saving nothing the
switched-off target test fails, with the section shown whatever the admin's switch says the admin
switch test fails and with the event webhook form always saving every event the two admin tests
fail; in a backend copy, with the away check skipped the on-the-page test fails (the away target is
called), with the enabled check skipped the switched-off target test fails and with the admin
switch ignored the switched-off test fails; on ebc6add67, in a frontend copy, with the edit form
saving no URL the edit test fails, with Remove deleting nothing the removal test fails and with
Make Default calling nothing the default test fails; in a backend copy that publishes
`chat.failed` when a reply ends in an error the failed-chat test passes.
"""

from __future__ import annotations

import time

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.actors import Actor, admin_of, create_user
from harness.listener import Listener, json_answer
from harness.web_retrieval import LOCAL_WEB_FETCH
from utils.chat_ui import expect_reply, send
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

ADMIN_CONFIG = "/api/v1/auths/admin/config"
PERMISSIONS = "/api/v1/users/default/permissions"
TARGETS = "/api/v1/notifications/targets"
SAVED = "Settings saved successfully!"
DELIVERY_TIMEOUT = 15.0


@pytest.fixture(scope="module")
def fetching_instance(instance_with):
    return instance_with(LOCAL_WEB_FETCH)


def _save_config(client, path: str, **changes) -> None:
    current = client.get(path)
    current.raise_for_status()
    saved = client.post(path, json={**current.json(), **changes})
    assert saved.status_code == 200, saved.text


@pytest.fixture
def user_webhooks(fetching_instance, preserve):
    """`user_webhooks(switched_on)` sets the admin's switch and lets users hold the permission."""
    preserve("admin_config", "permissions", on=fetching_instance)

    def configure(switched_on: bool = True) -> None:
        with admin_of(fetching_instance).client() as client:
            _save_config(client, ADMIN_CONFIG, ENABLE_USER_WEBHOOKS=switched_on)
            features = client.get(PERMISSIONS).json()["features"]
            _save_config(client, PERMISSIONS, features={**features, "webhooks": True})

    configure()
    return configure


@pytest.fixture
def hook(listener: Listener) -> Listener:
    for path in ("/hook", "/away", "/always", "/off", "/events"):
        listener.route("POST", path, json_answer({}))
    return listener


@pytest.fixture
def event_webhooks(fetching_instance, hook):
    """Removes the event webhooks a test pointed at its listener, so later tests never call it."""
    yield
    with admin_of(fetching_instance).client() as client:
        for webhook in client.get("/api/events/webhooks").json():
            if webhook["url"].startswith(hook.base_url):
                client.delete(f"/api/events/webhooks/{webhook['id']}")


def _add_target(account: Actor, target_id: str, url: str, **options) -> None:
    body = {"id": target_id, "config": {"url": url}, "events": ["chat.finished"], **options}
    with account.client() as client:
        created = client.post(TARGETS, json=body)
    assert created.status_code == 200, created.text


def _stored_targets(account: Actor) -> list[dict]:
    with account.client() as client:
        listed = client.get(TARGETS)
    listed.raise_for_status()
    return listed.json()["targets"]


def _stored_target(account: Actor, target_id: str) -> dict:
    return next(target for target in _stored_targets(account) if target["id"] == target_id)


def _wait_for(condition, timeout: float = DELIVERY_TIMEOUT) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _open_notifications(page: Page):
    page.get_by_role("button", name="User menu").first.click()
    page.get_by_role("menu").get_by_role("button", name="Settings").click()
    settings = page.get_by_role("dialog")
    settings.get_by_role("tab", name="Notifications").click()
    expect(settings.get_by_text("Browser Notifications", exact=True)).to_be_visible()
    return settings


def _reply_to(page: Page, instance, prompt: str, answer: str) -> None:
    instance.upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, answer)


def test_a_user_saves_a_target_and_sees_it_listed_with_a_masked_url(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    settings = _open_notifications(page_for(account))

    settings.get_by_role("button", name="Add Notification Target").click()
    form = settings.page.get_by_role("dialog").last
    form.get_by_placeholder("Target ID").fill("ops")
    form.get_by_placeholder("https://hooks.slack.com/services/...").fill(f"{hook.base_url}/hook")
    form.get_by_role("button", name="Chat finished", exact=True).click()
    form.get_by_role("button", name="Always", exact=True).click()
    form.get_by_role("button", name="Save", exact=True).click()

    expect(settings.page.get_by_text(SAVED)).to_be_visible()
    expect(settings.get_by_text("ops", exact=True)).to_be_visible()
    expect(settings.get_by_text(f"{hook.base_url}/hook")).to_have_count(0)
    [stored] = _stored_targets(account)
    assert (stored["id"], stored["events"], stored["delivery"]) == (
        "ops",
        ["chat.finished"],
        "always",
    )
    assert stored["enabled"] is True and "url" not in stored["config"]


def test_the_test_button_calls_the_saved_url(fetching_instance, user_webhooks, hook, page_for):
    account = create_user(fetching_instance)
    _add_target(account, "ops", f"{hook.base_url}/hook", events=[])
    settings = _open_notifications(page_for(account))

    settings.get_by_role("button", name="Send Test").click()

    expect(settings.page.get_by_text("Test notification sent.")).to_be_visible()
    assert _wait_for(lambda: hook.requests_to("/hook")), "the test button called nothing"
    assert hook.requests_to("/hook")[0].json() == {"action": "test", "user_id": account.id}


def test_a_finished_reply_calls_an_always_target_with_the_reply(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "ops", f"{hook.base_url}/hook", delivery="always")
    page = page_for(account)

    _reply_to(page, fetching_instance, "say the codeword", "the codeword is pelican")

    assert _wait_for(lambda: hook.requests_to("/hook")), "the finished reply called nothing"
    [delivered] = hook.requests_to("/hook")
    body = delivered.json()
    assert (body["action"], body["message"]) == ("chat", "the codeword is pelican")


def test_the_chat_link_in_a_finished_reply_notification_opens_the_chat(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "ops", f"{hook.base_url}/hook", delivery="always")
    page = page_for(account)
    _reply_to(page, fetching_instance, "say the codeword", "the codeword is pelican")
    assert _wait_for(lambda: hook.requests_to("/hook")), "the finished reply called nothing"
    link = hook.requests_to("/hook")[0].json()["url"]

    page.goto(link)

    expect_reply(page, "the codeword is pelican")


def test_an_away_target_stays_quiet_while_the_user_is_on_the_page(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "away", f"{hook.base_url}/away", delivery="away")
    _add_target(account, "always", f"{hook.base_url}/always", delivery="always")
    page = page_for(account)

    _reply_to(page, fetching_instance, "say the codeword", "the codeword is heron")

    # targets are tried in order, so the away one has been decided when the last one is called
    assert _wait_for(lambda: hook.requests_to("/always")), "the always target was not called"
    assert hook.requests_to("/away") == []


def test_a_target_switched_off_in_the_list_is_not_called(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "off", f"{hook.base_url}/off", delivery="always")
    _add_target(account, "always", f"{hook.base_url}/always", delivery="always")
    page = page_for(account)
    settings = _open_notifications(page)

    row = settings.get_by_text("off", exact=True).locator(
        "xpath=ancestor::div[.//button[@role='switch']][1]"
    )
    row.get_by_role("switch", name="Enabled").click()

    assert _wait_for(lambda: _stored_target(account, "off")["enabled"] is False)
    page.keyboard.press("Escape")
    _reply_to(page, fetching_instance, "say the codeword", "the codeword is ibis")
    assert _wait_for(lambda: hook.requests_to("/always")), "the enabled target was not called"
    assert hook.requests_to("/off") == []


def test_an_admin_switching_user_webhooks_on_and_off_shows_and_hides_the_section(
    fetching_instance, user_webhooks, page_for
):
    user_webhooks(switched_on=False)
    admin = create_user(fetching_instance, role="admin")
    account = create_user(fetching_instance)
    admin_page = page_for(admin)
    admin_page.goto("/admin/settings/general")
    general = admin_page.get_by_role("dialog")

    general.get_by_role("switch", name="User Webhooks").click()
    general.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text(SAVED)).to_be_visible()
    user_page = page_for(account)
    offered = _open_notifications(user_page)
    expect(offered.get_by_text("Notification Targets", exact=True)).to_be_visible()

    general.get_by_role("switch", name="User Webhooks").click()
    general.get_by_role("button", name="Save", exact=True).click()
    expect(admin_page.get_by_text(SAVED).last).to_be_visible()
    user_page.reload()
    withdrawn = _open_notifications(user_page)
    expect(withdrawn.get_by_text("Notification Targets", exact=True)).to_have_count(0)
    expect(withdrawn.get_by_role("button", name="Add Notification Target")).to_have_count(0)


def test_with_user_webhooks_off_a_finished_reply_calls_nothing(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "ops", f"{hook.base_url}/hook", delivery="always")
    user_webhooks(switched_on=False)
    page = page_for(account)

    _reply_to(page, fetching_instance, "say the first word", "the first word is kestrel")
    user_webhooks(switched_on=True)
    _reply_to(page, fetching_instance, "say the second word", "the second word is osprey")

    assert _wait_for(lambda: hook.requests_to("/hook")), (
        "the reply after switching on called nothing"
    )
    assert [call.json()["message"] for call in hook.requests_to("/hook")] == [
        "the second word is osprey"
    ]


def _webhook_enabled(client, name: str) -> bool:
    [webhook] = [w for w in client.get("/api/events/webhooks").json() if w["name"] == name]
    return webhook["enabled"]


def _add_event_webhook(page: Page, name: str, url: str, pattern: str) -> None:
    page.goto("/admin/settings/general")
    general = page.get_by_role("dialog")
    expect(general.get_by_text("Send product events as JSON")).to_be_visible()
    tooltip_button(general, "Add webhook").click()
    form = page.get_by_role("dialog").last
    form.get_by_label("Name").fill(name)
    form.get_by_label("URL").fill(url)
    form.get_by_label("All events").uncheck()
    form.get_by_placeholder("Search or add pattern").fill(pattern)
    form.get_by_placeholder("Search or add pattern").press("Enter")
    form.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Webhook saved")).to_be_visible()


def test_an_admin_adds_an_event_webhook_and_a_new_account_reaches_it(
    fetching_instance, hook, event_webhooks, page_for
):
    admin = create_user(fetching_instance, role="admin")
    page = page_for(admin)

    _add_event_webhook(page, "Sign-ups audit", f"{hook.base_url}/events", "user.created")
    expect(page.get_by_text("Sign-ups audit")).to_be_visible()
    account = create_user(fetching_instance)

    assert _wait_for(lambda: hook.requests_to("/events")), "the new account called nothing"
    body = hook.requests_to("/events")[0].json()
    assert (body["event"], body["subject"]["id"]) == ("user.created", account.id)
    assert body["data"] == {"role": "user"}


def test_an_event_webhook_switched_off_in_the_list_is_not_called(
    fetching_instance, hook, event_webhooks, page_for
):
    admin = create_user(fetching_instance, role="admin")
    page = page_for(admin)
    _add_event_webhook(page, "Paused audit", f"{hook.base_url}/events", "user.created")
    row = page.get_by_text("Paused audit", exact=True).locator(
        "xpath=ancestor::div[.//button[@role='switch']][1]"
    )

    row.get_by_role("switch").click()
    with admin_of(fetching_instance).client() as client:
        assert _wait_for(lambda: _webhook_enabled(client, "Paused audit") is False)
    create_user(fetching_instance)
    row.get_by_role("switch").click()
    second = create_user(fetching_instance)

    assert _wait_for(lambda: hook.requests_to("/events")), "the enabled webhook was not called"
    assert [call.json()["subject"]["id"] for call in hook.requests_to("/events")] == [second.id]


def _target_row(settings, target_id: str):
    return settings.get_by_text(target_id, exact=True).locator(
        "xpath=ancestor::div[.//button[@role='switch']][1]"
    )


def test_editing_a_target_sends_the_next_reply_to_its_new_url(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "ops", f"{hook.base_url}/hook", delivery="always")
    page = page_for(account)
    settings = _open_notifications(page)

    _target_row(settings, "ops").get_by_role("button", name="Edit").click()
    form = page.get_by_role("dialog").last
    form.get_by_placeholder("Keep current webhook URL").fill(f"{hook.base_url}/always")
    form.get_by_role("button", name="Save", exact=True).click()

    expect(page.get_by_text(SAVED)).to_be_visible()
    expect(_target_row(settings, "ops")).to_contain_text("/...ways")
    page.keyboard.press("Escape")
    _reply_to(page, fetching_instance, "say the codeword", "the codeword is plover")
    assert _wait_for(lambda: hook.requests_to("/always")), "the edited URL was not called"
    assert hook.requests_to("/hook") == []


def test_a_removed_target_leaves_the_list_and_is_not_called(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "gone", f"{hook.base_url}/off", delivery="always")
    _add_target(account, "always", f"{hook.base_url}/always", delivery="always")
    page = page_for(account)
    settings = _open_notifications(page)

    _target_row(settings, "gone").get_by_role("button", name="Remove").click()

    expect(settings.get_by_text("gone", exact=True)).to_have_count(0)
    assert [target["id"] for target in _stored_targets(account)] == ["always"]
    page.keyboard.press("Escape")
    _reply_to(page, fetching_instance, "say the codeword", "the codeword is curlew")
    assert _wait_for(lambda: hook.requests_to("/always")), "the remaining target was not called"
    assert hook.requests_to("/off") == []


def test_a_target_for_failed_chats_hears_of_a_reply_that_failed(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(
        account, "failures", f"{hook.base_url}/hook", events=["chat.failed"], delivery="always"
    )
    _add_target(account, "always", f"{hook.base_url}/always", delivery="always")
    page = page_for(account)
    settings = _open_notifications(page)
    expect(_target_row(settings, "failures")).to_contain_text("Chat failed")
    page.keyboard.press("Escape")

    prompt = "say the codeword"
    fetching_instance.upstream.queue(
        reply.error(500, "the provider is down", match=reply.answering(prompt))
    )
    send(page, prompt)
    expect(page.get_by_text("the provider is down")).to_be_visible()

    # the docs promise chat.failed for a failed response; on dev only a finished one is announced
    assert _wait_for(lambda: hook.requests_to("/hook")), (
        "the failed reply called nothing (open-webui/open-webui#32003)"
    )
    body = hook.requests_to("/hook")[0].json()
    assert body["action"] == "chat_failed" and "the provider is down" in body["message"]
    assert hook.requests_to("/always") == []


def test_make_default_sends_the_models_notification_to_that_target(
    fetching_instance, user_webhooks, hook, page_for
):
    account = create_user(fetching_instance)
    _add_target(account, "first", f"{hook.base_url}/hook", events=[])
    _add_target(account, "pager", f"{hook.base_url}/always", events=[])
    page = page_for(account)
    settings = _open_notifications(page)
    expect(_target_row(settings, "first")).to_contain_text("Default")

    _target_row(settings, "pager").get_by_role("button", name="Make Default").click()

    expect(_target_row(settings, "pager").get_by_text("Default", exact=True)).to_be_visible()
    expect(
        _target_row(settings, "first").get_by_role("button", name="Make Default")
    ).to_be_visible()
    page.keyboard.press("Escape")
    prompt = "ping me when the kettle is on"
    fetching_instance.upstream.queue(
        reply.tool_call("notify", {"message": "kettle is on"}, match=reply.answering(prompt)),
        reply.text("I sent it."),
    )
    send(page, prompt)
    expect_reply(page, "I sent it.")
    assert _wait_for(lambda: hook.requests_to("/always")), "the default target was not called"
    assert hook.requests_to("/always")[0].json()["message"] == "kettle is on"
    assert hook.requests_to("/hook") == []
