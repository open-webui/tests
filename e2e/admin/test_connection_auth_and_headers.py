"""Journey: how a connection the admin sets up signs a user's chats, and what headers it adds.

The connection dialog's Auth choice decides what the provider is shown when a user chats: with
None no key at all, with Session the user's own Open WebUI session in place of a key. A custom
header under Advanced may name the user with a placeholder such as `{{USER_NAME}}`, filled in per
chat. Forward cookies (off by default) sends the cookies the browser holds for Open WebUI on to
the provider (docs: starting-with-openai-compatible, Forward cookies). The provider is a listener
the admin adds through the dialog; the user picks its model and gets its answer.

Discriminates: passes on the dev ebc6add67 build; in a backend copy, the None choice still sending
the key turns the None test red, the Session choice sending the key turns the Session test red,
custom headers left unfilled turn the None test red and cookies forwarded whatever the switch says
(or never) turns the Session test red.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.listener import Listener, json_answer
from harness.second_provider import OPENAI_CONFIG, sse
from utils.admin_connections import (
    OPENAI_URL_PLACEHOLDER,
    connection_dialog,
    connection_row,
    is_openai_save,
    open_admin_connections,
)
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import select_model
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

EVERYONE_READS = {"principal_type": "user", "principal_id": "*", "permission": "read"}
ANSWER = "signed as the admin chose"


@pytest.fixture
def prefix() -> str:
    return f"auth{uuid.uuid4().hex[:6]}"


@pytest.fixture
def provider(listener: Listener) -> Listener:
    listener.route("GET", "/v1/models", json_answer({"data": [{"id": "keeper"}]}))
    listener.route("POST", "/v1/chat/completions", sse({"content": ANSWER}))
    return listener


def publish(admin, model_id: str) -> None:
    """Make the model readable by every account, as the admin's visibility toggle does."""
    with admin.client() as client:
        client.get("/api/models", params={"refresh": True}).raise_for_status()
        shared = client.post(
            "/api/v1/models/model/access/update",
            json={"id": model_id, "name": model_id, "access_grants": [EVERYONE_READS]},
        )
    assert shared.status_code == 200, shared.text


def add_connection(page: Page, url: str, prefix: str, auth: str, headers: str = "") -> None:
    tooltip_button(open_admin_connections(page), "Add Connection").click()
    form = connection_dialog(page, "Add Connection")
    form.get_by_label("URL", exact=True).fill(url)
    form.get_by_role("textbox", name="API Key").fill("sk-admin-key")
    form.get_by_role("combobox", name="Auth").select_option(label=auth)
    form.get_by_role("button", name="Advanced").click()
    form.get_by_role("textbox", name="Prefix ID").fill(prefix)
    if headers:
        form.get_by_placeholder("Enter additional headers in JSON format").fill(headers)
    with page.expect_response(is_openai_save):
        form.get_by_role("button", name="Save").click()
    expect(form).to_be_hidden()


def chat_once(page: Page, model_id: str) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()
    select_model(page, model_id)
    send(page, f"Who signs this, {uuid.uuid4().hex[:6]}?")
    expect_reply(page, ANSWER)


def test_auth_none_sends_no_key_and_a_custom_header_names_the_user(
    page_for, make_user, admin, preserve, provider, prefix
):
    preserve(OPENAI_CONFIG)
    admin_page = page_for(make_user(role="admin"))
    add_connection(
        admin_page,
        f"{provider.base_url}/v1",
        prefix,
        auth="None",
        headers='{"X-Asker": "{{USER_NAME}}"}',
    )
    publish(admin, f"{prefix}.keeper")
    asker = make_user(name=f"Harbor Keeper {prefix}")

    chat_once(page_for(asker), f"{prefix}.keeper")

    [chat] = provider.requests_to("/v1/chat/completions")
    assert "Authorization" not in chat.headers, "a connection set to no auth sent a key"
    assert chat.headers.get("X-Asker") == asker.name


def test_session_auth_signs_with_the_users_session_and_cookies_follow_the_switch(
    page_for, make_user, admin, preserve, provider, prefix
):
    preserve(OPENAI_CONFIG)
    url = f"{provider.base_url}/v1"
    admin_page = page_for(make_user(role="admin"))
    add_connection(admin_page, url, prefix, auth="Session")
    publish(admin, f"{prefix}.keeper")
    asker = make_user()
    page = page_for(asker)
    page.context.add_cookies([{"name": "harbor_pref", "value": "blue", "url": asker.base_url}])

    chat_once(page, f"{prefix}.keeper")
    signed = provider.requests_to("/v1/chat/completions")[-1]
    assert signed.headers.get("Authorization") == f"Bearer {asker.token}"
    assert "harbor_pref" not in signed.headers.get("Cookie", ""), "cookies went out while off"

    settings = open_admin_connections(admin_page)
    tooltip_button(connection_row(settings, OPENAI_URL_PLACEHOLDER, url), "Configure").click()
    editing = connection_dialog(admin_page, "Edit Connection")
    editing.get_by_role("button", name="Advanced").click()
    editing.get_by_role("switch", name="Forward cookies").click()
    with admin_page.expect_response(is_openai_save):
        editing.get_by_role("button", name="Save").click()
    expect(editing).to_be_hidden()

    chat_once(page, f"{prefix}.keeper")
    forwarded = provider.requests_to("/v1/chat/completions")[-1]
    assert "harbor_pref=blue" in forwarded.headers.get("Cookie", "")
    assert forwarded.headers.get("Authorization") == f"Bearer {asker.token}"
