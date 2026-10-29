"""Journey: an admin previews what an account can reach, changes its role and reads its chats.

In Admin Panel > Users the eye button on a row opens the account's access preview: the groups it
belongs to and the models and knowledge bases it can reach. A private model and a knowledge base
shared with a group show for a member and not for an account outside the group. The role button
opens the edit dialog, and setting the role to Pending there sends that account to the "Account
Activation Pending" screen on its next load. The Chats button lists the account's chats by title,
its search narrows them, and a chat opened from it shows the conversation to the admin. A group
named in the edit dialog links to that group's editor. Rotating the account's API key in the edit
dialog (360d81dfe, fdaa240d4) shows the new working key until the dialog closes, and deleting it
there clears the shown key and revokes it. Each test works as a fresh admin on accounts of its own.

Discriminates: passes on dev ebc6add67; in a frontend copy, the preview modal dropping the groups
and models it loaded turns both preview tests red, the edit dialog saving the stored role turns the
pending test red (the account stays in the chat), the chats modal sending no search query and
linking every chat to the home page turns the chats tests red, and the group names in the edit
dialog being plain text turns the group link test red. On dev 05e115366, a frontend copy keeping
the rotated key across closing the dialog and after a delete turns both API key tests red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.actors import Actor
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PAGE_TIMEOUT_MS = 30_000


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's settings."""
    return page_for(make_user(role="admin"))


@pytest.fixture
def crew(admin, make_user):
    """A group with one member, and an account outside it; the group is deleted afterwards."""
    member, outsider = make_user(), make_user()
    name = f"Crew {uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        created = client.post("/api/v1/groups/create", json={"name": name, "description": ""})
        assert created.status_code == 200, created.text
        group_id = created.json()["id"]
        added = client.post(
            f"/api/v1/groups/id/{group_id}/users/add", json={"user_ids": [member.id]}
        )
        assert added.status_code == 200, added.text
    yield member, outsider, name, group_id
    with admin.client() as client:
        client.delete(f"/api/v1/groups/id/{group_id}/delete")


def _group_grant(group_id: str) -> list[dict]:
    return [{"principal_type": "group", "principal_id": group_id, "permission": "read"}]


def _user_row(page: Page, account: Actor) -> Locator:
    page.goto("/admin/users")
    users = page.get_by_role("main")
    users.get_by_role("textbox", name="Search").fill(account.email)
    return users.get_by_role("row").filter(has_text=account.email)


def _preview(page: Page, account: Actor) -> Locator:
    _user_row(page, account).get_by_role("button", name="Preview Access").click()
    preview = page.get_by_role("dialog").filter(has_text="User Preview")
    expect(preview.get_by_text("Knowledge", exact=True)).to_be_visible()
    return preview


def _seed_chat(owner: Actor, title: str, answer: str) -> str:
    conversation = {
        "title": title,
        "history": {
            "currentId": "m2",
            "messages": {
                "m1": {
                    "id": "m1",
                    "parentId": None,
                    "childrenIds": ["m2"],
                    "role": "user",
                    "content": f"question for {title}",
                },
                "m2": {
                    "id": "m2",
                    "parentId": "m1",
                    "childrenIds": [],
                    "role": "assistant",
                    "content": answer,
                },
            },
        },
    }
    with owner.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": conversation})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _user_chats(page: Page, owner: Actor) -> Locator:
    _user_row(page, owner).get_by_role("button", name="Chats").click()
    chats = page.get_by_role("dialog").filter(has_text=f"{owner.name}'s Chats")
    expect(chats.get_by_text("Loading...")).to_have_count(0)
    return chats


@pytest.fixture
def shared_with_crew(admin, crew):
    """A private model and a knowledge base shared with the crew's group, by name."""
    _, _, _, group_id = crew
    suffix = uuid.uuid4().hex[:8]
    model_id, model_name, knowledge_name = (
        f"lighthouse-{suffix}",
        f"Lighthouse {suffix}",
        f"Charts {suffix}",
    )
    with admin.client() as client:
        model = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "name": model_name,
                "base_model_id": "mock-model",
                "meta": {},
                "params": {},
                "access_grants": _group_grant(group_id),
            },
        )
        assert model.status_code == 200, model.text
        knowledge = client.post(
            "/api/v1/knowledge/create",
            json={
                "name": knowledge_name,
                "description": "",
                "access_grants": _group_grant(group_id),
            },
        )
        assert knowledge.status_code == 200, knowledge.text
    yield model_name, knowledge_name
    with admin.client() as client:
        client.post("/api/v1/models/model/delete", json={"id": model_id})
        client.delete(f"/api/v1/knowledge/{knowledge.json()['id']}/delete")


def test_the_preview_lists_the_groups_and_what_a_group_shares_with_its_member(
    admin_page, crew, shared_with_crew
):
    member, _, group_name, _ = crew
    model_name, knowledge_name = shared_with_crew

    preview = _preview(admin_page, member)

    expect(preview).to_contain_text(member.name)
    expect(preview.get_by_text(group_name, exact=True)).to_be_visible()
    expect(preview.get_by_text(model_name, exact=True)).to_be_visible()
    expect(preview.get_by_text(knowledge_name, exact=True)).to_be_visible()


def test_the_preview_of_an_account_outside_the_group_leaves_out_what_the_group_shares(
    admin_page, crew, shared_with_crew
):
    member, outsider, group_name, _ = crew
    model_name, knowledge_name = shared_with_crew
    # the member's preview shows the model, so its absence below means something
    member_preview = _preview(admin_page, member)
    expect(member_preview.get_by_text(model_name, exact=True)).to_be_visible()
    member_preview.get_by_role("button").first.click()
    expect(member_preview).to_be_hidden()

    preview = _preview(admin_page, outsider)

    expect(preview).to_contain_text(outsider.name)
    expect(preview.get_by_text(group_name, exact=True)).to_have_count(0)
    expect(preview.get_by_text(model_name, exact=True)).to_have_count(0)
    expect(preview.get_by_text(knowledge_name, exact=True)).to_have_count(0)


def test_an_account_set_to_pending_meets_the_activation_screen_on_its_next_load(
    admin_page, page_for, make_user
):
    account = make_user()
    accounts_page = page_for(account)
    expect(chat_input(accounts_page)).to_be_visible(timeout=PAGE_TIMEOUT_MS)

    row = _user_row(admin_page, account)
    row.get_by_role("button", name="Change User Role").click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User")
    editing.get_by_role("combobox", name="Role").select_option(label="Pending")
    editing.get_by_role("button", name="Save").click()
    expect(row.get_by_role("button", name="Change User Role")).to_have_text("pending")

    accounts_page.reload()

    expect(accounts_page.get_by_text("Account Activation Pending")).to_be_visible(
        timeout=PAGE_TIMEOUT_MS
    )
    expect(chat_input(accounts_page)).to_be_hidden()


def test_the_chats_view_lists_an_accounts_chats_and_its_search_narrows_them(admin_page, make_user):
    owner = make_user()
    suffix = uuid.uuid4().hex[:8]
    harbour, orchard = f"Harbour plans {suffix}", f"Orchard notes {suffix}"
    _seed_chat(owner, harbour, "tides")
    _seed_chat(owner, orchard, "apples")

    chats = _user_chats(admin_page, owner)

    expect(chats.get_by_role("link", name=harbour)).to_be_visible()
    expect(chats.get_by_role("link", name=orchard)).to_be_visible()
    chats.get_by_role("textbox", name="Search Chats").fill("Orchard")
    expect(chats.get_by_role("link", name=harbour)).to_have_count(0)
    expect(chats.get_by_role("link", name=orchard)).to_be_visible()


def test_a_chat_opened_from_the_chats_view_shows_its_conversation_to_the_admin(
    admin_page, make_user
):
    owner = make_user()
    title = f"Harbour plans {uuid.uuid4().hex[:8]}"
    answer = f"The tide turns at {uuid.uuid4().hex[:6]}."
    chat_id = _seed_chat(owner, title, answer)

    chats = _user_chats(admin_page, owner)
    chats.get_by_role("link", name=title).click()

    expect(admin_page).to_have_url(re.compile(rf"/s/{chat_id}$"))
    expect(admin_page.get_by_text(f"question for {title}")).to_be_visible(timeout=PAGE_TIMEOUT_MS)
    expect(admin_page.get_by_text(answer)).to_be_visible()


def test_a_group_in_the_edit_dialog_links_to_that_groups_editor(admin_page, crew):
    member, _, group_name, group_id = crew
    row = _user_row(admin_page, member)
    with admin_page.expect_response(lambda response: response.url.endswith(f"/{member.id}/groups")):
        row.get_by_role("button", name="Edit User").click()
    editing = admin_page.get_by_role("dialog").filter(has_text="Edit User")

    editing.get_by_role("link", name=group_name).click()

    expect(admin_page).to_have_url(re.compile(rf"/admin/users/groups\?id={group_id}"))
    expect(admin_page.get_by_role("dialog").filter(has_text="Edit User Group")).to_be_visible()


@pytest.fixture
def api_keys_on(admin, preserve):
    preserve("admin_config", "permissions")
    with admin.client() as client:
        config = client.get("/api/v1/auths/admin/config").json()
        saved = client.post("/api/v1/auths/admin/config", json={**config, "ENABLE_API_KEYS": True})
        assert saved.status_code == 200, saved.text
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["features"]["api_keys"] = True
        saved = client.post("/api/v1/users/default/permissions", json=permissions)
        assert saved.status_code == 200, saved.text


def _edit_user(page: Page, account: Actor, row: Locator | None = None) -> Locator:
    row = row or _user_row(page, account)
    with page.expect_response(lambda response: response.url.endswith(f"/{account.id}/groups")):
        row.get_by_role("button", name="Edit User").click()
    editing = page.get_by_role("dialog").filter(has_text="Edit User")
    expect(editing.get_by_role("combobox", name="Role")).to_be_visible()
    return editing


def _rotate_in_dialog(page: Page, editing: Locator) -> str:
    editing.get_by_role("button", name="Rotate", exact=True).click()
    confirm = page.get_by_role("dialog", name="Rotate API Key?")
    with page.expect_response(lambda response: response.url.endswith("/api_key/rotate")) as rotated:
        confirm.get_by_role("button", name="Rotate", exact=True).click()
    assert rotated.value.status == 200, rotated.value.text()
    return rotated.value.json()["api_key"]


def _stored_key_status(account: Actor) -> int:
    with account.client() as client:
        return client.get("/api/v1/auths/api_key").status_code


def test_rotating_a_key_in_edit_user_shows_the_new_key_once(
    admin_page, make_user, instance, api_keys_on
):
    account = make_user()
    row = _user_row(admin_page, account)
    editing = _edit_user(admin_page, account, row)

    api_key = _rotate_in_dialog(admin_page, editing)

    expect(editing.get_by_role("textbox", name="Password", exact=True)).to_have_value(api_key)
    expect(editing.get_by_role("button", name="Copy API Key")).to_be_visible()
    with instance.client(api_key) as client:
        signed_in = client.get("/api/v1/users/user/status")
    assert signed_in.status_code == 200 and signed_in.json()["id"] == account.id, signed_in.text

    editing.get_by_role("button", name="Close").click()
    expect(editing).to_be_hidden()
    # reopened from the same row, without a reload that would reset the page
    reopened = _edit_user(admin_page, account, row)
    expect(reopened.get_by_role("button", name="Rotate", exact=True)).to_be_visible()
    expect(reopened.get_by_role("button", name="Copy API Key")).to_have_count(0)
    expect(reopened.get_by_role("textbox", name="Password", exact=True)).to_have_count(0)


def test_deleting_a_key_in_edit_user_clears_it_and_revokes_it(
    admin_page, make_user, instance, api_keys_on
):
    account = make_user()
    editing = _edit_user(admin_page, account)
    api_key = _rotate_in_dialog(admin_page, editing)
    expect(editing.get_by_role("textbox", name="Password", exact=True)).to_have_value(api_key)

    editing.get_by_role("button", name="Delete", exact=True).click()
    confirm = admin_page.get_by_role("dialog", name="Delete API Key?")
    with admin_page.expect_response(
        lambda response: response.request.method == "DELETE" and response.url.endswith("/api_key")
    ) as deleted:
        confirm.get_by_role("button", name="Delete", exact=True).click()

    assert deleted.value.status == 200, deleted.value.text()
    expect(editing.get_by_role("button", name="Copy API Key")).to_have_count(0)
    expect(editing.get_by_role("textbox", name="Password", exact=True)).to_have_count(0)
    assert _stored_key_status(account) == 404, "the account still holds a key"
    with instance.client(api_key) as client:
        assert client.get("/api/v1/users/user/status").status_code == 401
