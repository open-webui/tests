"""Journey: what other accounts meet in the browser as workspace items are shared and unshared.

- Public: an owner sets a model, prompt, tool and knowledge base to Public in its Access dialog,
  and a fresh user with no group finds each where it is used: the model in the model selector,
  the prompt under `/`, the tool under Integrations and the base under `#`.
- Revoked while open: the owner takes a reader's access away in the Access dialog while the
  reader holds the shared base attached in an unsent message, or a chat on a shared model. The
  message sent next carries none of the base's text and `#` no longer offers it, and the next
  message on the model is refused with an error. A reader who has the base's page open is told
  the file failed to load when they click it, and sent back to the list on a reload.
- Sharing switches: an admin switches a group's Knowledge Sharing on in the group editor, which
  brings up Knowledge Public Sharing below it; its member is then offered Add Access and Public on
  their own base while a user outside the group is not. An owner without public sharing whose
  base an admin made public can take it private but is not offered Public again.
- Base model: a model shared with everyone on a base model only admins may use is listed for a
  user, and a message to it is refused with an error and never reaches the base model; one on a
  base the user may read answers.

Discriminates: passes on dev 0f5a58f5f. A frontend build whose visibility select changes nothing
fails every public test and the public-to-private test (no change is saved), and one that shows
every Sharing switch whatever its parent fails the group switch test. A backend copy that skips the
model access checks on chat completions fails the revoked model and base model tests (the reply
arrives); one that keeps every attached base in retrieval fails the revoked base test (the model is
sent the text); one whose file access check always allows fails the open page test. The public,
group switch and public-to-private tests were retargeted for 784b72f19, whose access dialog picks
the visibility from a menu: they pass on dev 3dd1db147, a build whose visibility menu saves no
change fails the public tests, and one that always offers Public fails the group switch and
public-to-private tests.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Callable, Iterator

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.access import grant, make_group
from harness.actors import Actor
from harness.knowledge_bases import add_text_file
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.access_control import choose_visibility, visibility, visibility_choices
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_selector import model_options, select_model

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

TOOL_SOURCE = '''class Tools:
    def ping(self) -> str:
        """Answer pong."""
        return "pong"
'''
KNOWLEDGE_USER = {"workspace": {"knowledge": True}}


def _unique(label: str) -> str:
    return f"{label} {uuid.uuid4().hex[:6]}"


def _created(response) -> dict:
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def owner(make_user) -> Iterator[Actor]:
    """A fresh admin, whose models, prompts, tools and knowledge bases are deleted afterwards."""
    account = make_user(role="admin")
    yield account
    with account.client() as client:
        for model in client.get("/api/v1/models/list").json().get("items", []):
            if model["user_id"] == account.id:
                client.post("/api/v1/models/model/delete", json={"id": model["id"]})
        for prompt in client.get("/api/v1/prompts/").json():
            if prompt["user_id"] == account.id:
                client.delete(f"/api/v1/prompts/id/{prompt['id']}/delete")
        for tool in client.get("/api/v1/tools/").json():
            if tool["user_id"] == account.id:
                client.delete(f"/api/v1/tools/id/{tool['id']}/delete")
        for knowledge in client.get("/api/v1/knowledge/").json().get("items", []):
            if knowledge["user_id"] == account.id:
                client.delete(f"/api/v1/knowledge/{knowledge['id']}/delete")


def _model(owner: Actor, grants: list[dict], base_model_id: str = MOCK_MODEL_ID) -> dict:
    model_id = f"shared-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": base_model_id,
        "name": _unique("Pilot"),
        "meta": {},
        "params": {},
        "access_grants": grants,
    }
    with owner.client() as client:
        _created(client.post("/api/v1/models/create", json=form))
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
    return form


def _prompt(owner: Actor) -> dict:
    token = uuid.uuid4().hex[:8]
    form = {"command": f"tide{token}", "name": f"tide{token}", "content": "When is high tide?"}
    with owner.client() as client:
        return _created(client.post("/api/v1/prompts/create", json=form))


def _tool(owner: Actor) -> dict:
    form = {
        "id": f"tide_{uuid.uuid4().hex[:8]}",
        "name": _unique("Tide table"),
        "content": TOOL_SOURCE,
        "meta": {"description": "tides"},
    }
    with owner.client() as client:
        return _created(client.post("/api/v1/tools/create", json=form))


def _knowledge(owner: Actor, grants: list[dict] | None = None) -> dict:
    form = {"name": _unique("Harbour"), "description": "", "access_grants": grants or []}
    with owner.client() as client:
        return _created(client.post("/api/v1/knowledge/create", json=form))


def _access_dialog(page: Page) -> Locator:
    page.get_by_role("main").get_by_role("button", name="Access", exact=True).click()
    return page.get_by_role("dialog").filter(has_text="Access Control")


def _publish_model(page: Page, item: dict) -> None:
    page.goto(f"/workspace/models/edit?id={item['id']}")
    editor = page.get_by_role("main")
    expect(editor.get_by_placeholder("Model Name")).to_have_value(item["name"])
    choose_visibility(_access_dialog(page), "Public")
    page.keyboard.press("Escape")
    editor.get_by_role("button", name="Save & Update").click()
    expect(page).to_have_url(re.compile(r"/workspace/models/?$"))


def _publish_on_page(path: str) -> Callable[[Page, dict], None]:
    def publish(page: Page, item: dict) -> None:
        page.goto(path.format(**item))
        dialog = _access_dialog(page)
        choose_visibility(dialog, "Public")
        expect(page.get_by_text("Saved").first).to_be_visible()
        expect(dialog.get_by_text("Accessible to all users")).to_be_visible()

    return publish


def _new_chat(page: Page) -> None:
    page.goto("/")
    expect(chat_input(page)).to_be_visible()


def _typed_menu(page: Page, typed: str) -> Locator:
    _new_chat(page)
    chat_input(page).click()
    page.keyboard.type(typed)
    return page.get_by_role("tooltip")


def _model_offered(page: Page, item: dict) -> Locator:
    _new_chat(page)
    return model_options(page, item["name"])


def _prompt_offered(page: Page, item: dict) -> Locator:
    return _typed_menu(page, f"/{item['command']}").get_by_role("button", name=item["command"])


def _tool_offered(page: Page, item: dict) -> Locator:
    _new_chat(page)
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    return page.get_by_role("button", name=item["name"])


def _knowledge_offered(page: Page, item: dict) -> Locator:
    return _typed_menu(page, f"#{item['name'].split()[0]}").get_by_role("button", name=item["name"])


# kind: (create a private one, publish it in the browser, where a user finds it)
PUBLIC_KINDS = {
    "model": (lambda owner: _model(owner, []), _publish_model, _model_offered),
    "prompt": (_prompt, _publish_on_page("/workspace/prompts/{id}"), _prompt_offered),
    "tool": (_tool, _publish_on_page("/workspace/tools/edit?id={id}"), _tool_offered),
    "knowledge": (_knowledge, _publish_on_page("/workspace/knowledge/{id}"), _knowledge_offered),
}


@pytest.mark.parametrize("kind", PUBLIC_KINDS)
def test_an_item_made_public_in_its_access_dialog_reaches_any_user(
    kind, owner, make_user, page_for
):
    create, publish, offered = PUBLIC_KINDS[kind]
    item = create(owner)
    publish(page_for(owner), item)

    expect(offered(page_for(make_user()), item)).to_be_visible()


# ---------------------------------------------------------------- revoked while open


def _revoke_in_dialog(dialog: Locator, account: Actor) -> None:
    """Remove `account`'s row from the Access dialog."""
    page = dialog.page
    # the innermost box holding the name and a button is the account's row
    row = dialog.locator("div").filter(has_text=account.name).filter(has=page.get_by_role("button"))
    # the row's remove button has no label; it is the row's last button
    row.last.get_by_role("button").last.click()
    expect(dialog.get_by_text(account.name)).to_have_count(0)


def test_a_base_revoked_while_attached_sends_none_of_its_text(owner, make_user, page_for, upstream):
    reader = make_user()
    base = _knowledge(owner, [grant("user", reader.id, "read")])
    with owner.client() as client:
        add_text_file(client, base["id"], "gate.txt", "The harbour gate code is 4242.")
    reader_page = page_for(reader)
    _knowledge_offered(reader_page, base).click()
    expect(reader_page.get_by_role("button", name=base["name"])).to_be_visible()

    owner_page = page_for(owner)
    owner_page.goto(f"/workspace/knowledge/{base['id']}")
    with owner_page.expect_response(lambda response: "/access/update" in response.url) as saved:
        _revoke_in_dialog(_access_dialog(owner_page), reader)
    assert saved.value.ok

    question = "what is the harbour gate code?"
    upstream.queue(reply.text("I cannot tell.", match=reply.answering(question)))
    send(reader_page, question)
    expect_reply(reader_page, "I cannot tell.")
    sent = json.dumps(_asked(upstream, question))
    assert "4242" not in sent, "a chat still carried a base whose grant was taken away"
    expect(_knowledge_offered(reader_page, base)).to_have_count(0)


def test_a_base_revoked_while_its_page_is_open_no_longer_opens_its_files(
    owner, make_user, page_for
):
    reader = make_user()
    make_group(owner, [reader], KNOWLEDGE_USER)
    base = _knowledge(owner, [grant("user", reader.id, "read")])
    with owner.client() as client:
        add_text_file(client, base["id"], "gate.txt", "The harbour gate code is 4242.")
    reader_page = page_for(reader)
    reader_page.goto(f"/workspace/knowledge/{base['id']}")
    opened = reader_page.get_by_role("main")
    expect(opened.get_by_text("Read Only")).to_be_visible()

    with owner.client() as client:
        _created(
            client.post(f"/api/v1/knowledge/{base['id']}/access/update", json={"access_grants": []})
        )

    opened.get_by_role("button", name=re.compile(r"^gate\.txt(\s|$)")).click()
    expect(reader_page.get_by_text("Failed to load file content.")).to_be_visible()
    expect(reader_page.get_by_text("The harbour gate code is 4242.")).to_have_count(0)
    reader_page.reload()
    expect(
        reader_page.get_by_text("You do not have permission to access this resource")
    ).to_be_visible()
    expect(reader_page).to_have_url(re.compile(r"/workspace/knowledge/?$"))
    expect(reader_page.get_by_role("main").get_by_text(base["name"])).to_have_count(0)


def test_a_model_revoked_while_its_chat_is_open_refuses_the_next_message(
    owner, make_user, page_for, upstream
):
    reader = make_user()
    model = _model(owner, [grant("user", reader.id, "read")])
    reader_page = page_for(reader)
    _new_chat(reader_page)
    select_model(reader_page, model["name"])
    first = "is the channel clear?"
    upstream.queue(reply.text("All clear.", match=reply.answering(first)))
    send(reader_page, first)
    expect_reply(reader_page, "All clear.")

    owner_page = page_for(owner)
    owner_page.goto(f"/workspace/models/edit?id={model['id']}")
    _revoke_in_dialog(_access_dialog(owner_page), reader)
    owner_page.keyboard.press("Escape")
    owner_page.get_by_role("main").get_by_role("button", name="Save & Update").click()
    expect(owner_page).to_have_url(re.compile(r"/workspace/models/?$"))

    second = "and now?"
    upstream.queue(reply.text("Still clear.", match=reply.answering(second)))
    send(reader_page, second)
    expect(reader_page.get_by_text("Model not found").first).to_be_visible()
    assert not [r for r in upstream.chat_requests() if reply.answering(second)(r)]


# ---------------------------------------------------------------- sharing switches


def _open_group_permissions(page: Page, group_name: str) -> Locator:
    page.goto("/admin/users/groups")
    page.get_by_role("main").get_by_role("textbox", name="Search Groups").fill(group_name)
    page.get_by_label("Group hierarchy").get_by_role(
        "group", name=group_name, exact=True
    ).get_by_role("button", name=re.compile(rf"^{group_name} \d+ direct members")).click()
    editing = page.get_by_role("dialog").filter(has_text="Edit User Group")
    editing.get_by_role("button", name="Permissions", exact=True).click()
    return editing


def _base_access(page: Page, base: dict) -> Locator:
    page.goto(f"/workspace/knowledge/{base['id']}")
    return _access_dialog(page)


def test_a_groups_knowledge_sharing_switch_offers_its_members_add_access_and_public(
    admin, make_user, page_for
):
    member, outsider = make_user(), make_user()
    group_id = make_group(admin, [member], KNOWLEDGE_USER)
    make_group(admin, [outsider], KNOWLEDGE_USER)
    with admin.client() as client:
        group_name = client.get(f"/api/v1/groups/id/{group_id}").json()["name"]

    admin_page = page_for(make_user(role="admin"))
    editing = _open_group_permissions(admin_page, group_name)
    public_switch = editing.get_by_role("switch", name="Knowledge Public Sharing", exact=True)
    expect(public_switch).to_have_count(0)
    editing.get_by_role("switch", name="Knowledge Sharing", exact=True).click()
    public_switch.click()
    expect(public_switch).to_have_attribute("aria-checked", "true")
    editing.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Group updated successfully")).to_be_visible()

    member_dialog = _base_access(page_for(member), _knowledge(member))
    expect(member_dialog.get_by_role("button", name="Add Access")).to_be_visible()
    expect(visibility_choices(member_dialog)).to_have_text(["Private", "Public"])

    outsider_dialog = _base_access(page_for(outsider), _knowledge(outsider))
    expect(outsider_dialog.get_by_role("button", name="Add Access")).to_have_count(0)
    expect(visibility_choices(outsider_dialog)).to_have_text(["Private"])


def test_an_owner_without_public_sharing_can_make_a_public_base_private_but_not_public_again(
    admin, make_user, page_for
):
    base_owner = make_user()
    make_group(admin, [base_owner], KNOWLEDGE_USER)
    base = _knowledge(base_owner)
    with admin.client() as client:
        _created(
            client.post(
                f"/api/v1/knowledge/{base['id']}/access/update",
                json={"access_grants": [EVERYONE_READS]},
            )
        )

    page = page_for(base_owner)
    dialog = _base_access(page, base)
    expect(visibility(dialog)).to_have_text("Public")
    choices = visibility_choices(dialog)
    expect(choices).to_have_text(["Private", "Public"])
    with page.expect_response(lambda response: "/access/update" in response.url) as saved:
        choices.filter(has_text="Private").click()
    assert saved.value.ok
    expect(visibility(dialog)).to_have_text("Private")
    expect(visibility_choices(dialog)).to_have_text(["Private"])

    with make_user().client() as client:
        assert client.get(f"/api/v1/knowledge/{base['id']}").status_code != 200


# ---------------------------------------------------------------- base model


@pytest.fixture
def admin_only_base(admin, upstream) -> Iterator[str]:
    """A second provider model registered with no grants, so only admins may use it."""
    base_id = f"harbour-master-{uuid.uuid4().hex[:6]}"
    upstream.models.append(base_id)
    with admin.client() as client:
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        form = {"id": base_id, "base_model_id": None, "name": base_id, "meta": {}, "params": {}}
        _created(client.post("/api/v1/models/create", json={**form, "access_grants": []}))
        yield base_id
        client.post("/api/v1/models/model/delete", json={"id": base_id})
        client.get("/api/models", params={"refresh": "true"})


def _asked(upstream, question: str) -> list[dict]:
    return [request for request in upstream.chat_requests() if reply.answering(question)(request)]


def test_a_public_model_on_an_admin_only_base_is_refused_to_a_user(
    owner, admin_only_base, make_user, page_for, upstream
):
    model = _model(owner, [EVERYONE_READS], base_model_id=admin_only_base)
    page = page_for(make_user())
    _new_chat(page)
    select_model(page, model["name"])

    question = "who is on watch?"
    upstream.queue(reply.text("Ada is.", match=reply.answering(question)))
    send(page, question)

    expect(page.get_by_text("Model not found").first).to_be_visible()
    assert _asked(upstream, question) == []


def test_a_public_model_on_a_readable_base_answers_a_user(owner, make_user, page_for, upstream):
    model = _model(owner, [EVERYONE_READS])
    page = page_for(make_user())
    _new_chat(page)
    select_model(page, model["name"])

    question = "who is on watch?"
    upstream.queue(reply.text("Ada is.", match=reply.answering(question)))
    send(page, question)

    expect_reply(page, "Ada is.")
    assert [request["model"] for request in _asked(upstream, question)] == [MOCK_MODEL_ID]
