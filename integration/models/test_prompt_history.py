"""Journey: a prompt's version history, the way the prompt editor keeps and restores versions.

Every save that changes a prompt adds a history entry holding a snapshot of it and makes that
entry the active version. Restoring an earlier entry copies its snapshot back into the prompt.
Deleting an entry takes write access to the prompt: an account the prompt is shared with for
reading is refused and the entry stays, the owner, a writer and the admin may delete it. The
active version cannot be deleted. Two versions are compared line by line for anyone who may read
the prompt, and a version of another prompt is not compared.

Discriminates: in a backend copy, `update_prompt_by_id` skipping `create_history_entry` turns
`test_a_save_adds_a_history_entry_and_makes_it_active` red (and the delete matrix, whose first
entry stays active), `update_prompt_version` no longer copying the snapshot turns
`test_restoring_an_earlier_version_brings_its_content_back` red, and the history delete handler
asking for `read` instead of `write` turns
`test_only_accounts_that_may_write_can_delete_a_history_entry` red (the reader gets 200 and the
entry is gone). The comparison tests pass on dev b130fec73; the line-by-line and read access
tests go red in a backend copy with the routes in the order before 37138282f, where the
single-version route is declared first and answers every comparison with 404, and the other-prompt
test goes red where the comparison looks versions up without binding them to the prompt.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from harness.access import Shareable, attempts, cast

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]

REFUSED, ALLOWED = 401, 200


def _prompt_form(content: str, **fields) -> dict:
    command = fields.pop("command", f"history-{uuid.uuid4().hex[:8]}")
    return {"command": command, "name": f"Prompt {content}", "content": content, **fields}


PROMPT = Shareable(
    create_path="/api/v1/prompts/create",
    create_body=lambda: _prompt_form("first draft"),
    access_path="/api/v1/prompts/id/{id}/access/update",
)


def _allow_prompt_authoring(admin, account) -> None:
    """Workspace prompt rights through a group of the account's own, leaving the defaults alone."""
    with admin.client() as client:
        created = client.post(
            "/api/v1/groups/create",
            json={
                "name": f"prompt authors {uuid.uuid4().hex[:8]}",
                "description": "prompt history",
                "permissions": {"workspace": {"prompts": True}},
            },
        )
        assert created.status_code == 200, created.text
        added = client.post(
            f"/api/v1/groups/id/{created.json()['id']}/users/add",
            json={"user_ids": [account.id]},
        )
    assert added.status_code == 200, added.text


@pytest.fixture
def author(admin, make_user):
    account = make_user()
    _allow_prompt_authoring(admin, account)
    return account


def _create_prompt(client: httpx.Client, content: str) -> dict:
    created = client.post("/api/v1/prompts/create", json=_prompt_form(content))
    assert created.status_code == 200, created.text
    return created.json()


def _save(client: httpx.Client, prompt: dict, content: str, **fields) -> dict:
    form = _prompt_form(content, command=prompt["command"], **fields)
    saved = client.post(f"/api/v1/prompts/id/{prompt['id']}/update", json=form)
    assert saved.status_code == 200, saved.text
    return saved.json()


def _history(client: httpx.Client, prompt_id: str) -> list[dict]:
    listed = client.get(f"/api/v1/prompts/id/{prompt_id}/history")
    assert listed.status_code == 200, listed.text
    return listed.json()


def _entry_holding(entries: list[dict], content: str) -> dict:
    matching = [entry for entry in entries if entry["snapshot"]["content"] == content]
    assert len(matching) == 1, f"one entry should hold {content!r}: {entries}"
    return matching[0]


def test_a_save_adds_a_history_entry_and_makes_it_active(author):
    with author.client() as client:
        prompt = _create_prompt(client, "first draft")
        saved = _save(client, prompt, "second draft", commit_message="tighten the wording")
        entries = _history(client, prompt["id"])

    assert sorted(entry["snapshot"]["content"] for entry in entries) == [
        "first draft",
        "second draft",
    ]
    latest = _entry_holding(entries, "second draft")
    assert latest["commit_message"] == "tighten the wording"
    assert latest["snapshot"]["name"] == "Prompt second draft"
    assert latest["user_id"] == author.id
    assert saved["version_id"] == latest["id"]


def test_restoring_an_earlier_version_brings_its_content_back(author):
    with author.client() as client:
        prompt = _create_prompt(client, "first draft")
        _save(client, prompt, "second draft")
        first = _entry_holding(_history(client, prompt["id"]), "first draft")

        restored = client.post(
            f"/api/v1/prompts/id/{prompt['id']}/update/version",
            json={"version_id": first["id"]},
        )
        assert restored.status_code == 200, restored.text
        reread = client.get(f"/api/v1/prompts/id/{prompt['id']}")

    assert reread.status_code == 200, reread.text
    current = reread.json()
    assert (current["content"], current["name"]) == ("first draft", "Prompt first draft")
    assert current["version_id"] == first["id"]


def test_the_active_version_cannot_be_deleted(author):
    with author.client() as client:
        prompt = _create_prompt(client, "first draft")
        saved = _save(client, prompt, "second draft")

        refused = client.delete(f"/api/v1/prompts/id/{prompt['id']}/history/{saved['version_id']}")
        remaining = [entry["id"] for entry in _history(client, prompt["id"])]

    assert refused.status_code == 400, refused.text
    assert saved["version_id"] in remaining


def _earlier_entry(owner, prompt_id: str) -> dict:
    """A second version, so the first entry is no longer the active one."""
    with owner.client() as client:
        prompt = client.get(f"/api/v1/prompts/id/{prompt_id}").json()
        _save(client, prompt, "second draft")
        first = _entry_holding(_history(client, prompt_id), "first draft")
    return {"history_id": first["id"]}


def _history_ids(owner_client: httpx.Client, fields: dict) -> set[str]:
    return {entry["id"] for entry in _history(owner_client, fields["id"])}


def test_only_accounts_that_may_write_can_delete_a_history_entry(admin, make_user):
    accounts = cast(PROMPT, admin, make_user)
    _allow_prompt_authoring(admin, accounts.owner)

    answered = attempts(
        accounts,
        "DELETE",
        "/api/v1/prompts/id/{id}/history/{history_id}",
        setup=_earlier_entry,
        look=_history_ids,
    )

    statuses = {role: attempt.status for role, attempt in answered.items()}
    assert statuses == {
        "owner": ALLOWED,
        "stranger": REFUSED,
        "reader": REFUSED,
        "writer": ALLOWED,
        "admin": ALLOWED,
    }
    for role, attempt in answered.items():
        removed = attempt.before - attempt.after
        expected_removed = 1 if statuses[role] == ALLOWED else 0
        assert len(removed) == expected_removed, f"{role}: {attempt.before} -> {attempt.after}"


def test_two_versions_are_compared_line_by_line(author):
    with author.client() as client:
        prompt = _create_prompt(client, "first draft")
        saved = _save(client, prompt, "second draft")
        first = _entry_holding(_history(client, prompt["id"]), "first draft")
        compared = client.get(
            f"/api/v1/prompts/id/{prompt['id']}/history/diff",
            params={"from_id": first["id"], "to_id": saved["version_id"]},
        )

    assert compared.status_code == 200, compared.text
    lines = compared.json()["content_diff"]
    assert "-first draft" in lines
    assert "+second draft" in lines


def test_a_version_of_another_prompt_is_not_compared(author):
    with author.client() as client:
        prompt = _create_prompt(client, "first draft")
        other = _create_prompt(client, "someone else's draft")
        compared = client.get(
            f"/api/v1/prompts/id/{prompt['id']}/history/diff",
            params={"from_id": other["version_id"], "to_id": prompt["version_id"]},
        )

    assert compared.status_code == 404, compared.text


def _both_versions(owner, prompt_id: str) -> dict:
    with owner.client() as client:
        prompt = client.get(f"/api/v1/prompts/id/{prompt_id}").json()
        saved = _save(client, prompt, "second draft")
        first = _entry_holding(_history(client, prompt_id), "first draft")
    return {"from_id": first["id"], "to_id": saved["version_id"]}


def test_any_account_that_may_read_can_compare_two_versions(admin, make_user):
    accounts = cast(PROMPT, admin, make_user)
    _allow_prompt_authoring(admin, accounts.owner)

    answered = attempts(
        accounts,
        "GET",
        "/api/v1/prompts/id/{id}/history/diff?from_id={from_id}&to_id={to_id}",
        setup=_both_versions,
    )

    assert {role: attempt.status for role, attempt in answered.items()} == {
        "owner": ALLOWED,
        "stranger": REFUSED,
        "reader": ALLOWED,
        "writer": ALLOWED,
        "admin": ALLOWED,
    }
