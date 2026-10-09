"""Journey: a function's version history, as the admin function editor keeps and restores it.

Creating a function records its first version. A save that changes its source adds a version
with the message typed beside Save and makes it Production; a save that changes nothing adds
none. Comparing the older version to Production shows the source change as a diff. Setting the
older version as Production puts its source back, and the next chat with the pipe is answered by
that source. The Production version cannot be deleted; the older one can. Functions are the
admin's alone, so every history route refuses a user.

Discriminates: passes on dev 206bf9723 (3 of 3). In backend copies: a save that changes the
source adding no version turns `test_a_save_adds_a_version_and_makes_it_production`,
`test_comparing_the_old_version_to_production_shows_the_source_change` and
`test_the_production_version_cannot_be_deleted` red; every save adding a version turns
`test_a_save_that_changes_nothing_adds_no_version` red; Set as Production keeping the live
source and the history list route taking any signed-in account turn
`test_setting_an_old_version_as_production_answers_with_its_source` (the pipe still answers
with version two) and `test_a_user_reaches_none_of_the_history_routes` red.
"""

from __future__ import annotations

import textwrap

import httpx
import pytest

from harness.chat import ask
from harness.plugins import installed_function

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source]


def versioned_pipe(version: str) -> str:
    return (
        textwrap.dedent(
            f"""
            class Pipe:
                def pipe(self, body):
                    return "answered by version {version}"
            """
        ).strip()
        + "\n"
    )


def _save(client: httpx.Client, function_id: str, content: str, **fields) -> dict:
    form = {
        "id": function_id,
        "name": function_id,
        "content": content,
        "meta": {"description": "installed by a regression test"},
        **fields,
    }
    saved = client.post(f"/api/v1/functions/id/{function_id}/update", json=form)
    assert saved.status_code == 200, saved.text
    return saved.json()


def _function(client: httpx.Client, function_id: str) -> dict:
    read = client.get(f"/api/v1/functions/id/{function_id}")
    assert read.status_code == 200, read.text
    return read.json()


def _versions(client: httpx.Client, function_id: str) -> list[dict]:
    listed = client.get(f"/api/v1/functions/id/{function_id}/history")
    assert listed.status_code == 200, listed.text
    return listed.json()


def _pipe_reply(client: httpx.Client, pipe_id: str) -> str:
    client.get("/api/models").raise_for_status()
    _, message = ask(client, "who answers?", model=pipe_id)
    return message["content"]


def test_a_save_adds_a_version_and_makes_it_production(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id, admin.client() as client:
        first = _function(client, pipe_id)["version_id"]
        saved = _save(client, pipe_id, versioned_pipe("two"), commit_message="second wording")
        versions = _versions(client, pipe_id)
        latest = client.get(f"/api/v1/functions/id/{pipe_id}/history/{saved['version_id']}")

    assert saved["version_id"] not in (None, first)
    assert {version["id"] for version in versions} == {first, saved["version_id"]}
    assert latest.status_code == 200, latest.text
    assert latest.json()["commit_message"] == "second wording"
    assert latest.json()["snapshot"]["content"] == versioned_pipe("two")


def test_a_save_that_changes_nothing_adds_no_version(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id, admin.client() as client:
        first = _function(client, pipe_id)["version_id"]
        saved = _save(client, pipe_id, versioned_pipe("one"), commit_message="nothing changed")
        versions = _versions(client, pipe_id)

    assert [version["id"] for version in versions] == [first]
    assert saved["version_id"] == first


def test_comparing_the_old_version_to_production_shows_the_source_change(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id, admin.client() as client:
        first = _function(client, pipe_id)["version_id"]
        saved = _save(client, pipe_id, versioned_pipe("two"))
        compared = client.get(
            f"/api/v1/functions/id/{pipe_id}/history/diff",
            params={"from_id": first, "to_id": saved["version_id"]},
        )

    assert compared.status_code == 200, compared.text
    lines = compared.json()["content_diff"].splitlines()
    assert '-        return "answered by version one"' in lines
    assert '+        return "answered by version two"' in lines


def test_setting_an_old_version_as_production_answers_with_its_source(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id, admin.client() as client:
        first = _function(client, pipe_id)["version_id"]
        _save(client, pipe_id, versioned_pipe("two"))
        before = _pipe_reply(client, pipe_id)
        restored = client.post(
            f"/api/v1/functions/id/{pipe_id}/update/version", json={"version_id": first}
        )
        assert restored.status_code == 200, restored.text
        after = _pipe_reply(client, pipe_id)
        current = _function(client, pipe_id)

    assert (before, after) == ("answered by version two", "answered by version one")
    assert current["version_id"] == first
    assert current["content"] == versioned_pipe("one")


def test_the_production_version_cannot_be_deleted(admin):
    with installed_function(admin, versioned_pipe("one")) as pipe_id, admin.client() as client:
        first = _function(client, pipe_id)["version_id"]
        saved = _save(client, pipe_id, versioned_pipe("two"))
        refused = client.delete(f"/api/v1/functions/id/{pipe_id}/history/{saved['version_id']}")
        deleted = client.delete(f"/api/v1/functions/id/{pipe_id}/history/{first}")
        remaining = [version["id"] for version in _versions(client, pipe_id)]

    assert refused.status_code == 400, refused.text
    assert deleted.status_code == 200, deleted.text
    assert remaining == [saved["version_id"]]


def test_a_user_reaches_none_of_the_history_routes(admin, make_user):
    user = make_user()
    with installed_function(admin, versioned_pipe("one")) as pipe_id:
        with admin.client() as client:
            first = _function(client, pipe_id)["version_id"]
            _save(client, pipe_id, versioned_pipe("two"))
            before = (_function(client, pipe_id), _versions(client, pipe_id))
        with user.client() as client:
            answered = [
                client.get(f"/api/v1/functions/id/{pipe_id}/history").status_code,
                client.get(f"/api/v1/functions/id/{pipe_id}/history/{first}").status_code,
                client.get(
                    f"/api/v1/functions/id/{pipe_id}/history/diff",
                    params={"from_id": first, "to_id": first},
                ).status_code,
                client.delete(f"/api/v1/functions/id/{pipe_id}/history/{first}").status_code,
                client.post(
                    f"/api/v1/functions/id/{pipe_id}/update/version", json={"version_id": first}
                ).status_code,
            ]
        with admin.client() as client:
            after = (_function(client, pipe_id), _versions(client, pipe_id))

    assert answered == [401] * 5
    assert after == before
