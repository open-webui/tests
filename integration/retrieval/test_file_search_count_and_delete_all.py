"""Journey: finding and counting uploaded files, and the admin's delete-all.

The file picker searches the caller's files by name with `*` and `?` wildcards, a page at a
time; every other character, `_` included, matches only itself. The file text comes along unless
the caller asks for names only, and a pattern that matches nothing answers 404.
The count is the caller's own files. The admin searches and counts every account's files and
deletes all of them at once, which a user is refused. On an instance of its own, since deleting
everything would empty the files other modules share.

Discriminates: in a backend copy, leaving `_` unescaped in the search pattern turns the
literal-underscore test red, and dropping the owner filter from the file count turns the count
test red.
"""

from __future__ import annotations

import pytest

from harness.actors import create_user

pytestmark = [pytest.mark.journey, pytest.mark.api, pytest.mark.requires_source, pytest.mark.slow]

THROWAWAY = {"REGRESSION_THROWAWAY_INSTANCE": "files-delete-all"}


@pytest.fixture(scope="module")
def files_instance(instance_with):
    return instance_with(THROWAWAY)


def upload(client, filename: str, text: str = "some text") -> str:
    uploaded = client.post(
        "/api/v1/files/",
        params={"process_in_background": "false"},
        files={"file": (filename, text.encode(), "text/plain")},
    )
    assert uploaded.status_code == 200, uploaded.text
    return uploaded.json()["id"]


@pytest.fixture
def two_accounts(files_instance):
    """An account with three files and a stranger with one."""
    person, stranger = create_user(files_instance), create_user(files_instance)
    with person.client() as client:
        names = ["report_2024.txt", "reportX2025.txt", "notes.md"]
        ids = {name: upload(client, name, f"text of {name}") for name in names}
    with stranger.client() as client:
        upload(client, "report_stranger.txt")
    return person, stranger, ids


def search(actor_client, pattern: str, **params):
    return actor_client.get("/api/v1/files/search", params={"filename": pattern, **params})


def names(response) -> list[str]:
    assert response.status_code == 200, response.text
    return [file["filename"] for file in response.json()]


def test_wildcards_find_the_callers_files_a_page_at_a_time(two_accounts):
    person, _, _ = two_accounts
    with person.client() as client:
        assert sorted(names(search(client, "report*"))) == ["reportX2025.txt", "report_2024.txt"]
        assert names(search(client, "*.md")) == ["notes.md"]
        assert names(search(client, "?otes.md")) == ["notes.md"]
        pages = [names(search(client, "*", limit=2, skip=skip)) for skip in (0, 2)]

    assert [len(page) for page in pages] == [2, 1]
    assert sorted(pages[0] + pages[1]) == ["notes.md", "reportX2025.txt", "report_2024.txt"]


def test_an_underscore_matches_only_an_underscore(two_accounts):
    person, _, _ = two_accounts
    with person.client() as client:
        assert names(search(client, "report_*")) == ["report_2024.txt"]


def test_the_text_comes_along_unless_names_only_are_asked_for(two_accounts):
    person, _, _ = two_accounts
    with person.client() as client:
        [with_text] = search(client, "notes.md").json()
        [names_only] = search(client, "notes.md", content=False).json()

    assert with_text["data"]["content"] == "text of notes.md"
    assert "content" not in (names_only.get("data") or {})


def test_a_pattern_matching_nothing_answers_404(two_accounts):
    _, stranger, _ = two_accounts
    with stranger.client() as client:
        assert search(client, "notes.md").status_code == 404, "the stranger found someone's file"


def test_the_count_is_the_callers_own_files(two_accounts, files_instance):
    person, stranger, _ = two_accounts
    with person.client() as client:
        own = client.get("/api/v1/files/count").json()
    with stranger.client() as client:
        strangers = client.get("/api/v1/files/count").json()
    with files_instance.client() as admin:
        everyone = admin.get("/api/v1/files/count").json()
        everyone_named = names(search(admin, "report_*"))

    assert (own, strangers) == (3, 1)
    assert everyone >= 4
    assert set(everyone_named) >= {"report_2024.txt", "report_stranger.txt"}


def test_only_the_admin_deletes_every_file(two_accounts, files_instance):
    person, _, ids = two_accounts
    with person.client() as client:
        refused = client.delete("/api/v1/files/all")
        still_there = client.get(f"/api/v1/files/{ids['notes.md']}").status_code
    assert (refused.status_code, still_there) == (401, 200)

    with files_instance.client() as admin:
        deleted = admin.delete("/api/v1/files/all")
        remaining = admin.get("/api/v1/files/count").json()
    with person.client() as client:
        gone = [client.get(f"/api/v1/files/{file_id}").status_code for file_id in ids.values()]
        own = client.get("/api/v1/files/count").json()

    assert deleted.status_code == 200, deleted.text
    assert (remaining, own) == (0, 0)
    assert gone == [404, 404, 404]
