"""Regression: on SQLite, search and the prompt tag filter folded case for plain English only.

open-webui commit `26f37426b` (shipped in 0.11.2). SQLite's LIKE and LOWER() fold ASCII letters
only, and SQLAlchemy compiles `ilike` on SQLite to `lower(x) LIKE lower(?)`, so a chat titled
"ÆØÅ" or "ПРИВЕТ" was invisible to a lowercase search, and a prompt tagged "CAFÉ" to the
tag filter. Every SQLite connection now gets a `like` that lowercases both sides in Python, and
the tag clause goes through LIKE with its wildcards escaped.

Twin of unit/models/test_sqlite_unicode_case_insensitive_search.py.

Discriminates: passes on dev bbfa876af; without the `like` override the accented, Cyrillic and
Greek titles and the "CAFÉ" tag are not found; with the tag clause back on `LOWER(t.value) =`
the tag is not found, and without its wildcard escaping "5%" matches "50% OFF".
"""

from __future__ import annotations

import uuid

import httpx
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]

# (stored text, lowercase query that must find it). No ASCII letters: chat search also matches
# the ASCII runs of a query word by word, which would find "CAFÉ Süd" through "caf", "s", "d".
UNICODE_CASES = [
    pytest.param("ÆØÅ", "æøå", id="latin-accented"),
    pytest.param("ПРИВЕТ Мир", "привет мир", id="cyrillic"),
    pytest.param("ΑΘΗΝΑ", "αθηνα", id="greek"),
]


def _new_chat(client: httpx.Client, title: str) -> str:
    chat = {"title": title, "models": [], "history": {"currentId": None, "messages": {}}}
    created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _found_chats(client: httpx.Client, text: str) -> list[str]:
    found = client.get("/api/v1/chats/search", params={"text": text})
    assert found.status_code == 200, found.text
    return [chat["id"] for chat in found.json()]


@pytest.fixture
def prompt_author(make_user, preserve, admin):
    """A fresh account allowed to write prompts, so its list holds only its own."""
    preserve("permissions")
    with admin.client() as client:
        permissions = client.get("/api/v1/users/default/permissions").json()
        permissions["workspace"]["prompts"] = True
        granted = client.post("/api/v1/users/default/permissions", json=permissions)
    assert granted.status_code == 200, granted.text
    with make_user().client() as client:
        yield client


def _new_prompt(client: httpx.Client, name: str, tags: list[str]) -> str:
    command = f"p{uuid.uuid4().hex[:8]}"
    created = client.post(
        "/api/v1/prompts/create",
        json={"command": command, "name": name, "content": name, "tags": tags},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _listed_prompts(client: httpx.Client, **search) -> list[str]:
    listed = client.get("/api/v1/prompts/list", params=search)
    assert listed.status_code == 200, listed.text
    return [prompt["id"] for prompt in listed.json()["items"]]


# ---------------------------------------------------------------- narrow


@pytest.mark.parametrize(("stored", "query"), UNICODE_CASES)
def test_a_chat_title_is_found_by_its_lowercase_spelling(make_user, stored, query):
    with make_user().client() as client:
        chat_id = _new_chat(client, stored)

        assert _found_chats(client, query) == [chat_id], (
            f"searching {query!r} missed the chat titled {stored!r}: SQLite folded ASCII only"
        )


def test_a_prompt_tag_is_found_by_its_lowercase_spelling(prompt_author):
    prompt_id = _new_prompt(prompt_author, "Coffee notes", ["CAFÉ"])

    assert _listed_prompts(prompt_author, tag="café") == [prompt_id]


# ---------------------------------------------------------------- broad


@pytest.mark.parametrize(("stored", "query"), UNICODE_CASES)
def test_prompt_search_and_tag_filter_agree_on_case(prompt_author, stored, query):
    prompt_id = _new_prompt(prompt_author, stored, [stored])

    assert _listed_prompts(prompt_author, query=query) == [prompt_id]
    assert _listed_prompts(prompt_author, tag=query) == [prompt_id]


# ---------------------------------------------------------------- nearby


def test_an_ascii_title_and_tag_still_match(make_user, prompt_author):
    with make_user().client() as client:
        chat_id = _new_chat(client, "Weekly REPORT")
        assert _found_chats(client, "weekly report") == [chat_id]

    prompt_id = _new_prompt(prompt_author, "Weekly REPORT", ["Weekly"])
    assert _listed_prompts(prompt_author, query="weekly") == [prompt_id]
    assert _listed_prompts(prompt_author, tag="weekly") == [prompt_id]


def test_tag_wildcards_are_taken_literally(prompt_author):
    percent = _new_prompt(prompt_author, "Discount", ["50% OFF"])
    underscore = _new_prompt(prompt_author, "Snake", ["A_B"])

    assert _listed_prompts(prompt_author, tag="50% off") == [percent]
    assert _listed_prompts(prompt_author, tag="5%") == []
    assert _listed_prompts(prompt_author, tag="a_b") == [underscore]
    assert _listed_prompts(prompt_author, tag="axb") == []


def test_folding_case_does_not_strip_accents(make_user):
    with make_user().client() as client:
        _new_chat(client, "CAFÉ")

        assert _found_chats(client, "cafe") == []
