"""Regression: the sidebar search missed a chat whose title is not plain English.

open-webui commit `26f37426b` (shipped in 0.11.2). SQLite's LIKE and LOWER() fold ASCII letters
only, so the search behind the sidebar's Search dialog did not find "ΑΘΗΝΑ" when the user typed
"αθηνα". Every SQLite connection now gets a `like` that lowercases both sides in Python.

Twin of unit/models/test_sqlite_unicode_case_insensitive_search.py.

Discriminates: passes on dev bbfa876af; without the `like` override the dialog answers
"No results found".
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]


def _new_chat(client, title: str) -> None:
    chat = {"title": title, "models": [], "history": {"currentId": None, "messages": {}}}
    client.post("/api/v1/chats/new", json={"chat": chat}).raise_for_status()


def test_the_sidebar_search_finds_a_greek_title_typed_in_lowercase(page_for, make_user):
    account = make_user()
    with account.client() as client:
        _new_chat(client, "ΑΘΗΝΑ")
        _new_chat(client, "Weekly report")

    page = page_for(account)
    expect(chat_input(page)).to_be_visible()
    page.get_by_role("button", name="Search", exact=True).first.click()
    dialog = page.get_by_role("dialog")
    dialog.get_by_placeholder("Search").fill("αθηνα")

    expect(dialog.get_by_text("Weekly report")).to_have_count(0)
    expect(dialog.get_by_text("ΑΘΗΝΑ")).to_be_visible()
