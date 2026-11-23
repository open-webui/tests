"""The sources a reply cites, as the chat shows them: inline markers, the sources row, the dialog.

A reply's inline marker such as `[2]` shows as a small button named after the source it cites
(`View source: <name>`); markers written together (`[1][2]`, `[1, 2]`) show as one button named
after the first source with `+N more sources`, whose card shows them one at a time when hovered or
clicked, paged with Previous and Next.
Under the reply a `Toggle N sources` button folds out the numbered list of every source, and any
of them opens the source dialog on its cited text.

`stored_source` and `web_source` build a source the way a reply stores it, so a test can seed a
cited chat with `seed_cited_chat` and drive what a reopened chat shows.
"""

from __future__ import annotations

import re

from playwright.sync_api import Locator, Page, expect

from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
from utils.chat_ui import conversation

Chat = Page | ChatSurface


def stored_source(name: str, *passages: str, distance: float = 0.8) -> dict:
    """A file source as a reply stores it, with one entry per cited passage."""
    return {
        "source": {"id": name, "name": name, "type": "file"},
        "document": list(passages),
        "metadata": [{"source": name, "name": name} for _ in passages],
        "distances": [distance - index * 0.1 for index in range(len(passages))],
    }


def web_source(url: str, passage: str) -> dict:
    """A web page a reply read, stored as the page's address and the text read from it."""
    return {
        "source": {"id": url, "name": url},
        "document": [passage],
        "metadata": [{"source": url, "name": url, "url": url}],
    }


def seed_cited_chat(account, answer: str, sources: list[dict], question: str = "") -> str:
    """A stored chat whose one reply is `answer`, grounded on `sources`; returns its id."""
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": question or "what do my notes say?"},
                {"role": "assistant", "content": answer, "sources": sources},
            ],
        )
    return chat_id


def source_marker(scope: Locator, name: str) -> Locator:
    """The inline markers in `scope` that cite the single source `name`."""
    return scope.get_by_role("button", name=f"View source: {name}", exact=True)


def group_marker(scope: Locator, first: str, others: int) -> Locator:
    """The inline marker in `scope` citing `first` and `others` more sources at one spot."""
    # the marker's button sits inside a link-preview trigger of the same name
    return scope.get_by_role("button", name=f"{first} +{others} more sources", exact=True).last


def group_card(page: Page) -> Locator:
    """The card a hovered or clicked marker opens."""
    return page.locator("[data-link-preview-content]")


def card_entry(page: Page) -> Locator:
    """The source the open card shows, one at a time, as a button that opens it."""
    return group_card(page).get_by_role("button", name=re.compile(r"^View source: "))


def leave_marker(page: Page) -> None:
    """Move the pointer from a marker to the page's corner, over the page between as a hand does."""
    # the card stays open after a single jump that passes over nothing on its way
    page.mouse.move(0, 0, steps=10)


def sources_toggle(chat: Chat, count: int) -> Locator:
    label = "Toggle 1 source" if count == 1 else f"Toggle {count} sources"
    return conversation(chat).get_by_role("button", name=label, exact=True)


def listed_sources(chat: Chat) -> Locator:
    """The entries of an opened sources row, each reading its number and then its name."""
    return conversation(chat).get_by_role("button", name=re.compile(r"^View source: \d+ "))


def source_dialog(page: Page) -> Locator:
    return page.get_by_role("dialog")


def expect_source_dialog(page: Page, name: str, shows: list[str], hides: list[str]) -> Locator:
    """The source dialog is open on `name`, showing each passage in `shows` and none of `hides`."""
    dialog = source_dialog(page)
    expect(dialog, f"no source dialog opened for {name}").to_be_visible()
    expect(dialog).to_contain_text(name)
    for passage in shows:
        expect(dialog, f"the dialog of {name} lacks its cited text").to_contain_text(passage)
    for passage in hides:
        expect(dialog, f"the dialog of {name} shows another source's text").not_to_contain_text(
            passage
        )
    return dialog


def close_source_dialog(page: Page) -> None:
    dialog = source_dialog(page)
    dialog.get_by_role("button", name="Close citation modal").click()
    expect(dialog).to_be_hidden()
