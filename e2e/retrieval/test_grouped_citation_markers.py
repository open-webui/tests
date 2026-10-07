"""Journey: a reply citing several sources at one spot shows one marker that lists each of them.

A reply that cites two sources together, written `[1, 2]` or as adjacent `[1][2]`, shows one
marker named after the first source with "+1" for the other. Opening it lists a button per
source, and each opens the source dialog on that source's own text, while a single `[2]` beside
it opens the second source directly. The chat is stored with its sources, as a reply grounded
on two files leaves it, so the test reads what a reopened chat shows. Single markers, the source
list, a reload and shared links are covered in test_reply_citations.py.

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: a group marker opening the source after the one it lists) both grouped marker tests go
red; the single marker check in the first test passes there.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Page, expect

from harness.chat_history import seed_chat
from utils.chat_ui import conversation, expect_reply, last_reply

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SOURCES = {
    "herons.txt": "Grey herons wait motionless in the reeds at dawn.",
    "otters.txt": "Otters slide down the muddy bank after rain.",
}


def stored_source(name: str, text: str, distance: float) -> dict:
    return {
        "source": {"id": name, "name": name, "type": "file"},
        "document": [text],
        "metadata": [{"source": name, "name": name}],
        "distances": [distance],
    }


def open_cited_chat(page_for, make_user, answer: str) -> Page:
    account = make_user()
    sources = [stored_source(name, text, 0.8) for name, text in SOURCES.items()]
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "what do my notes say?"},
                {"role": "assistant", "content": answer, "sources": sources},
            ],
        )
    page = page_for(account)
    page.goto(f"/c/{chat_id}")
    expect_reply(page, "End of notes.")
    return page


def expect_source_dialog(page: Page, name: str) -> None:
    dialog = page.get_by_role("dialog")
    expect(dialog).to_contain_text(name)
    expect(dialog).to_contain_text(SOURCES[name])
    other = next(other for other in SOURCES if other != name)
    expect(dialog).not_to_contain_text(SOURCES[other])
    dialog.get_by_role("button", name="Close citation modal").click()
    expect(dialog).to_be_hidden()


@pytest.mark.parametrize("written", ["[1, 2]", "[1][2]"])
def test_a_group_marker_lists_each_source_and_opens_each_on_its_own_text(
    page_for, make_user, written
):
    page = open_cited_chat(
        page_for, make_user, f"Both animals are shy {written}. Otters play [2]. End of notes."
    )
    reply_shown = last_reply(page)
    # the marker's button sits inside a link-preview trigger of the same name
    group = reply_shown.get_by_role("button", name="herons.txt +1 more sources").last
    expect(group).to_be_visible()
    expect(group).to_contain_text("+1")
    expect(reply_shown).not_to_contain_text(written)

    for name in SOURCES:
        group.click()
        page.get_by_role("button", name=f"View source: {name}").click()
        expect_source_dialog(page, name)

    single = reply_shown.get_by_role("button", name="View source: otters.txt")
    expect(single).to_have_count(1)
    single.click()
    expect_source_dialog(page, "otters.txt")
    expect(conversation(page).get_by_role("button", name="Toggle 2 sources")).to_be_visible()
