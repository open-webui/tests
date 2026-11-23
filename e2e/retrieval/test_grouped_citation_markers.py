"""Journey: a reply citing several sources at one spot shows one marker that leads to each of them.

A reply that cites two sources together, written `[1, 2]` or as adjacent `[1][2]`, shows one
marker named after the first source with "+1" for the other. Opening it shows a card that pages
through the sources (1/2, Next, 2/2; since 1612ec981, a plain list before), and each source
opens the source dialog on its own text, while a single `[2]` beside it opens the second source
directly. The chat is stored with its sources, as a reply grounded on two files leaves it, so
the test reads what a reopened chat shows. Single markers, the source list, a reload and shared
links are covered in test_reply_citations.py.

Discriminates: passes on the dev 3b6e0855d build. In its mutation build (every source in a
group marker's card opening the first source) all four cases go red on the second source.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`), a stored chat
dragged into a tab beside a new chat.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness.chat_history import seed_chat
from utils.chat_surface import ChatSurface
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


def open_cited_chat(chat_for, make_user, answer: str) -> ChatSurface:
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
    chat = chat_for(account, chat_id=chat_id)
    expect_reply(chat, "End of notes.")
    return chat


def expect_source_dialog(chat: ChatSurface, name: str) -> None:
    dialog = chat.page.get_by_role("dialog")
    expect(dialog).to_contain_text(name)
    expect(dialog).to_contain_text(SOURCES[name])
    other = next(other for other in SOURCES if other != name)
    expect(dialog).not_to_contain_text(SOURCES[other])
    dialog.get_by_role("button", name="Close citation modal").click()
    expect(dialog).to_be_hidden()


@pytest.mark.parametrize("written", ["[1, 2]", "[1][2]"])
def test_a_group_marker_shows_each_source_and_opens_each_on_its_own_text(
    chat_for, make_user, written
):
    chat = open_cited_chat(
        chat_for, make_user, f"Both animals are shy {written}. Otters play [2]. End of notes."
    )
    reply_shown = last_reply(chat)
    # the marker's button sits inside a link-preview trigger of the same name
    group = reply_shown.get_by_role("button", name="herons.txt +1 more sources").last
    expect(group).to_be_visible()
    expect(group).to_contain_text("+1")
    expect(reply_shown).not_to_contain_text(written)

    # the card has no role, and its entry shares a name with the single [2]
    card = chat.page.locator("[data-link-preview-content]")
    for position, name in enumerate(SOURCES, start=1):
        group.click()
        expect(card).to_contain_text(f"1/{len(SOURCES)}")
        for _ in range(position - 1):
            card.get_by_role("button", name="Next").click()
        expect(card).to_contain_text(f"{position}/{len(SOURCES)}")
        card.get_by_role("button", name=f"View source: {name}").click()
        expect_source_dialog(chat, name)
        # a click on the marker while its card is open would toggle it shut
        chat.page.keyboard.press("Escape")
        expect(card).to_have_count(0)

    single = reply_shown.get_by_role("button", name="View source: otters.txt")
    expect(single).to_have_count(1)
    single.click()
    expect_source_dialog(chat, "otters.txt")
    expect(conversation(chat).get_by_role("button", name="Toggle 2 sources")).to_be_visible()
