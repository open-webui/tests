"""Diff blocks in an assistant reply: added and removed lines, numbered and marked.

A fence tagged `diff` or `patch` is drawn as a diff instead of an editor: each line is marked as
added, removed, a hunk header, a file header or context, the lines of a hunk carry old and new
line numbers, and the part of a paired removed and added line that changed is emphasised. Edit
swaps the drawn diff for the editable source and Done swaps it back. Any other fence is not drawn
as a diff.

Discriminates: passes on the 176d31d1d build; with removed lines no longer classified, the new-line
numbering, the changed-part emphasis or the Edit toggle removed, the matching test goes red, and
with every tagged fence drawn as a diff the other-language test does.

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`). On dev c15f28649 a
side chat showed code without its editor (open-webui/open-webui#34014, fix PR
open-webui/open-webui#34064), so the Edit and other-language tests are red there in a side chat.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from utils.chat_surface import ChatSurface
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PATCH = "\n".join(
    [
        "--- a/harbour.txt",
        "+++ b/harbour.txt",
        "@@ -1,3 +1,3 @@",
        " first line",
        "-ferries leave at nine",
        "+ferries leave at ten",
        " last line",
    ]
)


def fenced(lang: str, code: str) -> str:
    return f"```{lang}\n{code}\n```"


def ask_for(chat: ChatSurface, upstream, answer: str, prompt: str = "show me the patch") -> Locator:
    upstream.queue(reply.text(answer + "\n\nThat is all.", match=reply.answering(prompt)))
    send(chat, prompt)
    expect_reply(chat, "That is all.")
    return last_reply(chat)


def line(box: Locator, kind: str) -> Locator:
    return box.locator(f".diff-line.{kind}")


@pytest.mark.parametrize("lang", ["diff", "patch"])
def test_added_and_removed_lines_are_marked_apart_from_context(chat_for, make_user, upstream, lang):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced(lang, PATCH))

    expect(line(box, "addition")).to_have_count(1)
    expect(line(box, "deletion")).to_have_count(1)
    expect(line(box, "addition")).to_contain_text("ferries leave at ten")
    expect(line(box, "deletion")).to_contain_text("ferries leave at nine")
    expect(line(box, "context")).to_have_count(2)
    expect(line(box, "hunk")).to_have_text("@@ -1,3 +1,3 @@")
    expect(line(box, "file")).to_have_count(2)
    expect(line(box, "addition").locator(".line-prefix")).to_have_text("+")
    expect(line(box, "deletion").locator(".line-prefix")).to_have_text("-")


def test_added_and_removed_lines_are_drawn_in_different_colours(chat_for, make_user, upstream):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("diff", PATCH))

    backgrounds = {
        kind: line(box, kind).first.evaluate("element => getComputedStyle(element).backgroundColor")
        for kind in ("addition", "deletion", "context")
    }
    assert len(set(backgrounds.values())) == 3, f"kinds share a background: {backgrounds}"


def test_the_lines_of_a_hunk_carry_old_and_new_numbers(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    patch = "@@ -10,2 +20,2 @@\n kept\n-old text\n+new text"

    box = ask_for(chat, upstream, fenced("diff", patch))

    def numbers(kind: str) -> tuple[str | None, str | None]:
        row = line(box, kind)
        return (
            row.locator(".old-number").get_attribute("data-line-number"),
            row.locator(".new-number").get_attribute("data-line-number"),
        )

    expect(line(box, "context")).to_have_count(1)
    assert numbers("context") == ("10", "20")
    assert numbers("deletion") == ("11", "")
    assert numbers("addition") == ("", "21")


def test_only_the_changed_part_of_a_replaced_line_is_emphasised(chat_for, make_user, upstream):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("diff", "-ferries leave at nine\n+ferries leave at ten"))

    expect(line(box, "deletion").locator(".changed")).to_have_text("nine")
    expect(line(box, "addition").locator(".changed")).to_have_text("ten")


def test_edit_shows_the_source_and_done_draws_the_diff_again(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    box = ask_for(chat, upstream, fenced("diff", "-old text\n+new text"))
    expect(line(box, "addition")).to_have_count(1)

    box.get_by_role("button", name="Edit", exact=True).click()

    expect(box.locator(".cm-line").first).to_have_text("-old text")
    expect(line(box, "addition")).to_have_count(0)
    box.get_by_role("button", name="Done", exact=True).click()
    expect(line(box, "addition")).to_have_count(1)


def test_a_fence_in_another_language_is_not_drawn_as_a_diff(chat_for, make_user, upstream):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("python", "-old text\n+new text"))

    expect(box.locator(".cm-line").first).to_have_text("-old text")
    expect(box.locator(".diff-line")).to_have_count(0)
    expect(box.get_by_role("button", name="Edit", exact=True)).to_have_count(0)
