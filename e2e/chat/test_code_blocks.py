"""Code blocks in an assistant reply: the label, the highlighting and the controls a person uses.

A fenced block in a reply shows its language above the code, highlights the code by that language
and offers Collapse, Copy and Save (plus Preview on HTML and SVG). A `mermaid` fence draws a
diagram instead of showing its source, and falls back to the source with an error when the
diagram does not parse. A Python block offers Run only while the admin's Code Execution is on and
no other language offers it (running it is legacy and not covered), and in a long block the
language label and the buttons stay in view while its middle is read. The artifacts pane
opened by an HTML block is covered in test_artifacts_pane_after_delete.py. A fence made of
tildes is a code block like a backtick one (open-webui/open-webui#31543, issue #31542). Saving
an edited block keeps its dollar signs as typed, in a reply with output items and in one with
only text (open-webui/open-webui#31386, issue #31385): the edit used to go in as a replacement
pattern, so `$$` became `$` and `$&` the old code.

Discriminates: passes on the 176d31d1d build; with the language label, the highlighter, the
clipboard write, the collapse toggle, the save handler, the preview button or the mermaid
render removed, the matching test goes red. With the check that only backtick fences are code
blocks restored (the a5bc78300 mutation build), the tilde-fence test finds one plain line.
With `99f1eaa3f` reverted (the 015dbc861 mutation build) both dollar sign tests go red. On dev
ebc6add67, in its mutation build (the `rendering-front` copy: Run shown whatever the switch and
the buttons no longer sticky) the Code Execution off test and the long block test go red, the
Run-while-on test stays green. On dev f659c0d12 the HTML-and-SVG preview test goes red: the
Artifacts view the HTML block opened on its own took the chat's place and hid the reply's buttons
(open-webui/open-webui#33878).

Every test runs in the main chat and in a side chat (`utils/chat_surface.py`); the ones on Save and
Preview in every way a side chat starts and every state it can be in (a temporary one left out of
the Save tests, which reload), and the one on a saved chat also with that chat dragged into a tab.
A side chat answer's code blocks had no Save and no editing (open-webui/open-webui#34014) and no
Preview (open-webui/open-webui#34026), fix PR open-webui/open-webui#34064: on dev c15f28649 those
side chat variants are red, and a side chat showed its code without the editor, so the language,
highlighting, collapse and long block tests are red there in a side chat as well. On dev e791cbdde,
which holds that PR, the side chat's Preview button is there but the page's one artifacts pane
does not show its block (open-webui/open-webui#34026, not part of the PR), so the Preview pane
test stays red in every side chat.
"""

from __future__ import annotations

import json

import pytest
from playwright.sync_api import Locator, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_surface import (
    DRAGGED_TAB,
    EVERYWHERE,
    MAIN_CHAT,
    SIDE_CHAT,
    TEMPORARY_SIDE,
    ChatSurface,
)
from utils.chat_ui import expect_reply, last_reply, send
from utils.clipboard import read_text

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SCRIPT = "function greet(name) {\n  return `hi ${name}` + '$&';\n}\n\ngreet(\"x\");"
BROKEN_DIAGRAM = "graph TD\n  A[Start --> ((("
# the model count cannot matter: a code block's buttons belong to the one reply holding it
# a temporary chat is gone after a reload
EVERY_LASTING_PLACE = tuple(place for place in EVERYWHERE if place != TEMPORARY_SIDE)


def fenced(lang: str, code: str) -> str:
    return f"```{lang}\n{code}\n```"


def ask_for(chat: ChatSurface, upstream, answer: str, prompt: str = "show me the code") -> Locator:
    upstream.queue(reply.text(answer + "\n\nThat is all.", match=reply.answering(prompt)))
    send(chat, prompt)
    expect_reply(chat, "That is all.")
    return last_reply(chat)


def editor_lines(block: Locator) -> Locator:
    return block.locator(".cm-line")


def button(reply_box: Locator, name: str) -> Locator:
    return reply_box.get_by_role("button", name=name, exact=True)


def test_a_block_shows_its_language_and_highlights_the_code(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("javascript", "const answer = 42;\n// plain note"))

    expect(box.get_by_text("javascript", exact=True)).to_be_visible()
    expect(editor_lines(box).first).to_have_text("const answer = 42;")
    expect(editor_lines(box).last).to_have_text("// plain note")
    # a highlighted token sits in its own element, told apart from its neighbours by colour
    keyword = editor_lines(box).first.locator("span", has_text="const")
    number = editor_lines(box).first.locator("span", has_text="42")
    expect(keyword).to_be_visible()
    expect(number).to_be_visible()
    colours = {
        node.evaluate("element => getComputedStyle(element).color") for node in (keyword, number)
    }
    assert len(colours) == 2, f"keyword and number share one colour: {colours}"


def test_a_block_without_a_language_has_no_highlighted_tokens(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("", "const answer = 42;"))

    expect(editor_lines(box).first).to_have_text("const answer = 42;")
    expect(editor_lines(box).first.locator("span")).to_have_count(0)


def test_copy_puts_the_exact_code_on_the_clipboard(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    chat.page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    box = ask_for(chat, upstream, fenced("javascript", SCRIPT))

    button(box, "Copy").click()

    expect(button(box, "Copied")).to_be_visible()
    assert read_text(chat.page) == SCRIPT
    expect(button(box, "Copy")).to_be_visible()


@pytest.mark.regression
def test_a_tilde_fence_is_a_code_block_with_a_label_and_a_copy_button(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    chat = chat_for(make_user())
    chat.page.context.grant_permissions(["clipboard-read", "clipboard-write"])

    box = ask_for(chat, upstream, "~~~python\nfirst = 1\nsecond = 2\n~~~")

    expect(
        box.get_by_text("python", exact=True), "no language label on the ~~~ block"
    ).to_be_visible()
    expect(editor_lines(box), "the ~~~ block is not shown as lines of code").to_have_count(2)
    expect(editor_lines(box).first).to_have_text("first = 1")
    expect(editor_lines(box).last).to_have_text("second = 2")
    button(box, "Copy").click()
    expect(button(box, "Copied")).to_be_visible()
    assert read_text(chat.page) == "first = 1\nsecond = 2"


def test_each_block_copies_its_own_code(chat_for, make_user, upstream):
    chat = chat_for(make_user())
    chat.page.context.grant_permissions(["clipboard-read", "clipboard-write"])
    answer = fenced("bash", "echo first") + "\n\n" + fenced("bash", "echo second")
    box = ask_for(chat, upstream, answer)

    button(box, "Copy").last.click()

    expect(button(box, "Copied")).to_be_visible()
    assert read_text(chat.page) == "echo second"


def test_a_mermaid_fence_is_drawn_as_a_diagram(chat_for, make_user, upstream):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("mermaid", "graph TD\n  A[Start node] --> B[End node]"))

    diagram = box.locator("svg[aria-roledescription='flowchart-v2']")
    expect(diagram).to_be_visible()
    expect(diagram).to_contain_text("Start node")
    expect(diagram).to_contain_text("End node")
    expect(box).not_to_contain_text("graph TD")
    expect(box.locator(".cm-editor")).to_have_count(0)


def test_a_mermaid_fence_that_does_not_parse_shows_its_source_and_an_error(
    chat_for, make_user, upstream
):
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, fenced("mermaid", BROKEN_DIAGRAM))

    expect(box.get_by_text("Failed to render diagram")).to_be_visible()
    expect(box.locator("pre")).to_have_text(BROKEN_DIAGRAM)
    expect(box.locator("svg[aria-roledescription='flowchart-v2']")).to_have_count(0)


def test_collapse_hides_the_code_and_expand_brings_it_back(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    chat = chat_for(make_user())
    box = ask_for(chat, upstream, fenced("python", "first = 1\nsecond = 2\nthird = 3"))
    expect(editor_lines(box)).to_have_count(3)

    button(box, "Collapse").click()

    expect(box.get_by_text("3 hidden lines")).to_be_visible()
    expect(box.locator(".cm-editor")).to_have_count(0)
    expect(button(box, "Copy")).to_be_visible()
    button(box, "Expand").click()
    expect(editor_lines(box)).to_have_count(3)
    expect(box.get_by_text("3 hidden lines")).to_have_count(0)


def test_blocks_start_collapsed_when_the_account_prefers_it(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"collapseCodeBlocks": True}}
        )
    saved.raise_for_status()
    chat = chat_for(account)

    box = ask_for(chat, upstream, fenced("python", "first = 1\nsecond = 2"))

    expect(box.get_by_text("2 hidden lines")).to_be_visible()
    expect(box.locator(".cm-editor")).to_have_count(0)
    button(box, "Expand").click()
    expect(editor_lines(box)).to_have_count(2)


def stored_reply_text(client, chat_id: str) -> str:
    messages = client.get(f"/api/v1/chats/{chat_id}").json()["chat"]["history"]["messages"]
    return json.dumps(list(messages.values()))


@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_saving_an_edited_block_rewrites_the_reply(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    account = make_user()
    chat = chat_for(account)
    box = ask_for(chat, upstream, fenced("python", "total = 1"))

    chat_id = chat.page.url.rsplit("/", 1)[-1]
    editor_lines(box).first.click()
    chat.page.keyboard.press("End")
    chat.page.keyboard.type(" + 41")
    with chat.page.expect_response(
        lambda r: r.request.method == "POST" and f"/chats/{chat_id}" in r.url
    ):
        button(box, "Save").click()

    expect(button(box, "Saved")).to_be_visible()
    expect(editor_lines(box).first).to_have_text("total = 1 + 41")
    with account.client() as client:
        stored = stored_reply_text(client, chat_id)
    assert "total = 1 + 41" in stored, f"the edit was never stored: {stored!r}"
    # the stored reply keeps the edit in its output items
    assert "```python\\ntotal = 1 + 41\\n```" in stored
    chat = chat.reload()
    expect(editor_lines(last_reply(chat)).first).to_have_text("total = 1 + 41")


# replacement patterns in String.replace: `$$` is one dollar, `$&` the matched text
DOLLAR_EDIT = " # pays $$5 and keeps $&"


def save_an_edit(chat: ChatSurface, box: Locator, chat_id: str, typed: str) -> None:
    editor_lines(box).first.click()
    chat.page.keyboard.press("End")
    chat.page.keyboard.type(typed)
    with chat.page.expect_response(
        lambda r: r.request.method == "POST" and f"/chats/{chat_id}" in r.url
    ):
        button(box, "Save").click()
    expect(button(box, "Saved")).to_be_visible()


@pytest.mark.regression
@pytest.mark.chat_surfaces(*EVERY_LASTING_PLACE)
def test_a_saved_block_keeps_its_dollar_signs(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34014."""
    account = make_user()
    chat = chat_for(account)
    box = ask_for(chat, upstream, fenced("python", "total = 1"))
    chat_id = chat.page.url.rsplit("/", 1)[-1]

    save_an_edit(chat, box, chat_id, DOLLAR_EDIT)

    with account.client() as client:
        stored = stored_reply_text(client, chat_id)
    assert "total = 1 # pays $$5 and keeps $&" in stored, (
        f"the saved code lost its dollar signs (#31386): {stored!r}"
    )
    chat = chat.reload()
    expect(editor_lines(last_reply(chat)).first).to_have_text("total = 1 # pays $$5 and keeps $&")


# a saved chat opens in the main view or is dragged into a tab, the only places it can be opened
@pytest.mark.regression
@pytest.mark.chat_surfaces(MAIN_CHAT, SIDE_CHAT, DRAGGED_TAB)
def test_a_saved_block_in_a_reply_without_output_items_keeps_its_dollar_signs(chat_for, make_user):
    """In a side chat: open-webui/open-webui#34014."""
    account = make_user()
    answer = fenced("python", "total = 1") + "\n\nThat is all."
    with account.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "show me the code"},
                {"role": "assistant", "content": answer},
            ],
        )
    chat = chat_for(account, chat_id=chat_id)
    box = last_reply(chat)
    expect(editor_lines(box).first).to_have_text("total = 1")

    save_an_edit(chat, box, chat_id, DOLLAR_EDIT)

    with account.client() as client:
        stored = stored_reply_text(client, chat_id)
    assert "total = 1 # pays $$5 and keeps $&" in stored, (
        f"the saved code lost its dollar signs (#31386): {stored!r}"
    )
    chat = chat.reload()
    expect(editor_lines(last_reply(chat)).first).to_have_text("total = 1 # pays $$5 and keeps $&")


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_only_html_and_svg_blocks_offer_a_preview(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34026."""
    chat = chat_for(make_user())
    answer = fenced("python", "x = 1") + "\n\n" + fenced("html", "<p>page</p>")

    box = ask_for(chat, upstream, answer)

    expect(button(box, "Copy")).to_have_count(2)
    expect(button(box, "Preview")).to_have_count(1)


@pytest.mark.chat_surfaces(*EVERYWHERE)
def test_preview_opens_the_html_block_in_the_artifacts_pane(chat_for, make_user, upstream):
    """In a side chat: open-webui/open-webui#34026."""
    account = make_user()
    with account.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update", json={"ui": {"detectArtifacts": False}}
        )
    saved.raise_for_status()
    chat = chat_for(account)
    box = ask_for(chat, upstream, fenced("html", "<h1>hello preview</h1>"))
    pane = chat.page.locator("#artifacts-container")
    expect(pane).to_have_count(0)

    button(box, "Preview").click()

    expect(pane.frame_locator("iframe").locator("h1")).to_have_text("hello preview")


CODE_EXECUTION_CONFIG = ("/api/v1/configs/code_execution", "/api/v1/configs/code_execution")


def set_code_execution(admin, enabled: bool) -> None:
    with admin.client() as client:
        current = client.get(CODE_EXECUTION_CONFIG[0]).json()
        saved = client.post(
            CODE_EXECUTION_CONFIG[1], json={**current, "ENABLE_CODE_EXECUTION": enabled}
        )
    saved.raise_for_status()


PYTHON_AND_SCRIPT = fenced("python", "print('hi')") + "\n\n" + fenced("javascript", "alert(1);")


def test_only_python_blocks_offer_run_while_code_execution_is_on(
    chat_for, make_user, upstream, admin, preserve
):
    preserve(CODE_EXECUTION_CONFIG)
    set_code_execution(admin, True)
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, PYTHON_AND_SCRIPT, "show me two scripts")

    expect(button(box, "Copy")).to_have_count(2)
    expect(button(box, "Run")).to_have_count(1)
    python_block = box.locator(".language-python").locator("xpath=..")
    expect(button(python_block, "Run")).to_be_visible()


def test_no_block_offers_run_once_code_execution_is_off(
    chat_for, make_user, upstream, admin, preserve
):
    preserve(CODE_EXECUTION_CONFIG)
    set_code_execution(admin, False)
    chat = chat_for(make_user())

    box = ask_for(chat, upstream, PYTHON_AND_SCRIPT, "show me two scripts")

    expect(button(box, "Copy")).to_have_count(2)
    expect(button(box, "Collapse")).to_have_count(2)
    expect(button(box, "Run")).to_have_count(0)


LONG_SCRIPT = "\n".join(f"step_{number} = {number}" for number in range(1, 151))


def test_a_long_blocks_copy_button_stays_in_view_while_its_middle_is_read(
    chat_for, make_user, upstream
):
    """In a side chat: open-webui/open-webui#34014, the code shown without its editor."""
    chat = chat_for(make_user())
    box = ask_for(chat, upstream, fenced("python", LONG_SCRIPT), "show me a long script")
    middle_line = editor_lines(box).filter(has_text="step_100 = 100")

    middle_line.scroll_into_view_if_needed()

    expect(middle_line).to_be_in_viewport()
    expect(box.get_by_text("step_1 = 1", exact=True)).not_to_be_in_viewport()
    expect(button(box, "Copy")).to_be_in_viewport()
    expect(box.get_by_text("python", exact=True)).to_be_in_viewport()
