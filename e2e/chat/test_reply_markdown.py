"""Journey: the markdown in a reply is drawn as headings, lists, quotes and links a person can use.

A reply written in markdown shows its headings as headings, bold, italic, struck and code words as
such, a numbered list with a bulleted and a numbered list nested inside its steps and indented
under them, a quote with a quote inside it, a GitHub-style warning box, a task list with its boxes
ticked as written and a dividing rule, with none of the markdown signs left on screen. The reply is
drawn the same after a reload, and so is one holding a table, a code block and a diagram. A web
link opens in a new tab and leaves the chat where it is, a mail link keeps its address, and a link
to another chat of this app opens that chat in the same tab. A link whose address would run code
(`javascript:`, `data:`) is shown as plain words that do nothing, and HTML tags written into a
reply show as text: none of them runs.

Discriminates: passes on the dev ebc6add67 build. In its mutation build (the `rendering-front`
copy: links without `target`, every link address kept as written and quotes drawn as plain
blocks) the structure test, its reload twin, the new tab test and both unsafe link cases go red;
the in-app link and HTML text tests stay green there as controls. With diagrams no longer drawn
(the `rendering-front2` copy) the table, code and diagram reload test goes red.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from harness.listener import text_answer
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

PLAN = """# Trip plan

## Day one

Some **bold**, *italic*, ~~struck~~ and `inline code` words.

1. Pack the bags
   - passport
   - tickets
2. Catch the train
   1. platform four
   2. seat twelve

> A quote from the guide
>
> > and a quote inside it

> [!WARNING]
> Mind the gap

- [x] booked the hotel
- [ ] booked the dinner

---

Done planning."""

MARKDOWN_SIGNS = ["**", "~~", "# ", "> ", "[!WARNING]", "- [x]", "---"]


def ask(page: Page, upstream, answer: str, prompt: str, last_words: str) -> Locator:
    upstream.queue(reply.text(answer, match=reply.answering(prompt)))
    send(page, prompt)
    expect_reply(page, last_words)
    return last_reply(page)


def new_tabs_of(page: Page) -> list[Page]:
    """Collects every tab the page opens from now on."""
    opened: list[Page] = []
    page.context.on("page", lambda new_tab: opened.append(new_tab))
    return opened


def left_edge(element: Locator) -> float:
    return element.evaluate("element => element.getBoundingClientRect().left")


def expect_the_plan_drawn(box: Locator) -> None:
    expect(box.get_by_role("heading", level=1)).to_have_text("Trip plan")
    expect(box.get_by_role("heading", level=2)).to_have_text("Day one")
    expect(box.get_by_role("strong")).to_have_text("bold")
    expect(box.get_by_role("emphasis")).to_have_text("italic")
    expect(box.get_by_role("deletion")).to_have_text("struck")
    expect(box.get_by_role("code").filter(has_text="inline code")).to_be_visible()

    steps = box.get_by_role("list").filter(has_text="Pack the bags")
    first_step = steps.get_by_role("listitem").filter(has_text="Pack the bags")
    second_step = steps.get_by_role("listitem").filter(has_text="Catch the train")
    expect(first_step.get_by_role("listitem")).to_have_text(["passport", "tickets"])
    expect(second_step.get_by_role("listitem")).to_have_text(["platform four", "seat twelve"])
    assert first_step.evaluate("item => item.parentElement.tagName") == "OL"
    assert second_step.get_by_role("list").evaluate("list => list.tagName") == "OL"
    nested = first_step.get_by_role("listitem").first
    assert left_edge(nested) > left_edge(first_step), "the nested list is not indented"

    outer_quote = box.get_by_role("blockquote").filter(has_text="A quote from the guide")
    expect(outer_quote.get_by_role("blockquote")).to_have_text("and a quote inside it")
    warning = box.get_by_text("Mind the gap")
    expect(warning).to_be_visible()
    expect(box.get_by_text("WARNING", exact=True)).to_be_visible()

    checkboxes = box.get_by_role("checkbox")
    expect(checkboxes).to_have_count(2)
    expect(checkboxes.first).to_be_checked()
    expect(checkboxes.last).not_to_be_checked()
    expect(box.get_by_role("separator")).to_have_count(1)
    shown = box.inner_text()
    leftovers = [sign for sign in MARKDOWN_SIGNS if sign in shown]
    assert not leftovers, f"markdown signs left on screen: {leftovers}"


def test_headings_lists_quotes_and_tasks_are_drawn_as_such(page_for, make_user, upstream):
    page = page_for(make_user())

    box = ask(page, upstream, PLAN, "plan my trip", "Done planning.")

    expect_the_plan_drawn(box)


def test_the_drawn_markdown_is_the_same_after_a_reload(page_for, make_user, upstream):
    page = page_for(make_user())
    ask(page, upstream, PLAN, "plan my trip again", "Done planning.")

    page.reload()

    expect_reply(page, "Done planning.")
    expect_the_plan_drawn(last_reply(page))


MIXED = (
    "| Harbour | Depth |\n| --- | --- |\n| Portree | 6 m |\n\n"
    "```python\ndepth = 6\n```\n\n"
    "```mermaid\ngraph TD\n  A[Harbour mouth] --> B[Inner quay]\n```\n\n"
    "Charted in full."
)


def expect_the_mixed_reply_drawn(box: Locator) -> None:
    expect(box.get_by_role("table").get_by_role("cell", name="Portree")).to_be_visible()
    expect(box.get_by_text("python", exact=True)).to_be_visible()
    expect(box.locator(".cm-line").first).to_have_text("depth = 6")
    diagram = box.locator("svg[aria-roledescription='flowchart-v2']")
    expect(diagram).to_contain_text("Inner quay")
    expect(box).not_to_contain_text("graph TD")


def test_a_table_a_code_block_and_a_diagram_are_drawn_the_same_after_a_reload(
    page_for, make_user, upstream
):
    page = page_for(make_user())
    box = ask(page, upstream, MIXED, "chart the harbour", "Charted in full.")
    expect_the_mixed_reply_drawn(box)

    page.reload()

    expect_reply(page, "Charted in full.")
    expect_the_mixed_reply_drawn(last_reply(page))


@pytest.fixture
def guide(listener) -> str:
    listener.route("GET", "/guide", text_answer("<h1>The guide page</h1>"))
    return f"{listener.base_url}/guide"


def test_a_web_link_opens_in_a_new_tab_and_a_mail_link_keeps_its_address(
    page_for, make_user, upstream, guide
):
    page = page_for(make_user())
    answer = f"Read [the guide]({guide}) or [mail us](mailto:team@example.com). End of links."
    box = ask(page, upstream, answer, "where is the guide", "End of links.")
    chat_url = page.url

    link = box.get_by_role("link", name="the guide")
    expect(link).to_have_attribute("href", guide)
    with page.context.expect_page() as opened:
        link.click()

    new_tab = opened.value
    expect(new_tab.get_by_role("heading", name="The guide page")).to_be_visible()
    assert page.url == chat_url, "the chat tab itself left the chat"
    expect(box.get_by_role("link", name="mail us")).to_have_attribute(
        "href", "mailto:team@example.com"
    )


def test_a_link_to_another_chat_opens_it_in_the_same_tab(page_for, make_user, upstream):
    account = make_user()
    with account.client() as client:
        earlier_chat, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "the earlier question"},
                {"role": "assistant", "content": "The earlier answer about otters."},
            ],
        )
    page = page_for(account)
    answer = f"See [the earlier chat](/c/{earlier_chat}) for that. End of links."
    box = ask(page, upstream, answer, "where did we talk about otters", "End of links.")
    opened_pages = new_tabs_of(page)

    box.get_by_role("link", name="the earlier chat").click()

    expect(page).to_have_url(re.compile(f"/c/{earlier_chat}$"))
    expect_reply(page, "The earlier answer about otters.")
    assert opened_pages == [], "the in-app link opened a new tab"


UNSAFE_LINKS = {
    "a javascript link": "javascript:window.linkRan=true",
    "a data link": "data:text/html,<script>parent.linkRan=true</script>",
}


@pytest.mark.parametrize("kind", UNSAFE_LINKS)
def test_a_link_that_would_run_code_is_plain_words_that_do_nothing(
    page_for, make_user, upstream, kind
):
    page = page_for(make_user())
    answer = f"Do not [press here]({UNSAFE_LINKS[kind]}) please. End of links."
    box = ask(page, upstream, answer, f"show me {kind}", "End of links.")
    chat_url = page.url
    opened_pages = new_tabs_of(page)

    words = box.get_by_text("press here")
    expect(words).to_be_visible()
    expect(box.get_by_role("link")).to_have_count(0)
    words.click()

    expect(box).to_contain_text("End of links.")
    assert page.url == chat_url
    assert opened_pages == [], "the link opened a new tab"
    assert page.evaluate("window.linkRan") is None, "the link ran its code"
    hrefs = box.locator("a").evaluate_all("links => links.map(link => link.getAttribute('href'))")
    assert all(href is None for href in hrefs), f"an unsafe address is still linked: {hrefs}"


def test_html_tags_in_a_reply_show_as_text_and_never_run(page_for, make_user, upstream):
    page = page_for(make_user())
    answer = (
        'An image tag <img src="missing.png" onerror="window.tagRan=true"> and a bold tag '
        '<b>not bold</b> inline.\n\n<div onmouseover="window.tagRan=true">a div</div>\n\n'
        "End of tags."
    )

    box = ask(page, upstream, answer, "write some tags", "End of tags.")

    expect(box).to_contain_text('<img src="missing.png" onerror="window.tagRan=true">')
    expect(box).to_contain_text("<b>")
    expect(box.locator("img[src='missing.png']")).to_have_count(0)
    expect(box.get_by_role("strong")).to_have_count(0)
    box.get_by_text("a div").hover()
    assert page.evaluate("window.tagRan") is None, "a tag from the reply ran"
