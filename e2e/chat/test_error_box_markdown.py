"""Feature: a provider's error shows under the reply as Markdown, and nothing in it runs.

Issue open-webui/open-webui#34951, branch feat/error-box-markdown of Classic298/open-webui.
Providers and the proxies in front of them write errors for people: bold text, a list of things
to try, a status page link, an id to quote. The error box showed all of it as plain text, with
the Markdown marks visible and the link dead. It now renders the error the way replies are
rendered, keeping single line breaks, with HTML and images shown as text, and a plain one-line
error looks as it did.

Discriminates: passes on the feat/error-box-markdown 71c7e193a build, fails on the dev b1d3f9076
build (the marks show as typed, the list and the link are plain text). The checks that HTML,
scripts, images and unsafe links in an error never run or render, also in an error without a
message, pass on both; with only raw HTML blocked a `<details>` block still renders as a
collapsible, which they catch. A plain one-line error keeps its size, colour and height on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.chat_history import seed_chat
from utils.chat_ui import last_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PLAIN_ERROR = "The model is busy, please try again in a minute."
# how the box draws a one-line error on dev e2ae06477: 13px text on a 20px line, 8px above and below
PLAIN_FONT_SIZE = "13px"
PLAIN_LINE_HEIGHT = 20
PLAIN_BOX_HEIGHT = 36

MARKDOWN_ERROR = (
    "**Quota exceeded** for this key.\n"
    "Your plan allows 20 requests a minute.\n\n"
    "Try one of these:\n\n"
    "- wait a minute\n"
    "- switch to a smaller model\n\n"
    "See the [status page](https://status.example.com/incidents) and quote `req_demo_123`."
)

# each sets window.__xss_<flag> if it ever runs; `{flag}` is filled per use
PAYLOADS = {
    "script": "<script>window.__xss_{flag}=1</script>",
    "img": '<img src=x onerror="window.__xss_{flag}=1">',
    "svg": '<svg onload="window.__xss_{flag}=1"></svg>',
    "iframe": '<iframe src="javascript:parent.__xss_{flag}=1"></iframe>',
    "details": '<details open ontoggle="window.__xss_{flag}=1">details</details>',
    "math": '<math><maction xlink:href="javascript:window.__xss_{flag}=1">m</maction></math>',
    "collapsible": "<details>\n<summary>open me {flag}</summary>\n\nhidden\n\n</details>",
    "remote_image": "![pixel](http://127.0.0.1:9/{flag}.png)",
    "style": "<style>body{{background:rgb(255,0,0)}}</style>",
    "styled": '<p style="color:rgb(255,0,0)">styled text</p>',
    "html_link": '<a href="javascript:(opener||window).__xss_{flag}=1">html link</a>',
    "md_link": "[md link](javascript:(opener||window).__xss_{flag}=1)",
    "upper_link": "[upper link](JAVASCRIPT:(opener||window).__xss_{flag}=1)",
    "entity_link": "[entity link](&#106;avascript:(opener||window).__xss_{flag}=1)",
    "tab_link": "[tab link](java&#9;script:(opener||window).__xss_{flag}=1)",
    # a page holding <script>opener.__xss_data=1</script>
    "data_link": "[data link](data:text/html;base64,"
    "PHNjcmlwdD5vcGVuZXIuX194c3NfZGF0YT0xPC9zY3JpcHQ+)",
    "autolink": "<javascript:(opener||window).__xss_{flag}=1>",
}
UNSAFE_LINKS = ["html link", "md link", "upper link", "entity link", "tab link", "data link"]
# what fires by itself once it is live in the page, the proof that the payloads are not inert
SELF_FIRING = ["img", "details"]
LIVE_TAGS = "script, img, iframe, details, math, style, object, embed, video, audio, form, input"

LOOK = """
(box, text) => {
    const walker = document.createTreeWalker(box, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node && !node.textContent.includes(text)) node = walker.nextNode();
    const range = document.createRange();
    range.selectNodeContents(node);
    const textStyle = getComputedStyle(node.parentElement);
    const boxStyle = getComputedStyle(box);
    return {
        fontSize: textStyle.fontSize,
        lineHeight: parseFloat(textStyle.lineHeight),
        colour: textStyle.color,
        boxColour: boxStyle.color,
        textHeight: range.getBoundingClientRect().height,
        textTop: range.getBoundingClientRect().top - box.getBoundingClientRect().top,
        boxHeight: box.getBoundingClientRect().height,
    };
}
"""

# the top of the row each text is drawn on
ROWS = """
(scope, texts) => texts.map((text) => {
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    let node = walker.nextNode();
    while (node && !node.textContent.includes(text)) node = walker.nextNode();
    const range = document.createRange();
    range.selectNodeContents(node);
    return Math.round(range.getBoundingClientRect().top);
})
"""

OUTER_HTML = "(elements) => elements.map((element) => element.outerHTML)"
INSERT_LIVE = """
(html) => {
    const host = document.createElement('div');
    host.innerHTML = html;
    document.body.append(host);
}
"""
FLAGS_SET = "() => Object.keys(window).filter((key) => key.startsWith('__xss_'))"


def error_box(page: Page, text: str) -> Locator:
    """The box under the last reply that shows `text`, beside its info icon."""
    return (
        last_reply(page)
        .locator("div")
        .filter(has=page.get_by_text(text))
        .filter(has=page.locator("svg"))
        .last
    )


def send_and_fail(page: Page, upstream, prompt: str, message: str) -> None:
    upstream.queue(reply.error(400, message, match=reply.answering(prompt)))
    send(page, prompt)


def test_a_provider_error_shows_its_markdown_and_line_breaks(page_for, make_user, upstream):
    page = page_for(make_user())

    send_and_fail(page, upstream, "plain error please", message=PLAIN_ERROR)
    plain_box = error_box(page, PLAIN_ERROR)
    expect(plain_box).to_be_visible()
    plain = plain_box.evaluate(LOOK, PLAIN_ERROR)
    assert plain["fontSize"] == PLAIN_FONT_SIZE, f"a plain error is drawn at {plain['fontSize']}"
    assert plain["colour"] == plain["boxColour"], f"a plain error changed colour: {plain}"
    assert plain["lineHeight"] == PLAIN_LINE_HEIGHT, (
        f"a plain error has another line height: {plain}"
    )
    assert round(plain["boxHeight"]) == PLAIN_BOX_HEIGHT, f"a one-line error box grew: {plain}"

    send_and_fail(page, upstream, "markdown error please", message=MARKDOWN_ERROR)
    box = error_box(page, "Quota exceeded")
    expect(box).to_be_visible()
    expect(box).not_to_contain_text("**")
    expect(box).not_to_contain_text("`")
    expect(box).not_to_contain_text("](")

    bold = box.get_by_text("Quota exceeded", exact=True)
    expect(bold).to_be_visible()
    bold_weight = bold.evaluate("(element) => Number(getComputedStyle(element).fontWeight)")
    plain_weight = box.get_by_text("Try one of these:").evaluate(
        "(element) => Number(getComputedStyle(element).fontWeight)"
    )
    assert bold_weight > plain_weight, (
        f"the bold text is not bolder: {bold_weight} vs {plain_weight}"
    )

    rows = box.evaluate(ROWS, ["for this key.", "Your plan allows", "Try one of these:"])
    assert len(set(rows)) == 3 and rows == sorted(rows), f"the error's lines share rows: {rows}"
    assert rows[1] - rows[0] == PLAIN_LINE_HEIGHT, f"one line break left a blank row: {rows}"

    expect(box.get_by_role("listitem")).to_have_text(["wait a minute", "switch to a smaller model"])

    link = box.get_by_role("link", name="status page")
    expect(link).to_have_attribute("href", "https://status.example.com/incidents")
    expect(link).to_have_attribute("target", "_blank")

    code = box.get_by_text("req_demo_123", exact=True)
    expect(code).to_be_visible()
    code_font = code.evaluate("(element) => getComputedStyle(element).fontFamily")
    assert "mono" in code_font.lower(), f"the quoted id is not set as code: {code_font}"

    formatted = box.evaluate(LOOK, "for this key.")
    assert formatted["fontSize"] == PLAIN_FONT_SIZE, (
        f"the error text is drawn at {formatted['fontSize']}"
    )
    assert formatted["colour"] == formatted["boxColour"], (
        f"the error text changed colour: {formatted}"
    )
    assert formatted["textTop"] == plain["textTop"], (
        f"block spacing pushed the text down: {formatted}"
    )


def expect_nothing_in_it_runs(page: Page, box: Locator, dialogs: list[str]) -> None:
    live = box.locator(LIVE_TAGS)
    assert live.count() == 0, f"the error renders live HTML: {live.evaluate_all(OUTER_HTML)}"
    expect(box.locator("svg")).to_have_count(1)  # the info icon only
    protocols = box.get_by_role("link").evaluate_all(
        "(links) => links.map((link) => link.protocol)"
    )
    unsafe = [protocol for protocol in protocols if protocol not in ("http:", "https:")]
    assert not unsafe, f"the error has links with unsafe targets: {protocols}"
    # whatever a link was made of, following it must run nothing
    for link in box.get_by_role("link").all():
        link.click()

    # the same payloads put live into the page do run, so the check below would see them
    live_payloads = "".join(PAYLOADS[name].format(flag=f"control_{name}") for name in SELF_FIRING)
    page.evaluate(INSERT_LIVE, live_payloads)
    for name in SELF_FIRING:
        page.wait_for_function(f"() => window.__xss_control_{name} === 1")

    ran = [flag for flag in page.evaluate(FLAGS_SET) if "control" not in flag]
    assert not ran, f"script in the error ran: {ran}"
    assert not dialogs, f"script in the error opened dialogs: {dialogs}"


def watch_for_script(page: Page) -> list[str]:
    dialogs: list[str] = []
    page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
    page.context.on("popup", lambda popup: popup.close())
    return dialogs


def test_html_scripts_and_unsafe_links_in_a_provider_error_never_run(page_for, make_user, upstream):
    page = page_for(make_user())
    dialogs = watch_for_script(page)

    payloads = "\n\n".join(payload.format(flag=f"box_{name}") for name, payload in PAYLOADS.items())
    send_and_fail(page, upstream, "hostile error please", message=f"Blocked.\n\n{payloads}")
    box = error_box(page, "Blocked.")
    expect(box).to_contain_text("styled text")
    for name in UNSAFE_LINKS:
        expect(box).to_contain_text(name)

    styled = box.evaluate(LOOK, "styled text")
    assert styled["colour"] == styled["boxColour"], (
        f"a style attribute in the error applied: {styled}"
    )
    background = page.evaluate("() => getComputedStyle(document.body).backgroundColor")
    assert background != "rgb(255, 0, 0)", "a style tag in the error restyled the page"

    expect_nothing_in_it_runs(page, box, dialogs)


def test_an_error_without_a_message_shows_its_json_and_runs_nothing(page_for, make_user):
    owner = make_user()
    # stored as a provider's error line without a message leaves it, an object of its own fields
    error_object = {
        "code": PAYLOADS["img"].format(flag="json_img"),
        "type": PAYLOADS["md_link"].format(flag="json_link"),
        "param": "rate_limit_exceeded",
    }
    with owner.client() as client:
        chat_id, _ = seed_chat(
            client,
            [
                {"role": "user", "content": "hello?"},
                {"role": "assistant", "content": "", "error": {"content": error_object}},
            ],
        )
    page = page_for(owner, path=f"/c/{chat_id}")
    dialogs = watch_for_script(page)

    box = error_box(page, '{"code":"')
    expect(box).to_contain_text('"param":"rate_limit_exceeded"')
    expect(box).to_contain_text("json_img")
    expect_nothing_in_it_runs(page, box, dialogs)
