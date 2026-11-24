"""Journey: the sources a reply cites, followed from the marker in the text to the source.

A reply's inline markers show where in the text each statement was grounded. Every form a model
writes them in (`[1]`, `[1][2]`, `[2, 1]`, a marker right after a word or before a comma, in a
list item or a table cell) becomes a button at that spot, named after the source it cites; a
number no source has shows no button. A marker opens its own source's cited text, by click and by
keyboard, and hovering a marker previews what it cites until the pointer leaves. A marker citing
several sources shows them one at a time in its card, paged with Previous and Next or the arrow
keys, and opens the one it shows. The sources row
under the reply counts the sources, folds out their numbered list in the order the markers number
them and folds it away again. A web page the model read shows its site icon and opens the page
itself; a source with an embedded view opens it in a tab beside the chat, or in a new window from
a shared link. A source a tool sends through its event emitter is cited like any other, live and
after a reload. Copying a cited reply copies its text without the markers' labels, and on a phone
the markers and the sources row open their dialog on the screen.

Covers the citation rework of dev 9b6241e54 and cc2e033b0 for sources that are not files; the
file types and the file preview are covered in the file-type modules, single markers after a
send, a reload and a shared link in e2e/retrieval/test_reply_citations.py.

Every test that opens its chat with `chat_for` runs in the main chat and in a side tab
(`utils/chat_surface.py`): a stored chat dragged into a tab beside a new chat, and for the tool a
new chat opened in a tab.

Dev 1612ec981 previews what a single marker cites on hover (open-webui/open-webui#34829) and shows
the sources of a marker citing several in a card that pages through them, from the pointer and the
keyboard (open-webui/open-webui#34827); its sources are no longer listed side by side. Dev
072d0b51d opens an embedded source from a side tab, in that tab's group, so in a side chat it
opens in the side chat's place, in front of it, as a cited file does; from the main chat it opens
beside the chat. The tab is named after the source (open-webui/open-webui#34828).

Discriminates: the rest passes on dev c1f3b6fd1. In a frontend copy where `[2, 1]` no longer
parses as one marker, the sources row lists its sources in reverse, the row shows no site icons,
a shared link opens an embedded source nowhere and the page drops a tool's live source events,
the marker form, group marker, sources row, web source, shared link, tool and phone tests went
red, and a control journey (test_grouped_citation_markers.py) stayed green. In a build patched to
copy the reply as rendered, the copy test went red.
"""

from __future__ import annotations

import re
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness import upstream as reply
from harness.access import grant
from harness.listener import text_answer
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.chat_surface import MAIN_CHAT, ChatSurface, main_chat
from utils.chat_ui import conversation, expect_reply, last_reply, send, start_from_link
from utils.citations_ui import (
    card_entry,
    close_source_dialog,
    expect_source_dialog,
    group_card,
    group_marker,
    leave_marker,
    listed_sources,
    seed_cited_chat,
    source_dialog,
    source_marker,
    sources_toggle,
    stored_source,
    web_source,
)
from utils.clipboard import copied, loopback_clipboard
from utils.phone import PHONE, expect_on_screen, tap_on_screen

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

HERON_DAWN = "Grey herons wait motionless in the reeds at dawn."
HERON_NEST = "Herons nest together in tall trees."
OTTER_RAIN = "Otters slide down the muddy bank after rain."
BEAVER_DAM = "Beavers fell willows to build their dams."
HERONS = stored_source("herons.txt", HERON_DAWN, HERON_NEST)
OTTERS = stored_source("otters.txt", OTTER_RAIN)
BEAVERS = stored_source("beavers.txt", BEAVER_DAM)

MARKER_FORMS = (
    "Herons wait at dawn [1]. Otters slide after rain[2], and both are shy [1][2].\n\n"
    "- Herons nest in colonies [1].\n"
    "- Otters hold paws [2, 1]\n\n"
    "| Animal | Habit |\n"
    "| --- | --- |\n"
    "| Heron | Waits in reeds [1] |\n\n"
    "Beavers build dams [9]. End of notes."
)
# each marker in the sentence sits where it was written, the punctuation after it kept
SENTENCE_WITH_MARKERS = re.compile(
    r"Herons wait at dawn\s*herons\.txt\s*\.\s*Otters slide after rain\s*otters\.txt\s*,"
    r"\s*and both are shy\s*herons\.txt\s*\+1\s*\."
)


def _open_cited_chat(chat_for, account, answer: str, sources: list[dict]) -> ChatSurface:
    chat = chat_for(account, chat_id=seed_cited_chat(account, answer, sources))
    expect_reply(chat, "End of notes.")
    return chat


def test_each_marker_form_shows_at_its_spot_and_opens_the_source_it_names(chat_for, make_user):
    chat = _open_cited_chat(chat_for, make_user(), MARKER_FORMS, [HERONS, OTTERS])
    page = chat.page
    reply_shown = last_reply(chat)

    expect(reply_shown.get_by_role("paragraph").first).to_have_text(SENTENCE_WITH_MARKERS)
    for written in ("[1]", "[2]", "[1][2]", "[2, 1]"):
        expect(reply_shown, f"the marker {written} shows as typed").not_to_contain_text(written)
    nest_item = reply_shown.get_by_role("listitem").filter(has_text="Herons nest in colonies")
    expect(source_marker(nest_item, "herons.txt")).to_have_count(1)
    paws_item = reply_shown.get_by_role("listitem").filter(has_text="Otters hold paws")
    expect(group_marker(paws_item, "otters.txt", 1)).to_be_visible()
    reeds_cell = reply_shown.get_by_role("cell").filter(has_text="Waits in reeds")
    expect(source_marker(reeds_cell, "herons.txt")).to_have_count(1)

    # [9] names no source: the sentence stays, with nothing to click
    expect(reply_shown).to_contain_text("Beavers build dams")
    expect(reply_shown).not_to_contain_text("N/A")
    expect(reply_shown.get_by_role("button", name=re.compile(r"^View source: "))).to_have_count(4)
    expect(source_marker(reply_shown, "herons.txt")).to_have_count(3)

    for position in range(3):
        source_marker(reply_shown, "herons.txt").nth(position).click()
        expect_source_dialog(page, "herons.txt", [HERON_DAWN, HERON_NEST], [OTTER_RAIN])
        close_source_dialog(page)
    source_marker(reply_shown, "otters.txt").click()
    expect_source_dialog(page, "otters.txt", [OTTER_RAIN], [HERON_DAWN])
    close_source_dialog(page)


def test_a_marker_citing_several_sources_pages_through_them_on_hover_and_opens_each(
    chat_for, make_user
):
    chat = _open_cited_chat(chat_for, make_user(), MARKER_FORMS, [HERONS, OTTERS])
    page = chat.page
    reply_shown = last_reply(chat)
    texts = {"herons.txt": HERON_DAWN, "otters.txt": OTTER_RAIN}

    for first, second in (("herons.txt", "otters.txt"), ("otters.txt", "herons.txt")):
        for position, cited in enumerate((first, second), start=1):
            group_marker(reply_shown, first, 1).hover()
            card = group_card(page)
            expect(card, f"hovering the {first} +1 marker shows no card").to_be_visible()
            expect(card).to_contain_text("1/2")
            expect(card_entry(page)).to_have_accessible_name(f"View source: {first}")
            expect(card).to_contain_text(texts[first])
            card.get_by_role("button", name="Next").click()
            expect(card).to_contain_text("2/2")
            expect(card_entry(page)).to_have_accessible_name(f"View source: {second}")
            expect(card).to_contain_text(texts[second])
            if position == 1:
                card.get_by_role("button", name="Previous").click()
                expect(card).to_contain_text("1/2")
            card_entry(page).click()
            other = second if cited == first else first
            expect_source_dialog(page, cited, [texts[cited]], [texts[other]])
            expect(group_card(page), "the card stays open over the opened source").to_have_count(0)
            close_source_dialog(page)


def test_hovering_a_marker_previews_what_it_cites_until_the_pointer_leaves(chat_for, make_user):
    chat = _open_cited_chat(chat_for, make_user(), MARKER_FORMS, [HERONS, OTTERS])
    page = chat.page
    reply_shown = last_reply(chat)
    preview = page.get_by_text(OTTER_RAIN)
    expect(preview).to_have_count(0)

    group_marker(reply_shown, "herons.txt", 1).hover()
    expect(group_card(page), "hovering a group marker shows nothing").to_be_visible()
    leave_marker(page)
    expect(group_card(page), "the card stays after the pointer left").to_have_count(0)

    source_marker(reply_shown, "otters.txt").hover()
    expect(preview, "hovering a marker shows nothing of the passage it cites").to_be_visible()
    leave_marker(page)
    expect(preview).to_have_count(0)


def test_markers_and_the_sources_row_work_from_the_keyboard(chat_for, make_user):
    chat = _open_cited_chat(chat_for, make_user(), MARKER_FORMS, [HERONS, OTTERS])
    page = chat.page
    reply_shown = last_reply(chat)

    marker = source_marker(reply_shown, "otters.txt")
    marker.focus()
    page.keyboard.press("Enter")
    expect_source_dialog(page, "otters.txt", [OTTER_RAIN], [HERON_DAWN])
    page.keyboard.press("Escape")
    expect(source_dialog(page)).to_be_hidden()
    expect(marker, "closing the source dialog lost the marker's focus").to_be_focused()

    toggle = sources_toggle(chat, 2)
    toggle.focus()
    page.keyboard.press("Enter")
    expect(toggle).to_have_attribute("aria-expanded", "true")
    page.keyboard.press("Tab")
    first_listed = listed_sources(chat).nth(0)
    expect(first_listed).to_be_focused()
    page.keyboard.press("Enter")
    expect_source_dialog(page, "herons.txt", [HERON_DAWN, HERON_NEST], [OTTER_RAIN])
    page.keyboard.press("Escape")
    expect(source_dialog(page)).to_be_hidden()
    expect(first_listed).to_be_focused()

    # a marker citing several sources opens its card, the arrow keys page it, and Enter opens
    # the source it shows
    group = group_marker(reply_shown, "herons.txt", 1)
    group.focus()
    page.keyboard.press("Enter")
    expect(group_card(page)).to_contain_text("1/2")
    expect(card_entry(page)).to_have_accessible_name("View source: herons.txt")
    page.keyboard.press("ArrowRight")
    expect(group_card(page), "the arrow key does not page the card").to_contain_text("2/2")
    expect(card_entry(page)).to_have_accessible_name("View source: otters.txt")
    page.keyboard.press("Enter")
    expect_source_dialog(page, "otters.txt", [OTTER_RAIN], [HERON_DAWN])
    page.keyboard.press("Escape")
    expect(source_dialog(page)).to_be_hidden()


def test_the_sources_row_counts_numbers_and_folds_the_sources(chat_for, make_user):
    answer = "Herons wait [1], otters slide [2] and beavers build [3]. End of notes."
    chat = _open_cited_chat(chat_for, make_user(), answer, [HERONS, OTTERS, BEAVERS])
    page = chat.page
    reply_shown = last_reply(chat)
    toggle = sources_toggle(chat, 3)

    expect(toggle).to_have_text("3 Sources")
    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(listed_sources(chat)).to_have_count(0)

    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    listed = listed_sources(chat)
    expect(listed).to_have_count(3)
    names = ["herons.txt", "otters.txt", "beavers.txt"]
    texts = [HERON_DAWN, OTTER_RAIN, BEAVER_DAM]
    for number, name in enumerate(names, start=1):
        expect(listed.nth(number - 1)).to_have_accessible_name(f"View source: {number} {name}")
    # entry n is the source marker [n] cites
    for index, name in enumerate(names):
        others = [text for text in texts if text != texts[index]]
        listed.nth(index).click()
        expect_source_dialog(page, name, [texts[index]], others)
        close_source_dialog(page)
        source_marker(reply_shown, name).click()
        expect_source_dialog(page, name, [texts[index]], others)
        close_source_dialog(page)

    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "false")
    expect(listed_sources(chat)).to_have_count(0)


WEB_PAGES = {
    "/herons": "Two grey herons stand at the jetty every morning.",
    "/otters": "An otter family lives under the old boathouse.",
    "/beavers": "The beaver lodge is upstream of the weir.",
    "/egrets": "Little egrets roost in the alders by the lake.",
    "/voles": "Water voles dig burrows in the steep banks.",
}


@pytest.mark.chat_surfaces(MAIN_CHAT)
def test_a_web_source_shows_its_site_and_opens_the_page_it_cites(chat_for, make_user, listener):
    for path, text in WEB_PAGES.items():
        listener.route("GET", path, text_answer(f"<html><body><p>{text}</p></body></html>"))
    urls = [f"{listener.base_url}{path}" for path in WEB_PAGES]
    sources = [web_source(url, text) for url, text in zip(urls, WEB_PAGES.values())]
    answer = "Herons stand at the jetty [1] and voles dig in the banks [5]. End of notes."
    chat = _open_cited_chat(chat_for, make_user(), answer, sources)
    page = chat.page
    reply_shown = last_reply(chat)
    site = urls[0].split("/")[2]

    toggle = sources_toggle(chat, 5)
    icons = toggle.get_by_role("img", name="favicon")
    expect(icons, "the sources row shows no site icons").to_have_count(3)
    for index in range(3):
        expect(icons.nth(index)).to_be_visible()
    expect(toggle).to_contain_text("+2")
    toggle.click()
    expect(listed_sources(chat).nth(4)).to_have_accessible_name(f"View source: 5 {urls[4]}")

    markers = source_marker(reply_shown, site)
    expect(markers, "the web markers are not named after the page's site").to_have_count(2)
    for marker_index, source_index in ((0, 0), (1, 4)):
        markers.nth(marker_index).click()
        url, text = urls[source_index], list(WEB_PAGES.values())[source_index]
        dialog = source_dialog(page)
        expect(dialog).to_contain_text(text)
        title = dialog.get_by_role("link", name=url, exact=True)
        with page.expect_popup() as opened:
            title.click()
        popup = opened.value
        expect(popup, "the source's title does not open the page").to_have_url(url)
        expect(popup.get_by_text(text)).to_be_visible()
        popup.close()
        # the passage link opens the page scrolled to the cited text
        passage = dialog.get_by_role("link", name="Content", exact=True)
        expect(passage).to_have_attribute("href", re.compile(rf"^{re.escape(url)}#:~:text="))
        close_source_dialog(page)


CAM_PAGE = "<html><body><h1>Heron cam live</h1></body></html>"
CAM_SOURCE_TEXT = "The colony was filmed waking at first light."


def _cam_source(cam_url: str) -> dict:
    return {
        "source": {"id": "heron-cam", "name": "Heron cam", "embed_url": cam_url},
        "document": [CAM_SOURCE_TEXT],
        "metadata": [{"source": "heron-cam", "name": "Heron cam"}],
    }


def test_an_embedded_source_opens_in_a_tab_beside_the_chat(chat_for, make_user, listener):
    listener.route("GET", "/cam", text_answer(CAM_PAGE))
    cam_url = f"{listener.base_url}/cam"
    answer = "The colony is awake [1]. End of notes."
    chat = _open_cited_chat(chat_for, make_user(), answer, [_cam_source(cam_url)])
    page = chat.page

    marker = source_marker(last_reply(chat), "Heron cam")
    marker.click()
    shown_in = page.get_by_role("tabpanel").filter(has=page.locator(f"iframe[src^='{cam_url}']"))
    expect(shown_in, "the embedded source opened no tab").to_be_visible()
    # the tab that names the panel showing the source
    tab = page.locator(f"[id='{shown_in.get_attribute('aria-labelledby')}']")
    expect(tab).to_have_attribute("aria-selected", "true")
    view = page.frame_locator(f"iframe[src^='{cam_url}']")
    expect(view.get_by_text("Heron cam live"), "the tab does not show the page").to_be_visible()

    tab.locator("xpath=..").get_by_role("button", name=re.compile(r"^Close ")).click()
    expect(page.locator(f"iframe[src^='{cam_url}']")).to_have_count(0)
    expect(conversation(chat).get_by_text("The colony is awake")).to_be_visible()

    marker.click()
    expect(view.get_by_text("Heron cam live")).to_be_visible()
    expect(tab, "the source's tab is not named after it").to_have_text("Heron cam")
    if chat.is_side:
        # in front of the side chat, which comes back when its tab is picked
        chat.show()
    expect(
        conversation(chat).get_by_text("The colony is awake"),
        "the source's tab hides the chat that cites it",
    ).to_be_visible()


def test_a_shared_link_opens_an_embedded_source_in_a_new_window(page_for, make_user, listener):
    listener.route("GET", "/cam", text_answer(CAM_PAGE))
    cam_url = f"{listener.base_url}/cam"
    owner, viewer = make_user(), make_user()
    chat_id = seed_cited_chat(
        owner, "The colony is awake [1]. End of notes.", [_cam_source(cam_url)]
    )
    with owner.client() as client:
        shared = client.post(f"/api/v1/chats/{chat_id}/share")
        assert shared.status_code == 200, shared.text
        granted = client.post(
            f"/api/v1/chats/shared/{chat_id}/access/update",
            json={"access_grants": [grant("user", viewer.id, "read")]},
        )
        assert granted.status_code == 200, granted.text

    page = page_for(viewer, path=f"/s/{shared.json()['share_id']}")
    expect(conversation(page).get_by_text("The colony is awake")).to_be_visible()
    with page.expect_popup() as opened:
        source_marker(last_reply(page), "Heron cam").click()
    popup = opened.value
    expect(popup).to_have_url(re.compile(rf"^{re.escape(cam_url)}"))
    expect(popup.get_by_text("Heron cam live")).to_be_visible()


HARBOUR_LOG = "Portree harbour log: the tide stood at 3.2 metres at noon."
LOG_TOOL = f"""class Tools:
    async def read_harbour_log(self, harbour: str, __event_emitter__=None) -> str:
        \"\"\"Read the harbour log of a harbour.

        :param harbour: the harbour
        \"\"\"
        await __event_emitter__(
            {{
                "type": "source",
                "data": {{
                    "source": {{"name": "Harbour log", "id": "harbour-log"}},
                    "document": [{HARBOUR_LOG!r}],
                    "metadata": [{{"source": "harbour-log", "name": "Harbour log"}}],
                }},
            }}
        )
        return "3.2 metres"
"""


@pytest.fixture
def log_tool(admin):
    with python_tool(admin, LOG_TOOL, name="Harbour log") as tool_id:
        yield tool_id


def test_a_source_a_tool_sends_is_cited_live_and_after_a_reload(
    chat_for, make_user, upstream, log_tool
):
    chat = chat_for(make_user())
    question = f"how high was the tide in Portree? {uuid.uuid4().hex[:6]}"
    answering = reply.answering(question)
    upstream.queue(
        reply.tool_call("read_harbour_log", {"harbour": "Portree"}, match=answering),
        reply.text("The tide stood at 3.2 metres [1].", match=answering),
    )
    start_from_link(chat, f"/?models={MOCK_MODEL_ID}&tools={log_tool}")
    send(chat, question)
    expect_reply(chat, "The tide stood at 3.2 metres")

    for shown in ("live", "after a reload"):
        if shown == "after a reload":
            chat = chat.reload()
            expect_reply(chat, "The tide stood at 3.2 metres")
        expect(sources_toggle(chat, 1), f"the tool's source is not listed {shown}").to_be_visible()
        source_marker(last_reply(chat), "Harbour log").click()
        expect_source_dialog(chat.page, "Harbour log", [HARBOUR_LOG], [])
        close_source_dialog(chat.page)


def test_copying_a_cited_reply_copies_its_text_without_the_marker_labels(page_for, make_user):
    account = make_user()
    answer = "Herons wait at dawn [1] and otters slide after rain [2]. End of notes."
    chat_id = seed_cited_chat(account, answer, [HERONS, OTTERS])
    page = page_for(account, path=f"/c/{chat_id}")
    chat = main_chat(page)
    expect_reply(chat, "End of notes.")
    expect(source_marker(last_reply(chat), "otters.txt")).to_be_visible()
    reader = loopback_clipboard(page, account)

    chat.messages.last.get_by_role("button", name="Copy", exact=True).click()
    text = copied(reader)
    assert "Herons wait at dawn" in text and "otters slide after rain" in text, (
        f"the copied reply lost its text: {text!r}"
    )
    assert "herons.txt" not in text and "otters.txt" not in text, (
        f"the copied reply holds the markers' labels instead of the reply: {text!r}"
    )


def test_on_a_phone_markers_and_the_sources_row_open_their_source_on_screen(page_for, make_user):
    account = make_user()
    chat_id = seed_cited_chat(account, MARKER_FORMS, [HERONS, OTTERS])
    page: Page = page_for(account, path=f"/c/{chat_id}", **PHONE)
    # a phone shows the chat alone, outside any tab panel
    expect_reply(page, "End of notes.")
    reply_shown = last_reply(page)

    tap_on_screen(source_marker(reply_shown, "otters.txt"))
    expect_source_dialog(page, "otters.txt", [OTTER_RAIN], [HERON_DAWN])
    close = source_dialog(page).get_by_role("button", name="Close citation modal")
    expect_on_screen(close)
    close.tap()
    expect(source_dialog(page)).to_be_hidden()

    group = group_marker(reply_shown, "herons.txt", 1)
    tap_on_screen(group)
    expect(card_entry(page)).to_have_accessible_name("View source: herons.txt")
    tap_on_screen(group_card(page).get_by_role("button", name="Next"))
    expect(card_entry(page)).to_have_accessible_name("View source: otters.txt")
    tap_on_screen(card_entry(page))
    expect_source_dialog(page, "otters.txt", [OTTER_RAIN], [HERON_DAWN])
    close_source_dialog(page)

    toggle = sources_toggle(page, 2)
    toggle.scroll_into_view_if_needed()
    tap_on_screen(toggle)
    entry = listed_sources(page).nth(0)
    entry.scroll_into_view_if_needed()
    tap_on_screen(entry)
    dialog = expect_source_dialog(page, "herons.txt", [HERON_DAWN, HERON_NEST], [OTTER_RAIN])
    expect_on_screen(dialog.get_by_text(HERON_DAWN))
