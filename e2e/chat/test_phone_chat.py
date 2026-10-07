"""Journey: a person on a phone signs in, finds their way through the sidebar drawer and chats.

On a 390 by 844 touch screen the sign-in form fits and leads to the chat with the message box and
its buttons on the screen. The sidebar is a drawer: its button and a swipe from the left open it
over the chat, and its close button, a tap on the dimmed chat and a swipe back close it again; a
vertical drag leaves it shut, and picking a chat in it opens the chat and closes the drawer. Enter
on a touch keyboard starts a new line, so a message goes with the send button, and the reply
streams in with Stop on the screen until it is done. The model selector fits the screen and the
model picked there answers the next message. The + menu attaches a file and, with Capture, a photo
from the camera, and both reach the model. The buttons under the latest reply sit on the screen;
its Regenerate menu tries again with a version switcher, Edit saves a corrected reply, and the
header's chat menu and a chat row's menu in the drawer fit the screen, their Download choices
too. Every control is tapped only after it is shown to lie on the screen with nothing covering it.
The buttons under earlier messages and a question's Edit show on hover, which the emulated touch
screen never sets, so they are left out.

Discriminates: passes on the ebc6add67 build apart from the two Download choices tests (red, see
below). In frontend builds of ebc6add67: with the swipe panel ignoring touch the swipe test goes
red, with a vertical drag taken for a sideways swipe the vertical drag test, with the drawer's Close
Sidebar button or the dimmed chat doing nothing on a tap the button and dimmed chat tests, with a
picked chat leaving the drawer open the picking test, with Capture opening the screen capture on a
phone the photo test, with a chat row's menu shown on hover only the row menu test, and with Enter
sending from the message box on a touch screen the new line test (the box empties before the second
line). In a frontend build whose phone layout is 480 pixels wide (wider than the screen, so controls
on the right fall off it), the sign-in, streaming, reply button, Try Again, Edit and header menu
tests go red. In a backend copy whose model create drops `params` the model selector test goes red,
and in one that leaves the chat's files out of the request the file test.

The two Download choices tests are red on dev ebc6add67: on a screen this narrow the choices open
beside the menu, past the left edge of the screen, with their names cut off
(open-webui/open-webui#32015).
"""

from __future__ import annotations

import base64
import re
import uuid

import pytest
from playwright.sync_api import Browser, Locator, Page, expect

from conftest import AppConfig
from harness import upstream as reply
from harness.chat import ask
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, conversation, expect_reply, last_reply, stop_button
from utils.phone import (
    PHONE,
    SCREEN,
    expect_off_screen,
    expect_on_screen,
    send_button,
    send_by_tapping,
    swipe,
    tap_on_screen,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

CUT_OFF = "the Download choices open past the edge of the screen (open-webui/open-webui#32015)"

# a 2x2 red PNG
RED_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEElEQVR4nGP4z8AARAwQCgAf7gP9i18U1AAAAABJRU5E"
    "rkJggg=="
)


@pytest.fixture
def phone(page_for, make_user) -> Page:
    page = page_for(make_user(), **PHONE)
    expect_on_screen(chat_input(page))
    return page


@pytest.fixture
def signed_out_phone(browser: Browser, config: AppConfig):
    context = browser.new_context(base_url=config.base_url, **PHONE)
    context.set_default_timeout(config.default_timeout)
    yield context.new_page()
    context.close()


def sidebar(page: Page) -> Locator:
    return page.get_by_role("navigation", name="Chat history")


def sidebar_new_chat(page: Page) -> Locator:
    return sidebar(page).get_by_role("link", name="New Chat")


def open_sidebar_button(page: Page) -> Locator:
    return page.get_by_role("button", name="Open Sidebar").last


def expect_sidebar_open(page: Page) -> None:
    for entry in (
        sidebar_new_chat(page),
        sidebar(page).get_by_role("button", name="Search"),
        sidebar(page).get_by_label("User menu"),
    ):
        expect_on_screen(entry)


def expect_sidebar_closed(page: Page) -> None:
    expect_off_screen(sidebar_new_chat(page))
    expect_on_screen(open_sidebar_button(page))


def whole_reply(page: Page) -> Locator:
    """The latest reply with the buttons under it."""
    return last_reply(page).locator("xpath=ancestor::*[starts-with(@id, 'message-')][1]")


def ask_by_tapping(page: Page, upstream, question: str, answer: str) -> None:
    upstream.queue(reply.text(answer, match=reply.answering(question)))
    send_by_tapping(page, question)
    expect_reply(page, answer)


def request_answering(upstream, question: str) -> dict:
    [request] = [body for body in upstream.chat_requests() if reply.answering(question)(body)]
    return request


def open_plus_menu(page: Page) -> Locator:
    tap_on_screen(page.get_by_role("button", name="More", exact=True).last)
    menu = page.get_by_role("menu")
    expect_on_screen(menu.get_by_role("button", name="Upload Files"))
    return menu


# --------------------------------------------------------------------------- signing in


def test_signing_in_on_a_phone_opens_the_chat_with_the_composer_on_screen(
    signed_out_phone, make_user
):
    account = make_user()
    page = signed_out_phone
    page.goto("/auth")
    email = page.get_by_label("Email")
    password = page.get_by_label("Password", exact=True)
    sign_in = page.get_by_role("button", name="Sign in", exact=True)
    for field in (email, password, sign_in):
        expect_on_screen(field)

    email.tap()
    page.keyboard.type(account.email)
    password.tap()
    page.keyboard.type(account.password)
    sign_in.tap()

    expect_on_screen(chat_input(page))
    for button in ("More", "Integrations", "Voice mode"):
        expect_on_screen(page.get_by_role("button", name=button, exact=True).last)
    expect_on_screen(page.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}"))
    expect_on_screen(open_sidebar_button(page))
    scroll_width = page.evaluate("document.documentElement.scrollWidth")
    assert scroll_width <= SCREEN["width"], f"the chat scrolls sideways ({scroll_width}px wide)"


# --------------------------------------------------------------------------- the sidebar drawer


def test_the_sidebar_button_opens_the_drawer_and_its_close_button_shuts_it(phone):
    expect_sidebar_closed(phone)

    tap_on_screen(open_sidebar_button(phone))
    expect_sidebar_open(phone)
    tap_on_screen(sidebar(phone).get_by_role("button", name="Close Sidebar"))
    expect_sidebar_closed(phone)


def test_a_tap_on_the_dimmed_chat_shuts_the_drawer(phone):
    tap_on_screen(open_sidebar_button(phone))
    expect_sidebar_open(phone)
    sidebar_width = sidebar(phone).bounding_box()["width"]
    assert sidebar_width < SCREEN["width"] - 60, (
        f"the drawer leaves no chat to tap ({sidebar_width})"
    )

    phone.touchscreen.tap(SCREEN["width"] - 30, SCREEN["height"] // 2)
    expect_sidebar_closed(phone)


def test_a_swipe_from_the_left_opens_the_drawer_and_a_swipe_back_shuts_it(phone):
    swipe(phone, (40, 200), (330, 200))
    expect_sidebar_open(phone)

    swipe(phone, (220, 300), (10, 300))
    expect_sidebar_closed(phone)


def test_a_vertical_drag_leaves_the_drawer_shut(phone):
    swipe(phone, (60, 150), (200, 700))

    expect_sidebar_closed(phone)


def test_picking_a_chat_in_the_drawer_opens_it_and_shuts_the_drawer(page_for, make_user, upstream):
    account = make_user()
    title_word = f"Kestrel {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("They hover in the wind.", match=reply.answering(title_word)))
    with account.client() as client:
        turn, _ = ask(client, f"{title_word}: how do they hunt?")
        client.post(
            f"/api/v1/chats/{turn.chat_id}", json={"chat": {"title": title_word}}
        ).raise_for_status()
    page = page_for(account, **PHONE)
    expect_on_screen(chat_input(page))

    tap_on_screen(open_sidebar_button(page))
    tap_on_screen(sidebar(page).get_by_role("link", name=title_word))

    expect(page).to_have_url(re.compile(turn.chat_id))
    expect_reply(page, "They hover in the wind.")
    expect_sidebar_closed(page)
    expect_on_screen(conversation(page).get_by_text("They hover in the wind."))


# --------------------------------------------------------------------------- chatting


def test_enter_starts_a_new_line_and_the_send_button_sends_both_lines(phone, upstream):
    question = f"two lines {uuid.uuid4().hex[:6]}"
    upstream.queue(reply.text("Both lines arrived.", match=reply.answering(question)))
    tap_on_screen(chat_input(phone))
    phone.keyboard.type(question)
    phone.keyboard.press("Enter")
    phone.keyboard.type("and the second line")
    # a sent message would have emptied the box before the second line was typed
    expect(chat_input(phone)).to_contain_text(question)
    expect(conversation(phone).locator(".chat-user")).to_have_count(0)

    tap_on_screen(send_button(phone))
    expect_reply(phone, "Both lines arrived.")
    sent = request_answering(upstream, question)["messages"][-1]["content"]
    assert re.search(rf"{question}\n+and the second line", str(sent)), sent


def test_a_streamed_reply_shows_stop_on_screen_until_it_is_done(phone, upstream):
    question = f"count slowly {uuid.uuid4().hex[:6]}"
    pieces = ["One, ", "two, ", "three, ", "four, ", "five, ", "six. ", "Done counting."]
    upstream.queue(reply.text(pieces, chunk_delay=0.4, match=reply.answering(question)))
    send_by_tapping(phone, question)

    expect(last_reply(phone)).to_contain_text("One,")
    expect_on_screen(stop_button(phone))
    expect_reply(phone, "Done counting.")
    expect(stop_button(phone)).to_have_count(0)
    expect_on_screen(last_reply(phone).get_by_text("Done counting."))
    expect_on_screen(chat_input(phone))


def test_the_model_picked_in_the_selector_answers_the_next_message(phone, admin, upstream):
    marker = f"Speak like a lighthouse keeper {uuid.uuid4().hex[:6]}"
    model_name = f"Keeper {uuid.uuid4().hex[:6]}"
    form = {
        "id": f"keeper-{uuid.uuid4().hex[:8]}",
        "base_model_id": MOCK_MODEL_ID,
        "name": model_name,
        "meta": {},
        "params": {"system": marker},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        client.post("/api/v1/models/create", json=form).raise_for_status()
    try:
        phone.reload()
        tap_on_screen(phone.get_by_role("button", name=f"Selected model: {MOCK_MODEL_ID}"))
        search = phone.get_by_role("textbox", name="Search In Models")
        expect_on_screen(search)
        search.fill(model_name)
        option = phone.get_by_role("listbox", name="Available models").get_by_role(
            "option", name=f"Select {model_name} model"
        )
        tap_on_screen(option)

        expect_on_screen(phone.get_by_role("button", name=f"Selected model: {model_name}"))
        question = f"who are you {uuid.uuid4().hex[:6]}"
        ask_by_tapping(phone, upstream, question, "I keep the light.")
        system = request_answering(upstream, question)["messages"][0]
        assert system == {"role": "system", "content": marker}, system
    finally:
        with admin.client() as client:
            client.post("/api/v1/models/model/delete", json={"id": form["id"]})


def test_a_file_from_the_plus_menu_reaches_the_model(phone, upstream):
    menu = open_plus_menu(phone)
    with phone.expect_file_chooser() as chooser:
        tap_on_screen(menu.get_by_role("button", name="Upload Files"))
    chooser.value.set_files(
        {"name": "tides.txt", "mimeType": "text/plain", "buffer": b"High tide comes at noon."}
    )
    attached = phone.get_by_role("button").filter(has_text="tides.txt")
    expect_on_screen(attached)

    question = f"when is high tide {uuid.uuid4().hex[:6]}"
    ask_by_tapping(phone, upstream, question, "At noon.")
    assert "High tide comes at noon." in str(request_answering(upstream, question)["messages"])


def test_capture_opens_the_camera_and_the_photo_reaches_the_model(phone, upstream):
    menu = open_plus_menu(phone)
    with phone.expect_file_chooser() as chooser:
        tap_on_screen(menu.get_by_role("button", name="Capture"))
    camera = chooser.value.element
    assert camera.get_attribute("capture") == "environment", "Capture opened no camera"
    chooser.value.set_files({"name": "photo.png", "mimeType": "image/png", "buffer": RED_PNG})
    expect_on_screen(phone.get_by_role("button", name="Show image preview"))

    question = f"what colour is it {uuid.uuid4().hex[:6]}"
    ask_by_tapping(phone, upstream, question, "Red.")
    content = request_answering(upstream, question)["messages"][-1]["content"]
    images = [part for part in content if part.get("type") == "image_url"]
    assert len(images) == 1, content


# --------------------------------------------------------------------------- reply actions


def test_the_buttons_under_the_latest_reply_are_on_screen(phone, upstream):
    ask_by_tapping(phone, upstream, f"say hello {uuid.uuid4().hex[:6]}", "Hello there.")

    for name in ("Edit", "Copy", "Good Response", "Bad Response", "Regenerate", "Fork chat"):
        expect_on_screen(whole_reply(phone).get_by_role("button", name=name).last)


def test_try_again_from_the_regenerate_menu_adds_a_version(phone, upstream):
    question = f"name a river {uuid.uuid4().hex[:6]}"
    ask_by_tapping(phone, upstream, question, "The Danube.")
    upstream.queue(reply.text("The Mekong.", match=reply.answering(question)))

    tap_on_screen(whole_reply(phone).get_by_role("button", name="Regenerate").last)
    menu = phone.get_by_role("menu")
    for entry in ("Try Again", "Add Details", "More Concise"):
        expect_on_screen(menu.get_by_role("button", name=entry))
    expect_on_screen(menu.get_by_placeholder("Suggest a change"))
    tap_on_screen(menu.get_by_role("button", name="Try Again"))

    expect_reply(phone, "The Mekong.")
    expect_on_screen(conversation(phone).get_by_text("2/2"))
    tap_on_screen(conversation(phone).get_by_role("button", name="Previous message"))
    expect_reply(phone, "The Danube.")


def test_an_edited_reply_is_saved_from_the_phone(phone, upstream):
    ask_by_tapping(phone, upstream, f"spell it {uuid.uuid4().hex[:6]}", "Recieve")

    tap_on_screen(whole_reply(phone).get_by_role("button", name="Edit").last)
    editor = last_reply(phone).locator("textarea")
    expect_on_screen(editor)
    editor.fill("Receive")
    tap_on_screen(last_reply(phone).get_by_role("button", name="Save", exact=True))

    expect_reply(phone, "Receive")
    phone.reload()
    expect_reply(phone, "Receive")
    expect(conversation(phone).get_by_text("Recieve")).to_have_count(0)


def test_the_chat_menu_in_the_header_fits_the_screen(phone, upstream):
    ask_by_tapping(phone, upstream, f"a short chat {uuid.uuid4().hex[:6]}", "Short indeed.")

    tap_on_screen(phone.get_by_role("button", name="Chat actions").last)
    menu = phone.get_by_role("menu")
    for entry in ("Share", "Download", "Copy", "Archive", "Delete"):
        expect_on_screen(menu.get_by_role("button", name=entry))


def test_the_download_choices_of_the_chat_menu_fit_the_screen(phone, upstream):
    ask_by_tapping(phone, upstream, f"keep a copy {uuid.uuid4().hex[:6]}", "Copy kept.")
    tap_on_screen(phone.get_by_role("button", name="Chat actions").last)
    download = phone.get_by_role("menu").get_by_role("button", name="Download")
    expect_on_screen(download)

    # the click of a tap, without the emulated pointer leaving the trigger afterwards
    download.dispatch_event("click")

    for choice in ("Export chat (.json)", "Plain text (.txt)", "PDF document (.pdf)"):
        expect_on_screen(phone.get_by_role("button", name=choice), CUT_OFF)


def open_chat_row_menu(page_for, make_user, upstream) -> tuple[Page, Locator]:
    """A drawer listing one chat, with that chat's menu open."""
    account = make_user()
    upstream.queue(reply.text("Plenty of herons.", match=reply.answering("herons")))
    with account.client() as client:
        ask(client, "any herons today?")
    page = page_for(account, **PHONE)
    expect_on_screen(chat_input(page))
    tap_on_screen(open_sidebar_button(page))
    # the menu trigger wraps the row's button of the same name
    tap_on_screen(sidebar(page).get_by_role("button", name="Chat Menu").last)
    return page, page.get_by_role("menu")


def test_a_chat_rows_menu_in_the_drawer_fits_the_screen(page_for, make_user, upstream):
    _, menu = open_chat_row_menu(page_for, make_user, upstream)

    for entry in ("Share", "Download", "Rename", "Pin", "Clone", "Archive", "Delete"):
        expect_on_screen(menu.get_by_role("button", name=entry, exact=True))


def test_the_download_choices_of_a_chat_rows_menu_fit_the_screen(page_for, make_user, upstream):
    page, menu = open_chat_row_menu(page_for, make_user, upstream)
    download = menu.get_by_role("button", name="Download", exact=True)
    expect_on_screen(download)

    # the click of a tap, without the emulated pointer leaving the trigger afterwards
    download.dispatch_event("click")

    for choice in ("Export chat (.json)", "Plain text (.txt)", "PDF document (.pdf)"):
        expect_on_screen(page.get_by_role("button", name=choice), CUT_OFF)
