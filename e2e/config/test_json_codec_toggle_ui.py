"""Journey: mixed-language text is typed, streamed, stored and shown alike under either JSON codec.

`ENABLE_ORJSON` swaps the codec behind request bodies, stored JSON columns and the streamed chat
frames. Two instances share one database, one per value, so text written through one instance
is read and shown through the other. A person types a German, Arabic, Chinese and emoji line and
gets a streamed mixed reply that shows again after a reload; a chat stored through the API opens
from the sidebar; a tool returning a dict of mixed text shows its values when expanded; a note,
a model preset with a tag and a prompt written under one value show under the other.

Two differences between the values showed here as they do over HTTP until #31615 had stdlib json
store non-ASCII raw: on Postgres the tag filter hid a preset whose capitalised non-ASCII tag was
written with the switch off, and two long Cyrillic user variables saved from the account settings
were refused with it off and saved with it on (both explained in
integration/config/test_json_codec_toggle.py).

Data that crosses the switch is checked in the two crossing directions only (written with the
switch off and read with it on, and the other way round); the same-codec pairs and the CJK tag
spelling are left to integration/config/test_json_codec_toggle.py, which runs every pairing. The
tag case runs once, written with the switch off and with the capitalised spelling.

Discriminates: passes on dev a5bc78300; on dev 176d31d1d the two differences named above fail (the
tag case only on Postgres). In backend copies of dev 176d31d1d, the orjson codec writing mojibake
or orjson request parsing that mangles non-ASCII turns the cases with an orjson side red, and the
stdlib codec writing mojibake the ones with a stdlib side.
"""

from __future__ import annotations

import dataclasses
import re
import time
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import Actor, create_user
from harness.json_codecs import CODECS, CROSSINGS, MIXED_TEXT, codec_pair
from harness.python_tools import python_tool
from harness.upstream import MOCK_MODEL_ID
from utils.cached_chat import turn_on_tool
from utils.chat_ui import conversation, expect_reply, send
from utils.notes_page import note_row
from utils.user_menu import open_settings

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

GERMAN, ARABIC, CHINESE, EMOJI, HEBREW = (
    MIXED_TEXT[key] for key in ("german", "arabic", "chinese", "emoji", "hebrew")
)
SWITCHES = [(writer, reader) for writer, reader in CROSSINGS if writer != reader]
SWITCH_IDS = [f"{writer}-then-{reader}" for writer, reader in SWITCHES]

TOOL_SOURCE = '''class Tools:
    def look_up_greeting(self, topic: str) -> dict:
        """Look up the greeting for a topic.

        :param topic: what the greeting is about
        """
        return {
            "topic": topic,
            "german": "Grüße aus München: Straße, Äpfel, Öl und Übermaß",
            "arabic": "مرحبا بالعالم، كيف حالك اليوم؟",
            "emoji": "Launch 🚀 done ✅ family 👨‍👩‍👧‍👦 flag 🇦🇹",
        }
'''


@pytest.fixture(scope="module")
def pair(instance_with):
    return codec_pair(instance_with)


def _viewer(person: Actor, pair, reader: str) -> Actor:
    """The same account, signed in on the reader's instance (the token works on both)."""
    return dataclasses.replace(person, base_url=pair[reader].base_url)


def _unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


def _expect_texts(scope, *texts: str) -> None:
    for text in texts:
        expect(scope).to_contain_text(text)


# ---------------------------------------------------------------- typed and streamed


@pytest.mark.parametrize("codec", CODECS)
def test_a_mixed_prompt_and_streamed_reply_show_and_survive_a_reload(page_for, pair, codec):
    prompt = f"{GERMAN} {ARABIC} {CHINESE} {EMOJI}"
    pieces = [GERMAN, " ", ARABIC, " ", CHINESE, " ", HEBREW, " ", EMOJI]
    pair[codec].upstream.reset()
    pair[codec].upstream.queue(reply.text(pieces, match=reply.answering(prompt)))
    page = page_for(create_user(pair[codec]))

    send(page, prompt)
    expect_reply(page, "".join(pieces))
    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    sent = next(filter(reply.answering(prompt), pair[codec].upstream.chat_requests()))
    assert reply.conversation(sent)[-1] == {"role": "user", "content": prompt}

    page.reload()
    expect(conversation(page).get_by_text(prompt)).to_be_visible()
    expect_reply(page, "".join(pieces))


# ---------------------------------------------------------------- stored chat


@pytest.mark.parametrize(("writer", "reader"), SWITCHES, ids=SWITCH_IDS)
def test_a_chat_stored_with_mixed_text_shows_in_the_sidebar_and_opens(
    page_for, pair, writer, reader
):
    title = f"{GERMAN} {CHINESE} {EMOJI} {uuid.uuid4().hex[:6]}"
    question, answer = f"{ARABIC} {CHINESE}", f"{HEBREW} {GERMAN} {EMOJI}"
    person = create_user(pair[writer])
    now = int(time.time())
    messages = {
        "u1": {
            "id": "u1",
            "parentId": None,
            "childrenIds": ["a1"],
            "role": "user",
            "content": question,
            "timestamp": now,
        },
        "a1": {
            "id": "a1",
            "parentId": "u1",
            "childrenIds": [],
            "role": "assistant",
            "content": answer,
            "model": MOCK_MODEL_ID,
            "modelName": MOCK_MODEL_ID,
            "timestamp": now,
            "done": True,
        },
    }
    chat = {
        "title": title,
        "models": [MOCK_MODEL_ID],
        "history": {"currentId": "a1", "messages": messages},
        "messages": [messages["u1"], messages["a1"]],
    }
    with person.client() as client:
        created = client.post("/api/v1/chats/new", json={"chat": chat})
    assert created.status_code == 200, created.text

    page = page_for(_viewer(person, pair, reader))
    page.get_by_role("button", name="Open Sidebar", exact=True).click()
    sidebar = page.get_by_role("navigation", name="Chat history")
    sidebar.get_by_role("button", name=title).click()

    expect(page).to_have_url(re.compile(r"/c/[0-9a-f-]+$"))
    expect(conversation(page).get_by_text(question)).to_be_visible()
    expect_reply(page, answer)


# ---------------------------------------------------------------- tool result


@pytest.mark.parametrize("codec", CODECS)
def test_a_tool_result_of_mixed_text_shows_its_values_when_expanded(page_for, pair, codec):
    admin = create_user(pair[codec], role="admin")
    topic = f"{CHINESE} {EMOJI}"
    question = f"greet me about {_unique('topic')}"
    pair[codec].upstream.reset()
    pair[codec].upstream.queue(
        reply.tool_call("look_up_greeting", {"topic": topic}, match=reply.answering(question)),
        reply.text(f"{GERMAN} done", match=reply.answering(question)),
    )
    with python_tool(admin, TOOL_SOURCE, name=_unique("Greetings ")):
        page = page_for(admin)
        turn_on_tool(page, "Greetings ")
        send(page, question)
        expect_reply(page, f"{GERMAN} done")

        conversation(page).get_by_text("View Result from look_up_greeting").click()
        _expect_texts(conversation(page), topic, GERMAN, ARABIC, "family")


# ---------------------------------------------------------------- notes, presets, prompts


@pytest.mark.parametrize(("writer", "reader"), SWITCHES, ids=SWITCH_IDS)
def test_a_note_with_mixed_text_opens_and_shows_it(page_for, pair, writer, reader):
    title = f"{GERMAN} {_unique('note')}"
    person = create_user(pair[writer])
    body = f"{ARABIC}\n\n{CHINESE}\n\n{EMOJI}\n\n{HEBREW}"
    with person.client() as client:
        created = client.post(
            "/api/v1/notes/create",
            json={"title": title, "data": {"content": {"md": body}}, "access_grants": []},
        )
    assert created.status_code == 200, created.text

    page = page_for(_viewer(person, pair, reader), path="/notes")
    note_row(page, title).click()

    expect(page).to_have_url(re.compile(r"/notes/[0-9a-f-]+$"))
    expect(page.get_by_role("main").get_by_role("textbox", name="Title")).to_have_value(title)
    _expect_texts(page.get_by_role("main").get_by_label("Write something..."), ARABIC, CHINESE)
    _expect_texts(page.get_by_role("main").get_by_label("Write something..."), EMOJI, HEBREW)


# a capitalised non-ASCII tag stored escaped was missed on Postgres before #31615 (integration
# twin in integration/config/test_json_codec_toggle.py)
CAPITALISED_TAG = "Überblick"


@pytest.mark.parametrize(("writer", "reader"), [("stdlib", "orjson")], ids=["stdlib-then-orjson"])
def test_a_preset_with_a_non_ascii_name_and_tag_is_listed_and_found_by_its_tag(
    page_for, pair, writer, reader
):
    suffix = uuid.uuid4().hex[:6]
    tag = f"{CAPITALISED_TAG}-{suffix}"
    tagged, plain = f"Übersicht 请总结 {suffix}", f"Schlicht {suffix}"
    admin = create_user(pair[writer], role="admin")
    with admin.client() as client:
        for name, tags in ((tagged, [{"name": tag}]), (plain, [])):
            created = client.post(
                "/api/v1/models/create",
                json={
                    "id": f"preset-{uuid.uuid4().hex[:8]}",
                    "name": name,
                    "base_model_id": MOCK_MODEL_ID,
                    "meta": {"tags": tags},
                    "params": {},
                },
            )
            assert created.status_code == 200, created.text

    page = page_for(_viewer(admin, pair, reader), path="/workspace/models")
    listing = page.get_by_role("main")
    expect(listing.get_by_text(tagged, exact=True)).to_be_visible()
    expect(listing.get_by_text(plain, exact=True)).to_be_visible()

    listing.get_by_role("button", name="Tag", exact=True).click()
    page.get_by_role("button", name=tag, exact=True).click()
    # the untagged preset leaving shows the filtered list has arrived
    expect(listing.get_by_text(plain, exact=True)).to_have_count(0)
    expect(
        listing.get_by_text(tagged, exact=True),
        f"filtering by {tag!r} hid the preset written through the {writer} instance",
    ).to_be_visible()


@pytest.mark.parametrize(("writer", "reader"), SWITCHES, ids=SWITCH_IDS)
def test_a_prompt_with_a_mixed_text_title_shows_in_the_library(page_for, pair, writer, reader):
    suffix = uuid.uuid4().hex[:6]
    name = f"{GERMAN} {CHINESE} {suffix}"
    admin = create_user(pair[writer], role="admin")
    with admin.client() as client:
        created = client.post(
            "/api/v1/prompts/create",
            json={"command": f"gruss{suffix}", "name": name, "content": f"{ARABIC} {EMOJI}"},
        )
    assert created.status_code == 200, created.text

    page = page_for(_viewer(admin, pair, reader), path="/workspace/prompts")
    page.get_by_role("textbox", name="Search Prompts").fill(suffix)
    expect(page.get_by_role("main").get_by_text(name)).to_be_visible()


# ---------------------------------------------------------------- user variables

# two pasted documents, each well under the per-value limit (integration twin in
# integration/config/test_json_codec_toggle.py)
LONG_RUSSIAN = (MIXED_TEXT["russian"] + ". ") * 330


@pytest.mark.parametrize("codec", CODECS)
def test_long_non_ascii_user_variables_save_from_the_account_settings(page_for, pair, codec):
    person = create_user(pair[codec])
    page = page_for(person)
    open_settings(page)
    page.get_by_role("tab", name="Account").click()
    settings = page.get_by_role("dialog").first

    for key in ("style_guide", "contract"):
        settings.get_by_role("button", name="Add", exact=True).click()
        page.get_by_label("Variable key").fill(key)
        page.get_by_label("Variable value").fill(LONG_RUSSIAN)
        page.get_by_role("button", name="Done", exact=True).click()
    with page.expect_response(lambda response: "/user/variables/update" in response.url) as saved:
        settings.get_by_role("button", name="Save", exact=True).click()

    assert saved.value.status == 200, (
        f"saving two {len(LONG_RUSSIAN)}-character Cyrillic user variables through the {codec} "
        f"instance answered {saved.value.status}: {saved.value.text()}. The limit counts JSON "
        "characters, and stdlib json (ENABLE_ORJSON off) wrote six per Cyrillic letter"
        " until #31615"
    )
    with person.client() as client:
        stored = client.get("/api/v1/users/user/variables").json()["variables"]
    assert stored == {"style_guide": LONG_RUSSIAN, "contract": LONG_RUSSIAN}
