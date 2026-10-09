"""Regression: a YouTube link whose transcript YouTube refuses tells the user why, in the chat.

open-webui issue #28361, PR #28362 (commit 121f2404e): the transcript loader swallowed the
transcript API's error, so a YouTube link attached to a chat failed with a generic message and
the hint that fixes a blocked server (the Youtube Proxy URL setting) never reached the user.

The chat page attaches the video named in its `youtube` query parameter, the way a shared "chat
about this video" link does, once the person confirms it (since 0ffd86967). YouTube is played by
`harness.youtube` behind the admin's Youtube Proxy URL, on an instance that trusts its
certificate authority.

Twin of integration/retrieval/test_youtube_transcripts.py.

Discriminates: passes on dev ef67cc3fa; with the loader returning `[]` for a refused transcript
(121f2404e reverted) the refusal case fails (the toast names no reason); the transcript case
passes on both.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness import youtube
from harness.actors import create_user
from harness.upstream import MOCK_MODEL_ID
from harness.web_retrieval import LOCAL_WEB_FETCH, save_web_settings, web_settings_restored
from utils.chat_ui import expect_reply, link_dialog, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]


@pytest.fixture(scope="module")
def fake_youtube():
    with youtube.serving_youtube() as fake:
        yield fake


@pytest.fixture
def youtube_instance(instance_with, fake_youtube):
    launched = instance_with({**LOCAL_WEB_FETCH, **youtube.youtube_env(fake_youtube)})
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    with launched.client() as client, web_settings_restored(client):
        save_web_settings(
            client, YOUTUBE_LOADER_PROXY_URL=fake_youtube.proxy_url, YOUTUBE_LOADER_LANGUAGE=["en"]
        )
        yield launched


def open_chat_about(page_for, launched, video_id: str):
    page = page_for(create_user(launched))
    page.goto(f"/?models={MOCK_MODEL_ID}&youtube={video_id}")
    link_dialog(page).get_by_role("button", name="Confirm").click()
    return page


def test_a_blocked_server_is_told_to_set_a_youtube_proxy(page_for, youtube_instance, fake_youtube):
    fake_youtube.videos["botcheck002"] = youtube.unplayable("LOGIN_REQUIRED", youtube.BOT_CHECK)

    page = open_chat_about(page_for, youtube_instance, "botcheck002")

    expect(page.get_by_text("Youtube Proxy URL")).to_be_visible(timeout=30_000)


def test_a_transcript_reaches_the_model(page_for, youtube_instance, fake_youtube):
    fake_youtube.videos["dQw4w9WgXcR"] = youtube.with_transcript("never gonna", "give you up")
    prompt = "what does the video say?"
    youtube_instance.upstream.queue(reply.text("it sings", match=reply.answering(prompt)))
    page = open_chat_about(page_for, youtube_instance, "dQw4w9WgXcR")
    expect(page.get_by_text("https://www.youtube.com/watch?v=dQw4w9WgXcR")).to_be_visible()

    send(page, prompt)

    expect_reply(page, "it sings")
    sent = str(youtube_instance.upstream.chat_requests()[-1]["messages"])
    assert "never gonna give you up" in sent
