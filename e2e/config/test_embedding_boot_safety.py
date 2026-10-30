"""Regression: the local embedding engine still answers a chat once the missing-model check moved.

PR #25683 (issues #25634, #25165) moved the "no embedding model is loaded" error from building the
embedding function to calling it, so a blank model no longer bricks boot. With a model loaded the
local engine has to keep embedding: on an instance booted on a small SentenceTransformer on disk,
a fresh admin builds a knowledge base in the workspace, attaches it to a chat with `#` and the
model is sent the file's text.

Twin of unit/config/test_embedding_boot_safety.py.

Discriminates: passes on dev ef67cc3fa; in a backend copy, with the use-time check raising
whenever the local engine is selected the model is sent the question alone.
"""

from __future__ import annotations

import json
import re
import uuid

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import create_user
from harness.local_embedding import local_embedding_env, save_keyword_model
from utils.chat_ui import chat_input, expect_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

FILE_NAME = "orchard.txt"
FILE_TEXT = "The orchard gate code is 5151.\n"
QUESTION = "what is the orchard gate code?"


@pytest.fixture(scope="module")
def local_instance(instance_with, tmp_path_factory):
    model = save_keyword_model(tmp_path_factory.mktemp("local-embedding"), ["orchard", "gate"])
    launched = instance_with(local_embedding_env(model))
    if not launched.serves_frontend:
        pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
    return launched


def test_a_knowledge_base_embedded_locally_answers_a_chat(page_for, local_instance):
    name = f"Orchard {uuid.uuid4().hex[:6]}"
    page = page_for(create_user(local_instance, role="admin"), path="/workspace/knowledge/create")
    creating = page.get_by_role("dialog")
    creating.get_by_role("textbox", name="What are you working on?").fill(name)
    creating.get_by_role("textbox", name="What are you trying to achieve?").fill("the orchard")
    creating.get_by_role("button", name="Create Knowledge").click()
    expect(page).to_have_url(re.compile(r"/workspace/knowledge/[0-9a-f-]+$"))

    base = page.get_by_role("main")
    base.get_by_role("button", name="Add Content").first.click()
    with page.expect_file_chooser() as chooser:
        page.get_by_role("menu").get_by_role("menuitem", name="Upload files").click()
    chooser.value.set_files(
        {"name": FILE_NAME, "mimeType": "text/plain", "buffer": FILE_TEXT.encode()}
    )
    expect(base.get_by_role("button", name=re.compile(re.escape(FILE_NAME)))).to_be_visible()

    upstream = local_instance.upstream
    upstream.queue(reply.text("It is 5151.", match=reply.answering(QUESTION)))
    page.goto("/")
    chat_input(page).click()
    page.keyboard.type("#Orchard")
    page.get_by_role("tooltip").get_by_role("button", name=name).click()
    send(page, QUESTION)
    expect_reply(page, "It is 5151.")

    assert "gate code is 5151" in json.dumps(upstream.chat_requests()[-1]["messages"])
