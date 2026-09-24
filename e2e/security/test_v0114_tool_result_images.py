"""Regression: an image inside a tool result never showed in the reply.

open-webui 0.11.4, PR #29665 (issue #29208). Before `afda09454` only a tool result that was
exactly one image data URI became an image; one inside a returned object was serialised into
the model's context as base64 text and nothing was shown. Before `d372bec70` a shown tool image
sat inline in the saved chat as a data URL; it is now stored and served from the files API.

In native function calling, the default, tool images go to the model only. With function
calling set to legacy (a user setting) they are attached to the reply, so a user who picks the
tool from the Integrations menu sees the chart under the answer.

Twin of unit/security/test_v0114_tool_result_images.py.

Discriminates: passes on bbfa876af with its built frontend; with `afda09454` reverted in the
backend no image appears, with `d372bec70` reverted the image is an inline data URL.
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.python_tools import python_tool
from utils.chat_ui import conversation, expect_reply, send

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

TOOL_SOURCE = f'''
class Tools:
    def render_chart(self) -> dict:
        """Render the sales chart."""
        return {{"chart": {PNG!r}, "summary": "chart of sales"}}
'''


@pytest.fixture(scope="module")
def chart_tool(admin):
    with python_tool(admin, TOOL_SOURCE, name="Chart renderer") as tool_id:
        yield tool_id


@pytest.fixture
def legacy_tool_user(make_user):
    person = make_user()
    with person.client() as client:
        saved = client.post(
            "/api/v1/users/user/settings/update",
            json={"ui": {"params": {"function_calling": "legacy"}}},
        )
    assert saved.status_code == 200, saved.text
    return person


def test_an_image_returned_inside_a_tool_result_shows_under_the_reply(
    page_for, legacy_tool_user, upstream, chart_tool
):
    page = page_for(legacy_tool_user)
    page.get_by_label("Integrations").click()
    page.get_by_role("button", name=re.compile(r"^Tools")).click()
    page.get_by_role("button", name="Chart renderer").click()
    page.keyboard.press("Escape")

    # legacy mode asks the model which tool to call, then asks again for the answer
    upstream.queue(
        reply.text(json.dumps({"name": "render_chart", "parameters": {}})),
        reply.text("Here is the sales chart."),
    )
    send(page, "draw the sales chart")
    expect_reply(page, "Here is the sales chart.")

    chart = conversation(page).get_by_alt_text("Generated Image")
    expect(chart).to_be_visible()
    expect(chart).to_have_attribute("src", re.compile(r"/api/v1/files/[^/]+/content$"))
    page.wait_for_function(
        "image => image.complete && image.naturalWidth > 0", arg=chart.element_handle()
    )
