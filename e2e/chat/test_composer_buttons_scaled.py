"""At a larger UI Scale on a phone the model name covered the composer's buttons, open-webui#29989.

Fix commit `e6e9fc972` (PR open-webui/open-webui#30374, issue open-webui/open-webui#29989). The
model selector's width cap is rem based, so it grows with UI Scale, and the group holding the +
and Integrations buttons was allowed to shrink to nothing. On a 360px wide phone at 1.2x the
selector covered the Integrations button, from 1.3x the + button too, leaving no way to attach a
file. The two buttons now sit in a group that keeps its width and the model name is truncated.
Since 834c47ee0 the Integrations menu lives inside the + menu, so the + button is the one left.

Discriminates: passes on the efe63bd34 build, fails on it with `e6e9fc972` reverted (a tap on
the button's centre lands on the model selector).
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

PHONE = {"width": 360, "height": 780}
LONG_NAME = "Research assistant with a rather long display name"

# true when a tap on the element's centre reaches the element itself
TAP_REACHES = """(element) => {
    const box = element.getBoundingClientRect();
    const hit = document.elementFromPoint(box.left + box.width / 2, box.top + box.height / 2);
    return box.width > 0 && element.contains(hit);
}"""


@pytest.fixture
def long_named_model(admin) -> str:
    model_id = f"long-{uuid.uuid4().hex[:8]}"
    form = {
        "id": model_id,
        "base_model_id": MOCK_MODEL_ID,
        "name": LONG_NAME,
        "meta": {},
        "params": {},
        "access_grants": [EVERYONE_READS],
    }
    with admin.client() as client:
        created = client.post("/api/v1/models/create", json=form)
        assert created.status_code == 200, created.text
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def phone_page(page_for, make_user, model_id: str, scale: float) -> Page:
    account = make_user()
    ui = {"models": [model_id], "textScale": scale}
    with account.client() as client:
        client.post("/api/v1/users/user/settings/update", json={"ui": ui}).raise_for_status()
    page = page_for(account)
    page.set_viewport_size(PHONE)
    page.reload()
    expect(chat_input(page)).to_be_visible()
    return page


def composer_buttons(page: Page) -> dict[str, Locator]:
    return {"+": page.locator("#input-menu-button")}


def covered(page: Page) -> list[str]:
    return [
        name for name, button in composer_buttons(page).items() if not button.evaluate(TAP_REACHES)
    ]


@pytest.mark.parametrize("scale", [1.2, 1.3, 1.5])
def test_the_composer_buttons_stay_tappable_at_a_larger_scale(
    page_for, make_user, long_named_model, scale
):
    page = phone_page(page_for, make_user, long_named_model, scale)
    expect(page.get_by_role("button", name=f"Selected model: {LONG_NAME}")).to_be_visible()
    for button in composer_buttons(page).values():
        expect(button).to_be_visible()

    assert covered(page) == [], f"at {scale}x the model selector covers these buttons"


def test_a_tap_on_the_plus_button_opens_its_menu_at_1_5x(page_for, make_user, long_named_model):
    page = phone_page(page_for, make_user, long_named_model, 1.5)

    composer_buttons(page)["+"].click()

    expect(page.get_by_role("menu").get_by_role("menuitem", name="Upload Files")).to_be_visible()


def test_the_composer_buttons_are_tappable_at_the_default_scale(
    page_for, make_user, long_named_model
):
    page = phone_page(page_for, make_user, long_named_model, 1.0)
    assert covered(page) == []
