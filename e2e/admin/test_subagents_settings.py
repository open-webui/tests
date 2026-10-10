"""Journey: an admin sets up sub-agents, and the model is offered delegation only while allowed.

Each setting of the tab (enable, background, max concurrent, max background, max iterations, max
output and the system prompt) is typed in, saved and read back after a reload of the tab and from
the settings API. A further instance shows the tab starting from the environment variables of the
same names, and another one from the built-in defaults. The switches decide what the tab shows:
the limits appear only while sub-agents are on and the background limit only while background
sub-agents are. Turning sub-agents off withdraws the delegation tool (and the timer that comes
with it) from what the model is offered, and turning background on adds the `background` argument
to it. Unticking Sub-agents in a model's editor withdraws the tool from that model alone.

Discriminates: passes on dev 176d31d1d. In a frontend copy the save and reload test fails when
the save sends the max output for the max iterations or the max iterations for the max
background, or when the tab reads the system prompt back as empty; the off-and-on tests fail when
the save always sends sub-agents or background as on, or resets the limits while off; the
visibility test fails when the limits show while off or the background limit while background is
off; the env test fails when the tab ignores what it loads; the model test fails when the
Sub-agents tick cannot be unticked. In a backend copy the offered-tools tests fail when the tool
is offered whatever the setting says (or never), when the model's tick is ignored, or when the
`background` argument is never stripped; the tab tests fail when the max output is not stored,
when a default differs or when its environment variable is ignored. The model test was
retargeted for 8d0ff76f2, whose Sub-agents switch sits in the editor's Builtin Tools section:
it passes on dev 76ad6f97c (3 of 3) and fails in a build of it whose save leaves the builtin
tools as loaded.
"""

from __future__ import annotations

import uuid

import pytest
from playwright.sync_api import Locator, Page, expect

from harness import upstream as reply
from harness.actors import admin_of
from harness.python_tools import EVERYONE_READS
from harness.upstream import MOCK_MODEL_ID
from utils.chat_ui import chat_input, expect_reply, send
from utils.model_editor import section

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

SUBAGENTS = ("/api/v1/configs/subagents", "/api/v1/configs/subagents")
BUILT_IN_DEFAULTS = {
    "ENABLE_SUBAGENTS": False,
    "SUBAGENTS_BACKGROUND_ENABLED": False,
    "SUBAGENTS_MAX_CONCURRENT": 20,
    "SUBAGENTS_MAX_ASYNC": 20,
    "SUBAGENTS_MAX_ITERATIONS": 30,
    "SUBAGENTS_MAX_OUTPUT": 30000,
    "SUBAGENTS_SYSTEM_PROMPT": "",
}
FROM_THE_ENVIRONMENT = {
    "ENABLE_SUBAGENTS": True,
    "SUBAGENTS_BACKGROUND_ENABLED": True,
    "SUBAGENTS_MAX_CONCURRENT": 7,
    "SUBAGENTS_MAX_ASYNC": 5,
    "SUBAGENTS_MAX_ITERATIONS": 12,
    "SUBAGENTS_MAX_OUTPUT": 4000,
    "SUBAGENTS_SYSTEM_PROMPT": "Answer in one line, as the harbour master would.",
}


@pytest.fixture
def admin_page(make_user, page_for) -> Page:
    """A fresh admin's page, so nothing here touches the shared admin's own settings."""
    return page_for(make_user(role="admin"))


def _open_tab(page: Page) -> Locator:
    """The Sub-agents settings, freshly loaded from the server."""
    page.goto("/admin/settings/subagents")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("switch", name="Enable sub-agents")).to_be_visible()
    return settings


def _save(page: Page, settings: Locator) -> None:
    settings.get_by_role("button", name="Save", exact=True).click()
    expect(page.get_by_text("Settings saved successfully!").first).to_be_visible()


def _fields(settings: Locator) -> dict[str, Locator]:
    return {
        "ENABLE_SUBAGENTS": settings.get_by_role("switch", name="Enable sub-agents"),
        "SUBAGENTS_BACKGROUND_ENABLED": settings.get_by_role(
            "switch", name="Enable background sub-agents"
        ),
        "SUBAGENTS_MAX_CONCURRENT": settings.get_by_label("Max concurrent"),
        "SUBAGENTS_MAX_ASYNC": settings.get_by_label("Max background"),
        "SUBAGENTS_MAX_ITERATIONS": settings.get_by_label("Max iterations"),
        "SUBAGENTS_MAX_OUTPUT": settings.get_by_label("Max output"),
        "SUBAGENTS_SYSTEM_PROMPT": settings.get_by_label("System prompt"),
    }


def _expect_shown(settings: Locator, values: dict) -> None:
    for name, field in _fields(settings).items():
        if isinstance(values[name], bool):
            expect(field).to_be_checked(checked=values[name])
        else:
            expect(field).to_have_value(str(values[name]))


def _stored(admin) -> dict:
    with admin.client() as client:
        return client.get(SUBAGENTS[0]).json()


def _offered(request: dict) -> dict[str, dict]:
    return {tool["function"]["name"]: tool["function"] for tool in request.get("tools") or []}


def _ask_in_new_chat(page: Page, question: str, upstream, *, model: str = MOCK_MODEL_ID) -> dict:
    """The request the provider got for `question`, asked in a chat of its own."""
    upstream.queue(reply.text("Noted.", match=reply.answering(question)))
    page.goto(f"/?models={model}")
    expect(chat_input(page)).to_be_visible()
    send(page, question)
    expect_reply(page, "Noted.")
    return next(filter(reply.answering(question), upstream.chat_requests()))


def test_every_subagent_setting_saves_and_survives_a_reload(admin_page, admin, preserve):
    preserve(SUBAGENTS)
    prompt = f"Report in two lines only, batch {uuid.uuid4().hex[:6]}."
    settings = _open_tab(admin_page)
    fields = _fields(settings)
    fields["ENABLE_SUBAGENTS"].click()
    fields["SUBAGENTS_BACKGROUND_ENABLED"].click()
    fields["SUBAGENTS_MAX_CONCURRENT"].fill("3")
    fields["SUBAGENTS_MAX_ASYNC"].fill("2")
    fields["SUBAGENTS_MAX_ITERATIONS"].fill("11")
    fields["SUBAGENTS_MAX_OUTPUT"].fill("5000")
    fields["SUBAGENTS_SYSTEM_PROMPT"].fill(prompt)
    _save(admin_page, settings)

    expected = {
        "ENABLE_SUBAGENTS": True,
        "SUBAGENTS_BACKGROUND_ENABLED": True,
        "SUBAGENTS_MAX_CONCURRENT": 3,
        "SUBAGENTS_MAX_ASYNC": 2,
        "SUBAGENTS_MAX_ITERATIONS": 11,
        "SUBAGENTS_MAX_OUTPUT": 5000,
        "SUBAGENTS_SYSTEM_PROMPT": prompt,
    }
    assert _stored(admin) == expected
    _expect_shown(_open_tab(admin_page), expected)


def test_a_switched_off_setting_stays_off_after_a_reload(admin_page, admin, preserve):
    preserve(SUBAGENTS)
    settings = _open_tab(admin_page)
    fields = _fields(settings)
    fields["ENABLE_SUBAGENTS"].click()
    fields["SUBAGENTS_BACKGROUND_ENABLED"].click()
    _save(admin_page, settings)
    assert _stored(admin)["SUBAGENTS_BACKGROUND_ENABLED"] is True

    settings = _open_tab(admin_page)
    _fields(settings)["SUBAGENTS_BACKGROUND_ENABLED"].click()
    _save(admin_page, settings)

    stored = _stored(admin)
    assert stored["ENABLE_SUBAGENTS"] is True
    assert stored["SUBAGENTS_BACKGROUND_ENABLED"] is False
    expect(_fields(_open_tab(admin_page))["SUBAGENTS_BACKGROUND_ENABLED"]).not_to_be_checked()


def test_switching_sub_agents_off_keeps_the_limits_for_when_they_come_back(
    admin_page, admin, preserve
):
    preserve(SUBAGENTS)
    settings = _open_tab(admin_page)
    fields = _fields(settings)
    fields["ENABLE_SUBAGENTS"].click()
    fields["SUBAGENTS_MAX_CONCURRENT"].fill("4")
    fields["SUBAGENTS_MAX_ITERATIONS"].fill("9")
    fields["SUBAGENTS_MAX_OUTPUT"].fill("8000")
    fields["SUBAGENTS_SYSTEM_PROMPT"].fill("Keep it short.")
    _save(admin_page, settings)

    settings = _open_tab(admin_page)
    _fields(settings)["ENABLE_SUBAGENTS"].click()
    _save(admin_page, settings)
    assert _stored(admin)["ENABLE_SUBAGENTS"] is False

    settings = _open_tab(admin_page)
    _fields(settings)["ENABLE_SUBAGENTS"].click()
    fields = _fields(settings)
    expect(fields["SUBAGENTS_MAX_CONCURRENT"]).to_have_value("4")
    expect(fields["SUBAGENTS_MAX_ITERATIONS"]).to_have_value("9")
    expect(fields["SUBAGENTS_MAX_OUTPUT"]).to_have_value("8000")
    expect(fields["SUBAGENTS_SYSTEM_PROMPT"]).to_have_value("Keep it short.")


def test_the_limits_show_only_while_their_switches_are_on(admin_page, preserve):
    preserve(SUBAGENTS)
    settings = _open_tab(admin_page)
    fields = _fields(settings)
    if fields["ENABLE_SUBAGENTS"].is_checked():
        fields["ENABLE_SUBAGENTS"].click()

    for name in list(fields)[1:]:
        expect(fields[name]).to_have_count(0)

    fields["ENABLE_SUBAGENTS"].click()
    for name in ("SUBAGENTS_MAX_CONCURRENT", "SUBAGENTS_MAX_ITERATIONS", "SUBAGENTS_MAX_OUTPUT"):
        expect(fields[name]).to_be_visible()
    expect(fields["SUBAGENTS_SYSTEM_PROMPT"]).to_be_visible()
    expect(fields["SUBAGENTS_BACKGROUND_ENABLED"]).not_to_be_checked()
    expect(fields["SUBAGENTS_MAX_ASYNC"]).to_have_count(0)

    fields["SUBAGENTS_BACKGROUND_ENABLED"].click()
    expect(fields["SUBAGENTS_MAX_ASYNC"]).to_be_visible()


@pytest.mark.slow
def test_the_tab_starts_from_the_environment_variables(instance_with, page_for):
    booted = instance_with(
        {
            "ENABLE_SUBAGENTS": "true",
            "SUBAGENTS_BACKGROUND_ENABLED": "true",
            "SUBAGENTS_MAX_CONCURRENT": "7",
            "SUBAGENTS_MAX_ASYNC": "5",
            "SUBAGENTS_MAX_ITERATIONS": "12",
            "SUBAGENTS_MAX_OUTPUT": "4000",
            "SUBAGENTS_SYSTEM_PROMPT": FROM_THE_ENVIRONMENT["SUBAGENTS_SYSTEM_PROMPT"],
        }
    )
    page = page_for(admin_of(booted))

    _expect_shown(_open_tab(page), FROM_THE_ENVIRONMENT)


@pytest.mark.slow
def test_the_tab_starts_from_the_built_in_defaults(instance_with, page_for):
    # a fresh instance, since the shared one's settings may have been changed
    booted = instance_with({"ENABLE_SUBAGENTS": "false"})
    page = page_for(admin_of(booted))

    settings = _open_tab(page)
    fields = _fields(settings)
    expect(fields["ENABLE_SUBAGENTS"]).not_to_be_checked()
    fields["ENABLE_SUBAGENTS"].click()
    fields["SUBAGENTS_BACKGROUND_ENABLED"].click()
    switched_on = {
        **BUILT_IN_DEFAULTS,
        "ENABLE_SUBAGENTS": True,
        "SUBAGENTS_BACKGROUND_ENABLED": True,
    }
    _expect_shown(settings, switched_on)
    with booted.client() as client:
        assert client.get(SUBAGENTS[0]).json() == BUILT_IN_DEFAULTS


def test_the_model_is_offered_delegation_only_while_sub_agents_are_on(
    admin_page, preserve, upstream
):
    preserve(SUBAGENTS)
    settings = _open_tab(admin_page)
    enable = _fields(settings)["ENABLE_SUBAGENTS"]
    if not enable.is_checked():
        enable.click()
        _save(admin_page, settings)
    on = _ask_in_new_chat(admin_page, "is delegation on?", upstream)

    settings = _open_tab(admin_page)
    _fields(settings)["ENABLE_SUBAGENTS"].click()
    _save(admin_page, settings)
    off = _ask_in_new_chat(admin_page, "is delegation off?", upstream)

    assert {"delegate_task", "timer"} <= set(_offered(on))
    assert not {"delegate_task", "timer"} & set(_offered(off))
    assert _offered(off), "the model was offered no tools at all, so this shows nothing"


def test_the_background_switch_decides_whether_the_model_may_delegate_in_the_background(
    admin_page, preserve, upstream
):
    preserve(SUBAGENTS)
    settings = _open_tab(admin_page)
    fields = _fields(settings)
    if not fields["ENABLE_SUBAGENTS"].is_checked():
        fields["ENABLE_SUBAGENTS"].click()
    if fields["SUBAGENTS_BACKGROUND_ENABLED"].is_checked():
        fields["SUBAGENTS_BACKGROUND_ENABLED"].click()
    _save(admin_page, settings)
    foreground_only = _ask_in_new_chat(admin_page, "may you delegate in the foreground?", upstream)

    settings = _open_tab(admin_page)
    _fields(settings)["SUBAGENTS_BACKGROUND_ENABLED"].click()
    _save(admin_page, settings)
    with_background = _ask_in_new_chat(admin_page, "may you delegate in the background?", upstream)

    def arguments(request: dict) -> set[str]:
        return set(_offered(request)["delegate_task"]["parameters"]["properties"])

    assert "background" not in arguments(foreground_only)
    assert "background" in arguments(with_background)
    assert {"task", "context"} <= arguments(foreground_only)


@pytest.fixture
def model_with_delegation(admin, preserve):
    """A model every account may use, with delegation on and the model's own tick untouched."""
    preserve(SUBAGENTS)
    model_id = f"helper-{uuid.uuid4().hex[:8]}"
    with admin.client() as client:
        current = client.get(SUBAGENTS[0]).json()
        client.post(SUBAGENTS[1], json={**current, "ENABLE_SUBAGENTS": True}).raise_for_status()
        created = client.post(
            "/api/v1/models/create",
            json={
                "id": model_id,
                "base_model_id": MOCK_MODEL_ID,
                "name": model_id,
                "meta": {},
                "params": {},
                "access_grants": [EVERYONE_READS],
            },
        )
        assert created.status_code == 200, created.text
        client.get("/api/models", params={"refresh": "true"}).raise_for_status()
        yield model_id
        client.post("/api/v1/models/model/delete", json={"id": model_id})


def test_unticking_sub_agents_in_a_models_editor_withdraws_delegation_from_that_model(
    admin_page, model_with_delegation, upstream
):
    admin_page.goto(f"/workspace/models/edit?id={model_with_delegation}")
    editor = admin_page.get_by_role("main")
    subagents_tick = section(editor, "Builtin Tools").get_by_role("switch", name="Sub-agents")
    expect(subagents_tick).to_have_attribute("aria-checked", "true")
    subagents_tick.click()
    admin_page.get_by_role("button", name="Save & Update").click()
    expect(admin_page.get_by_role("button", name="Save & Update")).to_have_count(0)

    without = _ask_in_new_chat(
        admin_page, "may this model delegate?", upstream, model=model_with_delegation
    )
    plain = _ask_in_new_chat(admin_page, "may the plain model delegate?", upstream)

    assert not {"delegate_task", "timer"} & set(_offered(without))
    assert _offered(without), "the model was offered no tools at all, so this shows nothing"
    assert "delegate_task" in _offered(plain)
