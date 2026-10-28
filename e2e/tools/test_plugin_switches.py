"""Journey: with ENABLE_TOOLS, ENABLE_FUNCTIONS and ENABLE_TOOL_SERVERS off, their areas are gone.

Since 0.12.0 each switch also hides its own part of the web client: the workspace loses its
Tools tab and page, the admin panel its Functions tab and page, and Admin Settings > Integrations
greys out External Tool Servers, so no server can be added. One instance has all three off;
the shared instance, with all on, shows each of them. integration/tools/test_plugin_switches.py
covers what each switch withholds over the API and that each leaves the other kinds working.

Discriminates: on dev 948432b43, with the three flags of `/api/config` reported as on in a
backend copy, every switched-off test fails (the tabs show and the dialog's Add Connection is
enabled).
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import Page, expect

from harness.actors import Actor, create_user
from harness.instance import LaunchedInstance
from utils.tool_servers import open_admin_integrations
from utils.tooltips import tooltip_button

pytestmark = [
    pytest.mark.journey,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

ALL_OFF = {"ENABLE_TOOLS": "false", "ENABLE_FUNCTIONS": "false", "ENABLE_TOOL_SERVERS": "false"}


@pytest.fixture
def plugins_off(instance_with) -> LaunchedInstance:
    return instance_with(ALL_OFF)


@pytest.fixture(params=["on", "off"])
def switched(request, instance, plugins_off) -> tuple[bool, Actor]:
    """(whether the switches are off, a fresh admin of that instance)."""
    off = request.param == "off"
    return off, create_user(plugins_off if off else instance, role="admin")


def _tab(page: Page, name: str):
    return page.get_by_role("link", name=re.compile(rf"^{name}\b"))


def test_the_workspace_shows_tools_only_while_tools_are_on(page_for, switched):
    off, admin = switched
    page = page_for(admin, path="/workspace/models")

    expect(_tab(page, "Models")).to_be_visible()
    if off:
        expect(_tab(page, "Tools")).to_have_count(0)
    else:
        expect(_tab(page, "Tools")).to_be_visible()


def test_the_tools_page_opens_only_while_tools_are_on(page_for, switched):
    off, admin = switched
    page = page_for(admin, path="/workspace/tools")

    if off:
        expect(page).to_have_url(re.compile(r"/workspace/models$"))
    else:
        expect(page).to_have_url(re.compile(r"/workspace/tools$"))
        expect(page.get_by_role("heading", name="Tools", level=1)).to_be_visible()


def test_the_admin_panel_shows_functions_only_while_functions_are_on(page_for, switched):
    off, admin = switched
    page = page_for(admin, path="/admin/users")

    expect(_tab(page, "Evaluations")).to_be_visible()
    if off:
        expect(_tab(page, "Functions")).to_have_count(0)
    else:
        expect(_tab(page, "Functions")).to_be_visible()


def test_the_functions_page_opens_only_while_functions_are_on(page_for, switched):
    off, admin = switched
    page = page_for(admin, path="/admin/functions")

    if off:
        expect(page).to_have_url(re.compile(r"/admin/users$"))
    else:
        expect(page.get_by_placeholder("Search Functions")).to_be_visible()
        expect(page).to_have_url(re.compile(r"/admin/functions$"))


def test_tool_servers_can_be_added_only_while_tool_servers_are_on(page_for, switched):
    off, admin = switched
    page = page_for(admin, path="/admin/users")
    settings = open_admin_integrations(page)

    add = tooltip_button(settings, "Add Connection")
    if off:
        expect(add).to_be_disabled()
    else:
        expect(add).to_be_enabled()
