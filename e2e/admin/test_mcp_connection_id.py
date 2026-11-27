"""Journey: an admin adds an MCP tool server and its Connection ID follows from its name.

In Admin Settings > Integrations the Add Connection dialog for an MCP server asks for a
Connection ID, which fills itself with a slug of the Name as it is typed and can be replaced with
a name of the admin's own. An ID left empty is generated on save, so a connection is never stored
without one, and an ID with a colon or a bar is refused in the dialog's own words. An OpenAPI
connection keeps an optional "ID". Editing a saved connection shows its stored ID.

Discriminates: in a frontend build without the name-to-ID slug, the slug test and the generated
ID test go red (the ID stays empty); without the ID check message for MCP the refusal test goes
red.
"""

from __future__ import annotations

import pytest
from playwright.sync_api import Locator, Page, expect

from harness.mcp_server import TOOL_SERVERS, mcp_connection
from utils.tool_servers import choose_type
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

MCP_URL = "http://127.0.0.1:9/mcp"


def open_add_dialog(page: Page) -> Locator:
    page.goto("/admin/settings/integrations")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_text("External Tool Servers", exact=True)).to_be_visible()
    tooltip_button(settings, "Add Connection").click()
    form = page.get_by_role("dialog").filter(has=page.get_by_role("heading", name="Add Connection"))
    expect(form).to_be_visible()
    return form


def saved_connections(admin) -> list[dict]:
    with admin.client() as client:
        return client.get(TOOL_SERVERS[0]).json()["TOOL_SERVER_CONNECTIONS"]


def saved_mcp(admin, name: str) -> dict:
    [found] = [c for c in saved_connections(admin) if c["info"]["name"] == name]
    return found


@pytest.fixture
def admin_page(page_for, make_user, preserve):
    preserve(TOOL_SERVERS)
    return page_for(make_user(role="admin"))


def test_the_connection_id_follows_the_name_of_an_mcp_server(admin_page):
    form = open_add_dialog(admin_page)
    choose_type(form, "MCP")

    form.get_by_label("Name", exact=True).fill("Harbour Notes")

    expect(form.get_by_label("Connection ID")).to_have_value("harbour-notes")


def test_a_typed_connection_id_is_the_one_saved(admin_page, admin):
    form = open_add_dialog(admin_page)
    choose_type(form, "MCP")
    form.get_by_label("Name", exact=True).fill("Tide Table")
    form.get_by_label("Connection ID").fill("tides")
    form.get_by_label("URL", exact=True).fill(MCP_URL)

    form.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Connections saved successfully")).to_be_visible()

    assert saved_mcp(admin, "Tide Table")["info"]["id"] == "tides"


def test_an_empty_connection_id_is_generated_on_save(admin_page, admin):
    form = open_add_dialog(admin_page)
    choose_type(form, "MCP")
    form.get_by_label("Name", exact=True).fill("Pilot Boat")
    form.get_by_label("Connection ID").fill("")
    form.get_by_label("URL", exact=True).fill(MCP_URL)

    form.get_by_role("button", name="Save").click()
    expect(admin_page.get_by_text("Connections saved successfully")).to_be_visible()

    assert saved_mcp(admin, "Pilot Boat")["info"]["id"] == "pilot-boat"


def test_a_connection_id_with_a_colon_is_refused(admin_page, admin):
    form = open_add_dialog(admin_page)
    choose_type(form, "MCP")
    form.get_by_label("Name", exact=True).fill("Buoy Log")
    form.get_by_label("Connection ID").fill("buoy:log")
    form.get_by_label("URL", exact=True).fill(MCP_URL)

    form.get_by_role("button", name="Save").click()

    expect(
        admin_page.get_by_text('Connection ID cannot contain ":" or "|" characters')
    ).to_be_visible()
    expect(form).to_be_visible()
    assert not [c for c in saved_connections(admin) if c["info"]["name"] == "Buoy Log"]


def test_an_openapi_connection_keeps_an_optional_id(admin_page):
    form = open_add_dialog(admin_page)
    choose_type(form, "OpenAPI")

    expect(form.get_by_label("ID", exact=False).first).to_be_visible()
    expect(form.get_by_text("(optional)")).to_be_visible()
    expect(form.get_by_label("Connection ID")).to_have_count(0)


def test_editing_a_saved_connection_shows_its_stored_id(admin_page, admin):
    with admin.client() as client:
        saved = client.post(
            TOOL_SERVERS[1],
            json={"TOOL_SERVER_CONNECTIONS": [mcp_connection(MCP_URL, "customs-desk", [])]},
        )
    assert saved.status_code == 200, saved.text
    admin_page.goto("/admin/settings/integrations")
    settings = admin_page.get_by_role("dialog")
    expect(settings.get_by_text("customs-desk", exact=True)).to_be_visible()

    tooltip_button(settings, "Configure").click()

    form = admin_page.get_by_role("dialog").filter(
        has=admin_page.get_by_role("heading", name="Edit Connection")
    )
    expect(form.get_by_label("Connection ID")).to_have_value("customs-desk")
    expect(form.get_by_label("Name", exact=True)).to_have_value("customs-desk")
