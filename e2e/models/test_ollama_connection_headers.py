"""Regression: verifying an Ollama connection behind a gateway failed though its chats worked.

Fix commit `bc2416c5d` (open-webui/open-webui#29868). The connection dialog's verify button sent
only the URL and key for an Ollama connection, and the server's check sent only the key, so the
custom headers a gateway such as Cloudflare Access asks for never reached it and the check was
refused. The dialog now sends the connection's headers and authentication type, and the check
applies them.

The Ollama stand-in refuses `/api/version` without the gateway header, as the gateway would. The
admin opens the connection's settings and presses verify.

Twin of integration/models/test_ollama_connection_headers.py.

Discriminates: passes on the dev bc2416c5d build and backend; with the verify handler of the
connection dialog back to sending the URL and key alone, the check is refused and the dialog
reports the error.
Retargeted for 6e3e87b11, which shows a saved connection's address as a button.
"""

from __future__ import annotations

import re

import pytest
from playwright.sync_api import expect

from harness.listener import ReceivedRequest, json_answer
from harness.ollama_provider import OLLAMA_CONFIG, connect_ollama, serve_ollama
from utils.admin_connections import connection_row
from utils.tooltips import tooltip_button

pytestmark = [pytest.mark.regression, pytest.mark.requires_browser, pytest.mark.requires_source]

GATEWAY_HEADER = "CF-Access-Client-Id"
GATEWAY_VALUE = "gateway-client.access"


def gated_version(server):
    def answer(request: ReceivedRequest):
        sent = {name.lower(): value for name, value in request.headers.items()}
        if sent.get(GATEWAY_HEADER.lower()) != GATEWAY_VALUE:
            return json_answer({"error": "blocked by the access gateway"}, status=403)
        return json_answer({"version": server.version})

    return answer


@pytest.fixture
def gated_ollama(admin, preserve, listener):
    preserve(OLLAMA_CONFIG)
    server = serve_ollama(listener, "llama3:latest")
    listener.route("GET", "/api/version", gated_version(server))
    with admin.client() as client:
        connect_ollama(client, listener, headers={GATEWAY_HEADER: GATEWAY_VALUE})
    return server


def test_verifying_a_gated_ollama_connection_passes_the_gateway(page_for, make_user, gated_ollama):
    page = page_for(make_user(role="admin"), path="/admin/settings/connections")
    settings = page.get_by_role("dialog")
    expect(settings.get_by_role("heading", name="Connections", exact=True)).to_be_visible()

    ollama_row = connection_row(settings, gated_ollama.listener.base_url)
    tooltip_button(ollama_row, "Configure").click()
    editing = page.get_by_role("dialog").filter(has_text="Edit Connection")
    editing.get_by_role("button", name="Advanced").click()
    headers = editing.get_by_placeholder("Enter additional headers in JSON format")
    expect(headers).to_have_value(re.compile(re.escape(GATEWAY_VALUE)))
    editing.get_by_role("button", name="Verify Connection").click()

    expect(page.get_by_text("Server connection verified")).to_be_visible()
    verified = gated_ollama.listener.requests_to("/api/version")[-1]
    sent = {name.lower(): value for name, value in verified.headers.items()}
    assert sent.get(GATEWAY_HEADER.lower()) == GATEWAY_VALUE
