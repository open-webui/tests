"""Journey: a person creates a terminal on a Terminals connection from the chat input.

New on dev after 0.13.3 (6f2fc7bc7, titled `refac`, changed eight times up to c15f28649): a
person with write access to a Terminals connection finds New terminal (+) beside the Terminals
heading of the chat input's Terminal menu. The dialog asks for a name, for what gets a machine
of its own (nobody, each user, each chat, each model) and, under Advanced, for a start
directory, a setup script, an image, CPU, memory, storage, a network and environment variables.
Advanced reads what is set there on one line. The connection's limits show under CPU, memory
and storage and a value past them cannot be created; the image and network fields suggest what
the connection allows. With two connections to create on, Server chooses one. Create stores the
terminal on the orchestrator and picks it for the chat; a refusal shows the server's message in
the dialog.

The real Terminals service runs on Docker (`harness.terminals`); without its images or Docker
the tests skip.

`test_the_first_terminal_of_a_connection_is_created_from_the_chat_input` is red on dev
c15f28649: the chat input shows no Terminal menu until the account has a terminal, so nothing
leads to New terminal on a new connection (finding
`open-webui-no-terminal-menu-before-the-first-terminal`). The other tests start with one
terminal on the connection for that reason.

Discriminates: the other seven pass on the dev c15f28649 build with Terminals 0.4.1 and Open
Terminal 0.17.7. Each fails on a build of a frontend copy with one change to the dialog: the
name no longer checked (the name test: no "Enter a name."); the choice of what gets a machine
not sent (the per chat test: stored without it); the start directory not sent (the Advanced
test: missing from the stored variables) and CPU left out of the Advanced line (the same test);
the limit no longer compared (the limits test: no "Maximum is 2."); Server shown only from
three connections (the two connections test, and the no answer test, which chooses a server
too); wildcard images suggested (the two connections test); the limits of the last server that
answered kept (the no answer test: "Up to 2." still shows); the error of a failed request not
shown (the refused create test, and the name test, whose message takes the same way).
"""

from __future__ import annotations

import json
import os
import uuid

import pytest
from playwright.sync_api import Page, expect

from harness.terminal_server import configure_terminals, write_grant
from harness.terminals import (
    MAX_CPU,
    MAX_MEMORY,
    OPEN_TERMINAL_IMAGE_ENV,
    create_template,
    execute,
    output_of,
    serving_terminals,
)
from utils.user_terminals import (
    advanced,
    create_terminal,
    hint_of,
    listed_terminal,
    new_terminal_dialog,
    open_terminal_menu,
    show_advanced,
    start_new_terminal,
    suggestions_of,
)

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]

DEFAULT_IMAGES = [
    f"ghcr.io/open-webui/open-terminal:{tag}"
    for tag in ("latest", "agents", "cuda", "slim", "alpine", "openshift")
]


@pytest.fixture
def writer_page(page_for, make_user, admin, connect_terminals):
    """A person who may create terminals on the connected orchestrator, on a new chat; the
    connection already has a terminal. Yields (page, person, connection)."""
    writer = make_user()
    connection = connect_terminals(write_grant(writer.id))
    with admin.client() as client:
        create_template(client, connection, name=f"Berth {uuid.uuid4().hex[:8]}")
    return page_for(writer), writer, connection


def _stored(terminals, terminal_id: str) -> dict:
    """What the orchestrator keeps of the terminal."""
    template_id = terminal_id.rsplit(":", 1)[1]
    [template] = [row for row in terminals.templates() if row["id"] == template_id]
    return template["data"]


def _names(terminals) -> list[str]:
    return sorted(template["data"]["name"] for template in terminals.templates())


def _create(page: Page) -> None:
    new_terminal_dialog(page).get_by_role("button", name="Create", exact=True).click()


def test_the_first_terminal_of_a_connection_is_created_from_the_chat_input(
    page_for, make_user, connect_terminals, terminals
):
    writer = make_user()
    connect_terminals(write_grant(writer.id))
    name = f"First {uuid.uuid4().hex[:8]}"
    page = page_for(writer)

    dialog = start_new_terminal(page)
    dialog.get_by_label("Name", exact=True).fill(name)
    create_terminal(page, dialog)

    assert _names(terminals) == [name], f"the orchestrator holds {_names(terminals)}"


def test_a_new_terminal_needs_a_name_and_is_picked_for_the_chat_once_created(
    writer_page, terminals
):
    page, writer, _ = writer_page
    name = f"Quay {uuid.uuid4().hex[:8]}"
    before = _names(terminals)

    dialog = start_new_terminal(page)
    expect(dialog.get_by_label("Server")).to_have_count(0)
    expect(advanced(dialog)).to_contain_text("Default")
    dialog.get_by_label("Name", exact=True).fill("   ")
    _create(page)
    expect(dialog.get_by_role("alert")).to_have_text("Enter a name.")
    assert _names(terminals) == before, (
        f"a terminal without a name was created: {_names(terminals)}"
    )

    dialog.get_by_label("Name", exact=True).fill(f"  {name} ")
    terminal_id = create_terminal(page, dialog)

    expect(page.get_by_role("main").get_by_role("button", name=name, exact=True)).to_be_visible()
    open_terminal_menu(page, picked=name)
    expect(listed_terminal(page, name)).to_have_attribute("aria-checked", "true")
    stored = _stored(terminals, terminal_id)
    unset = {field: value for field, value in stored.items() if field != "name" and value}
    assert stored["name"] == name, f"the terminal is stored as {stored['name']!r}"
    assert not unset, f"a terminal created with a name alone has settings of its own: {unset}"


def test_separate_machine_per_explains_each_choice_and_the_terminal_keeps_the_one_made(
    writer_page, terminals
):
    page, writer, _ = writer_page
    name = f"Per chat quay {uuid.uuid4().hex[:8]}"
    dialog = start_new_terminal(page)
    choice = dialog.get_by_label("Separate machine per", exact=True)
    hints = {
        "None": "All chats share one machine.",
        "User": "Each person gets their own machine.",
        "Model": "Each model gets its own machine.",
        "Chat": "Each chat gets its own machine.",
    }

    expect(choice.locator("option:checked")).to_have_text("None")
    for label, hint in hints.items():
        choice.select_option(label=label)
        expect(dialog.get_by_text(hint, exact=True)).to_be_visible()
    dialog.get_by_label("Name", exact=True).fill(name)
    terminal_id = create_terminal(page, dialog)

    assert _stored(terminals, terminal_id).get("scope") == "chat", (
        f"the terminal made per chat is stored as {_stored(terminals, terminal_id)}"
    )
    with writer.client() as client:
        listed = {entry["id"]: entry["scope"] for entry in client.get("/api/v1/terminals/").json()}
    assert listed[terminal_id] == "chat", f"the Terminal menu is told its scope is {listed}"


def test_what_is_set_under_advanced_is_summed_up_and_reaches_the_machine(writer_page, terminals):
    page, writer, _ = writer_page
    tag = uuid.uuid4().hex[:8]
    name = f"Fitted quay {tag}"
    image = os.environ[OPEN_TERMINAL_IMAGE_ENV]
    start_folder = f"/home/user/hold-{tag}"
    variable = f"CARGO_{tag.upper()}"
    dialog = start_new_terminal(page)
    dialog.get_by_label("Name", exact=True).fill(name)
    show_advanced(dialog)

    dialog.get_by_label("Start directory", exact=True).fill(start_folder)
    dialog.get_by_label("Setup script", exact=True).fill(f"echo stowed-{tag} > setup.txt")
    dialog.get_by_label("Image", exact=True).fill(image)
    dialog.get_by_label("CPU", exact=True).fill("0.5")
    dialog.get_by_label("Memory", exact=True).fill("256Mi")
    rows = (
        (variable, f"tea-{tag}"),
        ("", "a value without a name"),
        ("OPEN_TERMINAL_START_DIR", "/tmp"),
    )
    for index, (key, value) in enumerate(rows):
        dialog.get_by_role("button", name="Add variable", exact=True).click()
        dialog.get_by_label("Variable name", exact=True).nth(index).fill(key)
        dialog.get_by_label("Variable value", exact=True).nth(index).fill(value)
    dialog.get_by_role("button", name="Add variable", exact=True).click()
    expect(dialog.get_by_label("Variable name", exact=True)).to_have_count(4)
    dialog.get_by_role("button", name="Remove variable", exact=True).last.click()
    expect(dialog.get_by_label("Variable name", exact=True)).to_have_count(3)

    summary = " · ".join(
        [start_folder, "Setup script", image.rsplit(":", 1)[1], "0.5 CPU", "256Mi", "2 variables"]
    )
    expect(advanced(dialog)).to_contain_text(summary)
    terminal_id = create_terminal(page, dialog)

    stored = _stored(terminals, terminal_id)
    assert stored["env"] == {
        variable: f"tea-{tag}",
        "OPEN_TERMINAL_START_DIR": start_folder,
        "OPEN_TERMINAL_SETUP_SCRIPT": f"echo stowed-{tag} > setup.txt",
    }, f"the terminal's variables are stored as {stored['env']}"
    assert (stored["image"], stored["cpu_limit"], stored["memory_limit"]) == (image, "0.5", "256Mi")
    with writer.client() as client:
        printed = output_of(
            execute(client, terminal_id, f"pwd && cat setup.txt && printenv {variable}")
        )
    assert printed.split() == [start_folder, f"stowed-{tag}", f"tea-{tag}"], (
        f"a command in the new terminal's machine printed {printed!r}"
    )
    [machine] = [
        m for m in terminals.instances() if m["template_id"] == terminal_id.rsplit(":")[-1]
    ]
    assert terminals.limits(machine["id"]) == (256 * 2**20, 500_000_000), (
        f"Docker holds the machine to {terminals.limits(machine['id'])}"
    )


def test_cpu_memory_and_storage_are_checked_against_what_the_connection_allows(
    writer_page, terminals
):
    page, _, _ = writer_page
    name = f"Sized quay {uuid.uuid4().hex[:8]}"
    dialog = start_new_terminal(page)
    dialog.get_by_label("Name", exact=True).fill(name)
    show_advanced(dialog)
    create = dialog.get_by_role("button", name="Create", exact=True)
    expect(hint_of(dialog, "CPU")).to_have_text(
        f"CPU limit, for example 1 or 0.5. Up to {MAX_CPU}."
    )
    expect(hint_of(dialog, "Memory")).to_have_text(
        f"Memory limit, for example 2Gi. Up to {MAX_MEMORY}."
    )
    expect(hint_of(dialog, "Storage")).to_have_text("Disk space for files, for example 10Gi.")
    expect(create).to_be_enabled()

    refused = (
        ("CPU", "two", "Enter a number, like 2."),
        ("CPU", "0", "Enter a number, like 2."),
        ("CPU", "2.5", f"Maximum is {MAX_CPU}."),
        ("Memory", "plenty", "Enter a size, like 2Gi."),
        ("Memory", "2Gi", f"Maximum is {MAX_MEMORY}."),
        ("Storage", "ten", "Enter a size, like 2Gi."),
    )
    for field, value, problem in refused:
        dialog.get_by_label(field, exact=True).fill(value)
        expect(hint_of(dialog, field)).to_have_text(problem)
        expect(dialog.get_by_label(field, exact=True)).to_have_attribute("aria-invalid", "true")
        expect(create).to_be_disabled()
        dialog.get_by_label(field, exact=True).fill("")
        expect(create).to_be_enabled()

    dialog.get_by_label("CPU", exact=True).fill(MAX_CPU)
    dialog.get_by_label("Memory", exact=True).fill("1024Mi")
    expect(hint_of(dialog, "Memory")).to_have_text(
        f"Memory limit, for example 2Gi. Up to {MAX_MEMORY}."
    )
    terminal_id = create_terminal(page, dialog)

    stored = _stored(terminals, terminal_id)
    assert (stored["cpu_limit"], stored["memory_limit"]) == (MAX_CPU, "1024Mi"), (
        f"the terminal at the connection's limits is stored as {stored}"
    )


def test_a_create_the_server_refuses_shows_its_message_and_keeps_the_dialog(writer_page, terminals):
    page, _, _ = writer_page
    tag = uuid.uuid4().hex[:8]
    name = f"Moored quay {tag}"
    before = _names(terminals)
    dialog = start_new_terminal(page)
    dialog.get_by_label("Name", exact=True).fill(name)
    show_advanced(dialog)
    expect(dialog.get_by_role("alert")).to_have_count(0)

    dialog.get_by_label("Network", exact=True).fill(f"nowhere-{tag}")
    _create(page)

    expect(dialog.get_by_role("alert")).to_have_text(f"Unknown network profile: 'nowhere-{tag}'")
    expect(dialog.get_by_label("Name", exact=True)).to_have_value(name)
    assert _names(terminals) == before, f"the refused terminal exists: {_names(terminals)}"

    dialog.get_by_label("Network", exact=True).fill("")
    create_terminal(page, dialog)
    assert _names(terminals) == sorted([*before, name])


@pytest.mark.slow
def test_with_two_connections_the_server_is_chosen_and_its_limits_and_suggestions_show(
    page_for, make_user, admin, connect_terminals, terminals
):
    writer = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Far quay {tag}"
    image = os.environ[OPEN_TERMINAL_IMAGE_ENV]
    first = connect_terminals(write_grant(writer.id), name=f"Near servers {tag}")
    settings = {
        "TERMINALS_MAX_CPU": "1",
        "TERMINALS_MAX_MEMORY": "768Mi",
        "TERMINALS_MAX_STORAGE": "5Gi",
        "TERMINALS_ALLOWED_IMAGES": f"{image},ghcr.io/example/*",
        "TERMINALS_NETWORK_PROFILES": json.dumps({f"inland-{tag}": {"docker_network": "bridge"}}),
    }
    with serving_terminals(settings) as far_terminals:
        second = far_terminals.connection(
            name=f"Far servers {tag}", config={"access_grants": [write_grant(writer.id)]}
        )
        with admin.client() as client:
            configure_terminals(client, first, second)
            create_template(client, first, name=f"Berth {tag}")
        page = page_for(writer)
        dialog = start_new_terminal(page)
        server = dialog.get_by_label("Server", exact=True)
        expect(server.locator("option")).to_have_text([first["name"], second["name"]])
        expect(server.locator("option:checked")).to_have_text(first["name"])
        show_advanced(dialog)
        expect(hint_of(dialog, "CPU")).to_contain_text(f"Up to {MAX_CPU}.")
        assert suggestions_of(dialog, "Image") == DEFAULT_IMAGES
        assert suggestions_of(dialog, "Network") == []

        server.select_option(label=second["name"])

        expect(hint_of(dialog, "CPU")).to_contain_text("Up to 1.")
        expect(hint_of(dialog, "Memory")).to_contain_text("Up to 768Mi.")
        expect(hint_of(dialog, "Storage")).to_contain_text("Up to 5Gi.")
        assert suggestions_of(dialog, "Image") == [image]
        assert suggestions_of(dialog, "Network") == [f"inland-{tag}"]
        dialog.get_by_label("Storage", exact=True).fill("6Gi")
        expect(hint_of(dialog, "Storage")).to_have_text("Maximum is 5Gi.")
        dialog.get_by_label("Storage", exact=True).fill("")
        dialog.get_by_label("Name", exact=True).fill(name)
        try:
            terminal_id = create_terminal(page, dialog)

            assert terminal_id.startswith(f"template:{second['id']}:"), terminal_id
            assert _names(far_terminals) == [name], (
                f"the chosen server holds {_names(far_terminals)}"
            )
            assert name not in _names(terminals), "the terminal was created on the other server"
        finally:
            with admin.client() as client:
                for template in far_terminals.templates():
                    client.delete(
                        f"/api/v1/terminals/templates/template:{second['id']}:{template['id']}"
                    )


@pytest.mark.slow
def test_without_an_answer_from_the_server_no_limits_show_and_create_says_so(
    page_for, make_user, admin, connect_terminals, terminals
):
    writer = make_user()
    tag = uuid.uuid4().hex[:8]
    first = connect_terminals(write_grant(writer.id), name=f"Near servers {tag}")
    with serving_terminals() as lost_terminals:
        second = lost_terminals.connection(
            name=f"Lost servers {tag}", config={"access_grants": [write_grant(writer.id)]}
        )
        with admin.client() as client:
            configure_terminals(client, first, second)
            create_template(client, first, name=f"Berth {tag}")
        page = page_for(writer)
        dialog = start_new_terminal(page)
        dialog.get_by_label("Server", exact=True).select_option(label=second["name"])
        show_advanced(dialog)
        expect(hint_of(dialog, "CPU")).to_have_text(
            f"CPU limit, for example 1 or 0.5. Up to {MAX_CPU}."
        )
        dialog.get_by_role("button", name="Close", exact=True).click()
        expect(dialog).to_be_hidden()

        lost_terminals.stop()
        dialog = start_new_terminal(page)
        dialog.get_by_label("Server", exact=True).select_option(label=second["name"])
        show_advanced(dialog)

        expect(hint_of(dialog, "CPU")).to_have_text("CPU limit, for example 1 or 0.5.")
        expect(hint_of(dialog, "Memory")).to_have_text("Memory limit, for example 2Gi.")
        dialog.get_by_label("CPU", exact=True).fill("64")
        expect(hint_of(dialog, "CPU")).to_have_text("CPU limit, for example 1 or 0.5.")
        dialog.get_by_label("Name", exact=True).fill(f"Adrift {tag}")
        _create(page)
        expect(dialog.get_by_role("alert")).to_have_text("Terminals is unavailable")
        expect(dialog.get_by_label("Name", exact=True)).to_have_value(f"Adrift {tag}")
