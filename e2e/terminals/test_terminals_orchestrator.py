"""Journey: a person picks a terminal of a Terminals orchestrator and the model works in it.

New on dev after 0.13.3 (titled `refac`, 404d27a36 to 473a0fb24): an admin connects a Terminals
orchestrator once, and each terminal on it is a template shown by its name in the chat input's
Terminal menu. A person picks a terminal the admin shared with them and asks the model to run a
command; the model's `run_command` runs in the machine the terminal gives that person, started
by this first use, and its output goes back to the model.

The real Terminals service (open-webui/terminals) runs on Docker and starts a real Open Terminal
container per machine (`harness.terminals`); without its images or Docker the test skips.

Discriminates: passes on the dev c15f28649 build with Terminals 0.4.1 and Open Terminal 0.17.7;
with the template header left out of what the backend sends the orchestrator, in a backend copy,
the terminal has no `run_command` to offer the model and the test fails.
"""

from __future__ import annotations

import json
import uuid

import pytest

from harness import upstream as reply
from harness.terminal_server import read_grant
from harness.terminals import create_template, share_template
from harness.upstream import conversation
from utils.cached_chat import ask, calling, choose_terminal

pytestmark = [pytest.mark.journey, pytest.mark.requires_browser, pytest.mark.requires_source]


def test_the_model_runs_a_command_in_the_machine_of_the_terminal_a_person_picked(
    page_for, make_user, admin, connect_terminals, terminals, upstream
):
    person = make_user()
    tag = uuid.uuid4().hex[:8]
    name = f"Workbench {tag}"
    connection = connect_terminals(read_grant(person.id))
    with admin.client() as client:
        terminal_id = create_template(client, connection, name=name)["terminal_id"]
        share_template(client, terminal_id, read_grant(person.id))
    prompt = f"note the tide {tag}"
    command = f"echo high-{tag} > ~/tide.txt && cat ~/tide.txt"

    page = page_for(person)
    choose_terminal(page, name)
    ask(
        page,
        upstream,
        prompt,
        calling("run_command", {"command": command, "wait": 10}),
        reply.text("Noted."),
    )

    answered = [request for request in upstream.chat_requests() if reply.answering(prompt)(request)]
    results = [entry["content"] for entry in conversation(answered[-1]) if entry["role"] == "tool"]
    assert results, f"run_command never ran; the model was sent {conversation(answered[-1])}"
    printed = "".join(entry["data"] for entry in json.loads(results[-1])["output"])
    assert printed.strip() == f"high-{tag}", f"the model was told the command printed {printed!r}"
    machines = terminals.instances()
    assert [machine["context_id"] for machine in machines] == [f"user:{person.id}"], (
        f"the command did not start the person's own machine: {machines}"
    )
    assert (terminals.home(machines[0]["id"]) / "tide.txt").read_text().strip() == f"high-{tag}"
