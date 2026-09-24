"""Regression: the code interpreter's module blocklist and its chart patch, in the browser.

Two 0.11.0 fixes to Python that Open WebUI generates for the Pyodide sandbox, the default
engine, which runs in the user's browser:

`dc4b82885` (#27245): the preamble prepended to interpreter code when
`CODE_INTERPRETER_BLOCKED_MODULES` is set defined its import hook as `async def`, so blocked
modules were not blocked and every other import bound a coroutine. `utils/middleware.py` (the
`<code_interpreter>` tag) and `tools/builtin.py` (the `execute_code` tool) each carry a copy.

`b940cd529` (#26800, issue #26660): `pyodideSandboxHost.ts`, the sandbox used while
ENABLE_PYODIDE_FILE_PERSISTENCE is off, wrote the indented lines of its matplotlib `show()`
override with doubled backslashes inside `String.raw`, so the override did not compile and no
chart could be drawn.

The tool's result is read in the chat. The tag's output is read from what the model is sent
back, because the chat shows a tag's code but not its output on dev bbfa876af.

Twin of unit/security/test_code_interpreter_module_blocking.py.

Discriminates: passes on dev bbfa876af; the hook made `async def` again in `tools/builtin.py`
fails the tool tests (the chart ones because the sandbox never answers) and in
`utils/middleware.py` the tag test; the doubled backslashes restored fail both chart tests on
the default sandbox and pass on the persistent worker.
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import expect

from harness import upstream as reply
from harness.actors import create_user
from utils.chat_ui import expect_reply, last_reply, send

pytestmark = [
    pytest.mark.regression,
    pytest.mark.requires_browser,
    pytest.mark.requires_source,
    pytest.mark.slow,
]

BLOCKLIST = {"CODE_INTERPRETER_BLOCKED_MODULES": "socket"}
SANDBOXES = {
    "default-sandbox": BLOCKLIST,
    "persistent-worker": {**BLOCKLIST, "ENABLE_PYODIDE_FILE_PERSISTENCE": "true"},
}
TAG_FORMAT = '<code_interpreter type="code" lang="python">'
IMPORTS_CELL = "\n".join(
    [
        "try:",
        "    import socket",
        "    print('socket => imported')",
        "except ImportError as error:",
        "    print('socket =>', error)",
        "try:",
        "    import json",
        "    print('json =>', json.dumps([1, 2]))",
        "except Exception as error:",
        "    print('json =>', type(error).__name__)",
    ]
)
# Guarded so a broken import reports back; an uncaught error there can wedge the sandbox.
CHART_CELL = "\n".join(
    [
        "try:",
        "    import matplotlib.pyplot as plt",
        "    plt.plot([1, 2, 3])",
        "    plt.show()",
        "except Exception as error:",
        "    print('chart =>', type(error).__name__)",
    ]
)
SOCKET_REFUSED = "socket => Direct import of module socket is restricted"
PYODIDE_TIMEOUT_MS = 90_000


@pytest.fixture
def page_on(browser):
    """`page_on(instance)` signs a new account in to that instance in a browser of its own."""
    contexts = []

    def open_page(launched, path="/?code-interpreter=true"):
        if not launched.serves_frontend:
            pytest.skip("no built frontend (set OPEN_WEBUI_BUILD_DIR)")
        account = create_user(launched)
        with account.client() as client:
            settings = {"ui": {"showChangelog": False}}
            client.post("/api/v1/users/user/settings/update", json=settings).raise_for_status()
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080}, base_url=launched.base_url
        )
        context.set_default_timeout(30_000)
        contexts.append(context)
        page = context.new_page()
        page.add_init_script(
            f"try {{ localStorage.setItem('token', {json.dumps(account.token)}); }} catch (e) {{}}"
        )
        page.goto(path)
        return page, account

    yield open_page
    for context in contexts:
        context.close()


def _tool_result(page):
    """The expanded result of the reply's execute_code call."""
    reply_box = last_reply(page)
    reply_box.get_by_role("button", name=re.compile("execute_code")).click()
    return reply_box


# Narrow


def test_the_execute_code_tool_refuses_a_blocked_import(page_on, instance_with):
    blocking = instance_with(BLOCKLIST)
    page, _ = page_on(blocking)
    blocking.upstream.queue(
        reply.tool_call("execute_code", {"code": IMPORTS_CELL}), reply.text("ran it")
    )
    send(page, "run it")
    expect_reply(page, "ran it")

    result = _tool_result(page)
    expect(result).to_contain_text(SOCKET_REFUSED)
    expect(result).to_contain_text("json => [1, 2]")


def test_a_code_interpreter_tag_refuses_a_blocked_import(page_on, instance_with):
    blocking = instance_with(BLOCKLIST)
    page, account = page_on(blocking)
    with account.client() as client:
        legacy = {"ui": {"params": {"function_calling": "legacy"}}}
        client.post("/api/v1/users/user/settings/update", json=legacy).raise_for_status()
    page.reload()
    blocking.upstream.queue(
        reply.text(f"{TAG_FORMAT}\n{IMPORTS_CELL}\n</code_interpreter>"), reply.text("ran it")
    )
    send(page, "run it")
    expect_reply(page, "ran it")

    expect(last_reply(page).get_by_text("Analyzed")).to_be_visible()
    sent_back = blocking.upstream.chat_requests()[-1]["messages"][-1]["content"]
    assert SOCKET_REFUSED in sent_back, (
        f"the browser sandbox imported a blocked module from a code interpreter tag (#27245): "
        f"{sent_back}"
    )
    assert "json => [1, 2]" in sent_back, (
        f"`import json` gave no usable module in the browser sandbox (#27245): {sent_back}"
    )


def test_a_chart_in_a_code_block_runs_and_renders(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.text(f"Here is a chart:\n\n```python\n{CHART_CELL}\n```\n"))
    send(page, "draw a line")
    expect_reply(page, "Here is a chart:")

    last_reply(page).get_by_role("button", name="Run").click()
    chart = last_reply(page).get_by_role("img", name="Output")
    refused = last_reply(page).get_by_text("SyntaxError")
    expect(chart.or_(refused)).to_be_visible(timeout=PYODIDE_TIMEOUT_MS)
    expect(refused).to_have_count(0)
    expect(chart).to_be_visible()


# Broad


@pytest.mark.parametrize("sandbox", SANDBOXES)
def test_a_chart_from_the_execute_code_tool_renders_with_a_blocklist(
    sandbox, page_on, instance_with
):
    launched = instance_with(SANDBOXES[sandbox])
    page, _ = page_on(launched)
    launched.upstream.queue(
        reply.tool_call("execute_code", {"code": CHART_CELL}), reply.text("drew it")
    )
    send(page, "draw a line")
    expect_reply(page, "drew it")

    result = _tool_result(page)
    expect(result).to_contain_text("Output Image")
    expect(result).not_to_contain_text("SyntaxError")
