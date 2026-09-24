# Writing regression tests in this suite

Read this before adding or repairing a test here.

## What this suite is

Regression tests for Open WebUI at three distances from the product. `integration/` drives the HTTP API of a scratch instance the suite boots itself, `e2e/` drives the built frontend of that same instance in a browser, and `unit/` imports backend modules from the checkout. Upstream refactors internals daily; the API and the UI change rarely. So a bug is pinned as far out as it can be seen.

## Choosing where a test lives

1. If the symptom shows over HTTP (a status code, a response body, what a later GET returns, what the model provider or an outside service was sent, a server log line), write an integration test.
2. If it also shows in the UI, or only there, add a browser test as well. Duplication across the two is fine.
3. Only when neither can see it (a pure function with no route to it, event-loop timing, a static guard over the source or packaging) does it stay a unit test, written by the rules below.

A unit test whose integration twin covers its narrow layer is deleted, not kept alongside.

## The harness

`harness/` boots a scratch backend per session from `OPEN_WEBUI_SOURCE_DIR`, serving the checkout's built frontend when there is one, with a scripted OpenAI-shaped model provider as its only connection. The fixtures in `harness/fixtures.py` are available everywhere: `instance`, `upstream` (script replies with `upstream.queue(...)`, read what was sent with `upstream.chat_requests()`), `admin`, `user`, `make_user`, `preserve` (restores global settings a test changes), `instance_with` (a further instance for env-only settings) and `listener` (a local service for the instance to call). `harness.chat` sends messages the way the web client does. The browser suite adds `page_for(actor)`, one signed-in browser per account.

```
OPEN_WEBUI_SOURCE_DIR=/path/to/checkout/backend python -m pytest integration e2e -q
```

The browser tests skip unless the checkout has a built frontend (`npm run build`, or point `OPEN_WEBUI_BUILD_DIR` at one). Title, tag, follow-up and query generation are off on the shared instance because they would take a scripted reply meant for the chat.

The unit tests import the checkout directly. Always set `DATA_DIR` and `STATIC_DIR` to scratch paths as well: importing `open_webui.config` creates `DATA_DIR` and deletes tracked files under `STATIC_DIR`, so an unset pair mutates the tree under test.

```
OPEN_WEBUI_SOURCE_DIR=/path/to/checkout/backend DATA_DIR=/tmp/scratch/data STATIC_DIR=/tmp/scratch/static WEBUI_SECRET_KEY=test   python -m pytest unit -q
```

## Writing an integration test

A twin of a unit test lives at the same path under `integration/` (`unit/security/test_x.py` becomes `integration/security/test_x.py`). Read the module docstrings in `harness/` first: fakes for OIDC, terminal servers, image engines, tool servers, a second provider and socket clients already exist.

```python
"""Regression: a user with read access saw the tool's source in /tools/list.

Fix c05de13b4 (#NNNNN) strips `content` for callers without write access. Twin of
unit/security/test_tool_source_exposure.py.

Discriminates: passes on dev bbfa876af, fails with c05de13b4 reverted (the list carries the source).
"""

pytestmark = [pytest.mark.regression, pytest.mark.api, pytest.mark.requires_source]


def test_a_read_grant_does_not_expose_the_source(admin, make_user):
    reader = make_user()
    tool_id = _create_tool(admin, source="SECRET = 'sk-live'")
    _grant(admin, tool_id, reader, "read")
    with reader.client() as client:
        listed = client.get("/api/v1/tools/list").json()
    assert "sk-live" not in json.dumps(listed)
```

`_create_tool` and `_grant` are a few lines each in the module, calling `/api/v1/tools/create` and `/api/v1/tools/id/{id}/access/update` as the admin panel does. See `integration/security/test_tool_source_exposure.py` for the full twin.

- **Own your state.** Anything a test changes belongs to an account from `make_user()`, never to the shared `user`. Global settings are changed only after `preserve(...)`, which restores them; `preserve(..., on=extra)` does the same on an `instance_with` instance.
- **Script the model, then read what it was sent.** `upstream.queue(reply.text(...), reply.tool_call(name, args), reply.error(500))` decides the replies; `upstream.chat_requests()` shows the payload Open WebUI built. Assert on that, on stored chats (`harness.chat.wait_for_reply`) and on `instance.log_since(offset)`, never on internals.
- **Outside services are local.** Point the setting at a `listener` (or a fake in `harness/`) and assert on what it received. A test never makes the instance reach past localhost.
- **Env-only settings get their own instance** through `instance_with({...})`. Reuse one env set across modules where you can; each boot costs about 13 seconds. Mark the tests that use one `slow`.

## Writing a browser test

Browser twins live under `e2e/<area>/`. `page_for(actor)` opens a signed-in page in a browser of its own, so two accounts can act in one test; `utils/chat_ui.py` sends messages and reads replies.

```python
def test_a_provider_error_is_shown_and_the_next_message_still_sends(page_for, make_user, upstream):
    page = page_for(make_user())
    upstream.queue(reply.error(500, "the provider is down"), reply.text("back again"))
    send(page, "hello?")
    expect_reply(page, "the provider is down")
    send(page, "hello again")
    expect_reply(page, "back again")
```

- **Locate the way a person does:** by role, label or visible text. A CSS class is the last resort, and a tooltip-only button is found through its tooltip text.
- **Wait on what appears, not on time.** Playwright's `expect` waits by itself. The only sleep allowed is a bounded check that nothing more happens (a stopped stream stays stopped).
- **Wait for the page to be ready for what you do.** Keyboard shortcuts bind after the layout loads; wait for the chat input, not for the first link.
- Every failing test leaves a trace per browser in `test-results/`; `playwright show-trace` replays it.

## Proving a test discriminates

Every narrow test is shown to fail with its fix undone, on a copy, never on a shared checkout.

- **Backend fix:** copy `backend/` to a scratch dir, undo the fix there (the smallest edit that reverses the fix commit), and run with `OPEN_WEBUI_SOURCE_DIR` pointing at the copy. The narrow tests must go red; the same tests on the clean checkout must pass. For browser tests keep `OPEN_WEBUI_BUILD_DIR` on the clean build.
- **Frontend fix:** copy `src/`, `static/` and the build config files to a scratch dir, symlink the checkout's `node_modules`, undo the fix in `src/`, and build with `npm_package_version=<version> NODE_OPTIONS=--max-old-space-size=6144 npx vite build`. Run the browser test with `OPEN_WEBUI_BUILD_DIR` on that build, plus one unrelated journey test as a control that the build itself works. Unrelated frontend mutations can share one build.
- **Unit tests that stay:** the same, with the unit command above.
- **Flakes:** run every new module three times on the clean checkout before committing. A test that fails once in three is a failing test.

## The three layers

Every fix gets all three. One of them alone is not coverage.

1. **Narrow.** Exactly this bug, exactly this fix. Must FAIL on the ref before the fix and PASS on the ref after it.
2. **Broad.** The invariant the bug was an instance of. If one endpoint gained an ownership check, assert the property across its siblings so the next instance is caught too.
3. **Nearby.** Adjacent behaviour that is currently correct: the positive path, boundaries, empty and None inputs, the admin-versus-user split. These SHOULD pass on both refs. They prove the fix did not over-correct.

Layer 3 passing on the old ref is correct and expected. Only layer 1 discriminates.

## Rules that exist because they were violated

- **Prove discrimination, do not assume it.** Run the narrow tests against the pre-fix ref and confirm they fail. An unproven regression test is decoration.
- **Prefer a behavioural failure** (wrong value, missing exception, a call that should not have happened) over a `TypeError` from a changed signature. A signature-only failure is weak evidence dressed as a guard.
- **Never write a test that hangs, crashes or exhausts memory on the pre-fix ref.** Denial-of-service fixes are guarded by tests whose pre-fix behaviour is, by definition, unbounded. Bound them by construction, or drive them out of process with a hard timeout, or guard them with a capability check that skips on a checkout lacking the fix. A test that wedges CI is worse than no test.
- **Never loosen an assertion to make a test pass.** When a rename breaks a test, retarget it at the new shape and keep it pinning the original bug. Softening until green destroys the only thing the test was for.
- **Mock only the I/O boundary.** Drive the real production function. Do not reimplement the logic in the test and then assert against your reimplementation.
- **Do not assert on your own mock.** If the thing you patched is the thing that makes the decision, the test measures the mock and not the code.
- **A unit test does not write to a real config store or database.** Patch what the code reads instead. Rows written to the shared store survive the session and poison later runs. (Integration tests write through the API to their scratch instance, inside `preserve`.)
- **Never evict a module from `sys.modules`.** Re-executing a backend module hands the test a second module object while the routers still hold the first, so patches silently miss, and re-executing one that declares ORM tables raises "Table is already defined". Use the `owui_module` fixture.
- **Skip narrowly or not at all.** A blanket `except Exception: pytest.skip(...)` turns real breakage into a green run. Skip only for a genuinely absent target, and name the reason.
- **A known-unfixed upstream bug is an `xfail`, not an allowlist.** An `xfail` flips to XPASS on its own when upstream fixes it. An allowlist sits there forever until someone remembers.
- **When a test is claimed to guard nothing, prove the fix by mutation.** Copy the backend file to a scratch dir, break the fix, point `OPEN_WEBUI_SOURCE_DIR` at the copy, and confirm the test goes red. Never mutate a shared worktree.

## Unit tests that survive refactors

Most repairs this suite ever needed came from a unit test knowing more about the code than the bug required. A unit test that stays:

- **Calls the most public function that shows the bug.** Not a route handler called as a Python function, not a private helper when a public caller exists.
- **Mocks what the code talks to, not what it is made of.** Patch the HTTP client, the clock, the filesystem; use a real in-memory SQLite with the real tables for the database. Never patch a model method by name to steer the code under test.
- **Uses real or specced stand-ins.** A hand-built fake of a Playwright page, an aiohttp response or `request.app.state` breaks the day upstream calls one more method on it. `create_autospec(RealClass)` or the real object does not.
- **Passes arguments by keyword** and awaits through a helper when upstream may flip a function between sync and async.
- **Audits source by meaning.** Parse with `ast` and assert the property; never match whitespace, formatting or variable names. An inventory ratchet fails when a new entry appears and passes when upstream removes one.
- **Fails loudly when its target is gone**, naming what to retarget. Never skip on a missing attribute.

## Style

Terse comments: at most one short line, and only where the WHY is non-obvious. Never narrate the mechanism. Descriptive names, flat control flow. No em-dashes. No Oxford comma.

Module docstring states what regressed, the fix commit and PR or issue numbers, the mechanism in a sentence or two, `Twin of unit/<path>.` when it has one, and closes with a line reading `Discriminates: passes on <fixed ref>, fails on <buggy ref> (<why>).` (or the mutation that turned it red). Keep that line accurate when you change the test.

`pytestmark` at module level: `pytest.mark.regression` for unit tests, `[regression, api, requires_source]` for integration, `[regression, requires_browser, requires_source]` for browser tests. Async tests use `@pytest.mark.asyncio`. Run `ruff check` and `ruff format` before you finish; the config lives in `pyproject.toml`.

## Reporting

Say what you did, the pass counts on both refs, and which tests discriminate and why. If something could not be tested, say so plainly rather than writing a test that only looks like coverage. A finding you cannot substantiate is worse than no finding, because it makes someone else disprove it.
