# Writing regression tests in this suite

Read this before adding or repairing a test here.

## What this suite is

Regression tests for Open WebUI at three distances from the product. `integration/` drives the HTTP API of a scratch instance the suite boots itself, `e2e/` drives the built frontend of that same instance in a browser, and `unit/` imports backend modules from the checkout. Upstream refactors internals daily; the API and the UI change rarely. So a bug is pinned as far out as it can be seen, where the next refactor leaves the test alone.

## Choosing where a test lives

1. If the symptom shows over HTTP (a status code, a response body, what a later GET returns, what the model provider or an outside service was sent, a server log line), write an integration test.
2. If it also shows in the UI, or only there, add a browser test as well. Duplication across the two is fine.
3. When neither can see it (a pure function with no route to it, event-loop timing, a static guard over the source or packaging), write a unit test by the rules in "Unit tests that survive refactors".

Once an integration twin covers a unit test's narrow layer, delete the unit test: two copies of one guard double the repair work and add no coverage.

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

- **Own your state.** Give anything a test changes to an account from `make_user()`, so the shared `user` stays as the next test expects it. Change a global setting only after `preserve(...)`, which restores it; `preserve(..., on=extra)` does the same on an `instance_with` instance.
- **Script the model, then read what it was sent.** `upstream.queue(reply.text(...), reply.tool_call(name, args), reply.error(500))` decides the replies; `upstream.chat_requests()` shows the payload Open WebUI built. Assert on that, on stored chats (`harness.chat.wait_for_reply`) and on `instance.log_since(offset)`: all three outlive a refactor that renames every internal.
- **Keep outside services local.** Point the setting at a `listener` (or a fake in `harness/`) and assert on what it received, so the instance only ever talks to localhost.
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

- **Locate the way a person does:** by role, label or visible text. Use a CSS class only when nothing else identifies the element, and find a tooltip-only button through its tooltip text.
- **Tie a scripted reply to its prompt** with `reply.text(..., match=reply.answering(prompt))` when a browser test counts or depends on provider requests: a request left over from an earlier test can otherwise take the reply.
- **Wait on what appears.** Playwright's `expect` waits by itself, so a sleep belongs only in a bounded check that nothing more happens (a stopped stream stays stopped).
- **Wait for the page to be ready for what you do.** Keyboard shortcuts bind after the layout loads, so wait for the chat input before pressing them.
- Every failing test leaves a trace per browser in `test-results/`; `playwright show-trace` replays it.

## The three layers

Every fix gets all three; one of them alone is not coverage.

1. **Narrow.** Exactly this bug, exactly this fix. It fails on the ref before the fix and passes on the ref after it, which makes it the only layer that discriminates.
2. **Broad.** The invariant the bug was an instance of. If one endpoint gained an ownership check, assert the property across its siblings so the next instance is caught too.
3. **Nearby.** Adjacent behaviour that is currently correct: the positive path, boundaries, empty and None inputs, the admin-versus-user split. These pass on both refs, which shows the fix did not over-correct.

## Journey tests

A journey test is broad baseline coverage of a feature as a person or a client uses it, pinned to no single issue: an LDAP sign-in, a live note edit, the role every route asks for. It carries `pytest.mark.journey` in place of `regression`. With no fix to undo, it still gets a mutation proof: break the behaviour it guards in a backend copy and name that edit, and what it turned red, in its `Discriminates:` line. `harness/access.py` gives the owner, stranger, reader, writer and admin matrix for a shared resource.

## Proving a test discriminates

An unproven regression test is decoration, so every narrow test is shown to fail with its fix undone. Undo the fix on a scratch copy: a shared checkout or worktree may be in use by someone else.

- **Backend fix:** copy `backend/` to a scratch dir, undo the fix there (the smallest edit that reverses the fix commit), and run with `OPEN_WEBUI_SOURCE_DIR` pointing at the copy. The narrow tests go red there and pass on the clean checkout. For browser tests keep `OPEN_WEBUI_BUILD_DIR` on the clean build.
- **Frontend fix:** copy `src/`, `static/` and the build config files to a scratch dir, symlink the checkout's `node_modules`, undo the fix in `src/`, and build with `npm_package_version=<version> NODE_OPTIONS=--max-old-space-size=6144 npx vite build`. Run the browser test with `OPEN_WEBUI_BUILD_DIR` on that build, plus one unrelated journey test as a control that the build itself works. Unrelated frontend mutations can share one build.
- **Unit tests that stay:** the same, with the unit command above.
- **A test said to guard nothing** gets the same proof: if it stays green with the fix undone, fix the test or delete it.
- **Flakes:** run every new module three times on the clean checkout before committing. A test that fails once in three is a failing test.

## Rules for every test

- **Fail on behaviour.** A wrong value, a missing exception or a call that should not have happened is evidence of the bug; a `TypeError` from a changed signature only shows that the code moved.
- **Bound every test on the pre-fix ref.** A test for a denial-of-service fix meets unbounded behaviour on the ref before the fix, and a test that wedges CI is worse than no test. Bound it by construction, drive it out of process with a hard timeout, or guard it with a capability check that skips on a checkout lacking the fix.
- **When a change breaks a test, retarget it.** Point it at the new shape and keep it pinning the original bug. Loosening an assertion until it passes destroys the only thing the test was for.
- **Skip only for a genuinely absent dependency, and name it.** A blanket `except Exception: pytest.skip(...)` turns real breakage into a green run. When the code a test targets is gone, fail and name what to retarget.
- **A regression stays red until its fix merges.** Write it as a plain failing test that names the issue and the fix PR, so every run shows it. A strict `xfail` is for a known bug nobody is fixing yet: it flips to XPASS on its own once upstream fixes it. Keep allowlists out of the suite, since an allowlist sits there until someone remembers it.

## Unit tests that survive refactors

A unit test breaks whenever upstream renames something the test knows about, so a unit test knows only what its bug requires. It:

- **Calls the most public function that shows the bug.** A route handler called as a Python function, or a private helper that has a public caller, breaks on every signature change.
- **Drives the real code and patches only what it talks to:** the HTTP client, the clock, the filesystem. The database is a real in-memory SQLite with the real tables. When the thing you patched makes the decision, or the test reimplements the logic it checks, the test measures itself and the product goes unchecked.
- **Uses real or specced stand-ins.** A hand-built fake of a Playwright page, an aiohttp response or `request.app.state` breaks the day upstream calls one more method on it; `create_autospec(RealClass)` and the real object keep up.
- **Passes arguments by keyword** and awaits through a helper when upstream may flip a function between sync and async.
- **Audits source by meaning.** Parse with `ast` and assert the property, independent of whitespace, formatting and variable names. An inventory ratchet fails when a new entry appears and passes when upstream removes one.
- **Leaves shared state alone.** It patches what the code reads from the config store or database, because rows written to the shared store survive the session and poison later runs. It reaches backend modules through the `owui_module` fixture: re-executing a module hands the test a second copy while the routers still hold the first, so patches silently miss, and a module that declares ORM tables raises "Table is already defined".

## Style

Comments: one short line at most, only where the why is non-obvious, and about the why. Descriptive names, flat control flow. No em-dashes. No Oxford comma.

Module docstring states what regressed, the fix commit and PR or issue numbers, the mechanism in a sentence or two, `Twin of unit/<path>.` when it has one, and closes with a line reading `Discriminates: passes on <fixed ref>, fails on <buggy ref> (<why>).` (or the mutation that turned it red). Keep that line accurate when you change the test.

`pytestmark` at module level: `pytest.mark.regression` for unit tests, `[regression, api, requires_source]` for integration, `[regression, requires_browser, requires_source]` for browser tests. Async tests use `@pytest.mark.asyncio`. Run `ruff check` and `ruff format` before you finish; the config lives in `pyproject.toml`.

## Reporting

Say what you did, the pass counts on both refs, and which tests discriminate and why. If something could not be tested, say so plainly: a test that only looks like coverage hides the gap from the next reader. A finding you cannot substantiate is worse than no finding, because it makes someone else disprove it.
