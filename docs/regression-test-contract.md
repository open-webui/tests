# Writing regression tests in this suite

Read this before adding or repairing a test here.

## What this suite is

Regression tests for Open WebUI at three distances from the product. `integration/` drives the HTTP API of a scratch instance the suite boots itself, `e2e/` drives the built frontend of that same instance in a browser, and `unit/` imports backend modules from the checkout. Upstream refactors internals daily; the API and the UI change rarely. So a bug is pinned as far out as it can be seen, where the next refactor leaves the test alone.

## Choosing where a test lives

1. If the symptom shows over HTTP (a status code, a response body, what a later GET returns, what the model provider or an outside service was sent, a server log line), write an integration test.
2. If it also shows in the UI, or only there, add a browser test as well. Duplication across the two is fine.
3. **Never write a unit test.** Every new test is an integration or a browser test, with no exceptions. When neither of them can see a bug, the bug gets no new test. The unit tests in `unit/` are left over from before this rule; they only shrink, as integration and browser twins replace them, and none are added or extended.

4. **Extend an existing test first.** Every new test function and every new module costs CI time (setup, accounts, a browser page, often a whole instance), so a new check goes into what already exists, in this order: add the assertion to an existing test that already walks the same path; failing that, add a test function to the existing module of that area so it reuses its fixtures and instance; only when no module fits, write a new module. Extending must keep the existing test's own check intact and its name still true.

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
- **Env-only settings get their own instance** through `instance_with({...})`. Reuse one env set across modules where you can; each boot costs several seconds. Mark the tests that use one `slow`.

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
- **Read a property from the element that shows it.** Take text direction or alignment from the paragraph or list item holding the text, never from the list or block around it: upstream moves `dir` between a container and its items, and the container's value then changes while the reader sees the same thing.
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
- **Flakes:** run every new module five times on the clean checkout before committing (the browser runs throttled fourfold by default) and once more with two runs of it in parallel, as a busy CI runner is slower than your machine. A test that fails once in five is a failing test.

## Rules for every test

- **Fail on behaviour.** A wrong value, a missing exception or a call that should not have happened is evidence of the bug; a `TypeError` from a changed signature only shows that the code moved.
- **Bound every test on the pre-fix ref.** A test for a denial-of-service fix meets unbounded behaviour on the ref before the fix, and a test that wedges CI is worse than no test. Bound it by construction, drive it out of process with a hard timeout, or guard it with a capability check that skips on a checkout lacking the fix.
- **When a change breaks a test, retarget it.** Point it at the new shape and keep it pinning the original bug. Loosening an assertion until it passes destroys the only thing the test was for.
- **Skip only for a genuinely absent dependency, and name it.** A blanket `except Exception: pytest.skip(...)` turns real breakage into a green run. When the code a test targets is gone, fail and name what to retarget.
- **A regression stays red until its fix merges.** Write it as a plain failing test that names the issue and the fix PR, so every run shows it. A strict `xfail` is for a known bug nobody is fixing yet: it flips to XPASS on its own once upstream fixes it. Keep allowlists out of the suite, since an allowlist sits there until someone remembers it.

## Writing tests that do not flake

A test that fails without a code change costs a release its trust in the suite. Every rule here comes from a test that did exactly that.

- **Wait for the server, not the screen.** The UI shows a message, a reply or an edit before the server has stored it. Before the next step depends on something being saved (deleting a message that a reply quotes, approving a call, reading it as another account), wait for the request's response or read it back over HTTP.
- **Act only on a settled page.** After opening the sidebar, a menu or a dialog, or picking from a suggestion list, wait for the state that proves it took effect (the dialog visible, the picked mention in the input) before the next click or keystroke. A click that can be lost while something animates may be repeated until its effect shows, with a one-line comment on why; never repeat an assertion.
- **Check only what the test is about.** When a step only needs an earlier result (a reply before opening its sources), and the live view can lag behind the server, read the result from the stored chat. Keep live-rendering assertions in the tests that are about live rendering.
- **Own every piece of state the test relies on.** Create the account, skill, channel or setting the test needs in the test itself, so it passes alone and in any order. A test that only passes because an earlier one left something behind is broken.
- **Derive expectations from data you do not control.** When a test reads data that changes between releases (the changelog, the default prompts, the model list), check each item against its own source instead of assuming some item somewhere has a property.
- **Tie every scripted reply to its prompt** with `reply.answering(prompt)`, and every count of provider requests to the requests of this test, so a request left over from an earlier test cannot take or add a reply.
- **Pin time.** Compute expected dates in the zone the page uses, keep clear of midnight, and assert relative wording ("a few seconds ago") only where the page promises it.
- **Keep reminders out of a test that does not test them.** A calendar event a test makes on a fixed day and hour can start within minutes of the run, and its reminder toast then sits over the page header. Create it with `NO_REMINDER` from `harness/calendar_api.py`, or pick None as its Reminder in the editor.
- **Compare listeners only after the last event.** Before comparing what two tabs or sockets received, wait on both for the final event the server sends (a chat task ends with `chat:active` false, after its `done`), since a tab snapshotted at an earlier event may still be missing the rest.
- **Assert the settled state, never a passing one.** Wait for what stays on screen once the action has finished (the stored file's thumbnail), not what shows only while it runs (the file name during the upload), which a fast server can skip before the first check.
- **Never depend on timing a real provider cannot produce.** The scripted provider is faster than any real one, so it waits `REPLY_START_DELAY` (0.09s, scaled in slow mode when the run includes browser tests) before each reply. Without it the first reply of a new chat could finish before the page had the chat's id, and the page dropped it. Keep that delay as it is and wait on events; never change it to make a test pass.
- **Wait for the reply itself, never for its request.** The provider has the request well before the first piece reaches Open WebUI, so a stop sent a fixed time after the request can land before anything arrived. When a step needs part of the reply in hand, wait for a piece on the account's socket (`wait_until_streaming`), and after a stop wait for the stored reply to be marked done (`wait_for_reply`) before reading what was kept. The task ending is not enough: with Redis the stop returns before the instance running the reply has saved it.
- **Hold a reply the test must outlast; never slow it down.** When a reply has to finish only after a step (leaving the chat before it ends), queue it with `hold=` an event and set the event once the step shows on screen. A slow stream only bets that the step is quicker, and the wait for the reply then grows with it.
- **Type into a new chat only in a page that left no draft there.** The new chat page restores the tab's unsent draft once it has loaded, over whatever was typed in the meantime. A second check that types into a fresh chat opens it in a page of its own.
- **A page that refreshes on a timer is not a wait.** When the page reads a list once, a set time after an action, a slower run misses it. Wait for the server to hold the result, then reload the page.
- **Let any provider answer at once.** The page drops the live events of a chat whose id it has not read yet, and a pipe or a local stand-in can finish a new chat's reply before a slow page reads the id from the send's answer. Every page `page_for` opens therefore holds a new chat's socket events until it joins that chat's room, so a pipe or a stand-in needs no delay of its own. Open chat pages through `page_for`.
- **Switch the hold off only for a bug in that race.** A test that guards a bug the hold would hide (a new chat's first reply or error lost because it came before the page knew the chat) carries `pytest.mark.real_socket_timing`, and its `page_for` pages get the socket events as they arrive. Never use it for anything else.
- **Show that something did not wait with a dependency that never answers.** A wall-clock bound on a page load measures the page's speed as well, so it fails on a slow runner. Hold the dependency until the test ends and check that the page got there without it.

- **Never assert an order the database does not guarantee.** A list read without a defined sort (search hits across files) comes back in whatever order the database hands out, and Postgres changes it between runs where SQLite does not. Compare such a list sorted or as a set, and assert an order only where Open WebUI defines one (a documented sort flag, newest first).
- **Prove the list loaded before checking something is missing.** `to_have_count(0)` passes at once on a list still empty because it is loading. Wait first for an item the same account surely sees (an everyone-read note, its own prompt), then check the absence.
- **Set the starting state; never read it.** Put a setting into the state the test starts from (over the API, after `preserve`) instead of relying on the instance default or branching on what a switch currently shows.
- **Look for text inside the region that holds it.** A page or dialog can show the same text twice (a search result and its preview, a title and its breadcrumb), and a locator over the whole dialog then matches both. Scope it to the list, row or panel the test is about.
- **Wait for a filtered list to match its filter.** A search or filter shows the unfiltered items first, so the match is visible before the query has run. Assert the narrowed count first, then what the remaining items hold.
- **Drive a shared widget through one helper.** When tests in several modules use the same dialog or menu (the search dialog, the model selector), keep its locators and waits in one `utils/` module, so a redesign is repaired in one place instead of drifting apart in each copy.
- **Let the page finish what it does on load before undoing it.** The chat page focuses the message box shortly after it shows, so a blur right after the box appears can come first and the focus lands afterwards. Wait for the load-time state (the box focused) and then change it.
- **Open a toggle by its own state, never by whether its content shows.** Content inside an open group can vanish for a moment while it re-renders, and clicking the toggle then closes it. Click only a toggle that reads closed (`expanded=False`).
- **Check every setup call.** Assert the status of each request that builds the test's state, so a failed setup shows the server's answer instead of a misleading error later.

## Checklist before committing a test

Every test in the suite meets all of these. Check each one against your test before you commit.

1. **It pins one behaviour a person can see.** The name is a sentence about what a user, admin or client sees, and the test fails when that behaviour breaks and for no other reason.
2. **It is proven.** A narrow test is shown red with its fix undone, a journey test with a named mutation; the `Discriminates:` line says which.
3. **It asserts on the outside.** HTTP responses, what is stored and read back, what the provider or a local service received, the server log, and in the browser what a person sees, found by role, label or visible text. Never on internal function names, private attributes or CSS classes when anything else identifies the element.
4. **It cannot pass vacuously.** A loop or `all(...)` over a list first asserts the list is not empty; a check that something did not happen is paired with one showing the same setup does make it happen.
5. **Its failure message says what went wrong for a person**, with the actual value in it, so a red run can be read without opening the test.
6. **It owns its state.** Fresh accounts from `make_user()`, unique names (a uuid tag) for everything it creates, global settings changed only after `preserve(...)`. It passes alone, in any order, repeated and next to a parallel run of itself.
7. **It is deterministic.** The scripted provider and local listeners stand in for every outside service; no internet, no real API keys, no randomness without a fixed seed, no dependence on the wall clock beyond what is pinned.
8. **It waits on events, never on time.** Every wait is for a state (an element, a stored value, a request seen) with a bound; a fixed sleep appears only to show that nothing more happens, with a one-line comment. A wait gets a longer limit than its default only when the work it waits on is legitimately slow and nothing races it (a cold app load, Python running in the browser), only after a maintainer approved it, and then scaled with `slow_machine.scaled` and a one-line comment why. `page_for` hands over a page only once the app has finished loading, within the navigation limit, so a test never adds a wait of its own for the load. A test that starts on another page opens it directly with `page_for(actor, path="/admin/users")`, which gets the same wait, and never loads the chat page first only to navigate away.
9. **It is bounded.** It cannot hang CI on any ref, including the one before the fix.
10. **It hides nothing.** No blanket `except`, no skip except for a named absent dependency, no `xfail`, no retry around an assertion.
11. **It runs on every backend** (SQLite, Postgres, Redis) or skips on the one it cannot run on and says why.
12. **It leaves nothing running.** Processes, servers and temporary files it starts are stopped and removed in a `finally` or a fixture.
13. **It extends before it adds.** It lives in the existing test or module that already walks its path, unless none fits.
14. **It reads plainly.** Descriptive names, flat control flow, helpers named for what they do, comments only for a non-obvious why, ruff clean.
15. **It passes five runs in a row on a slow machine.** A new or changed browser test passes `for i in 1 2 3 4 5; do pytest <module> || break; done` with the default slow mode on. It throttles the browser's CPU fourfold, so a race a fast machine wins fails here instead of on CI. Never switch it off with `OWUI_TEST_SLOW=0` to get a test green.

## Keeping the remaining unit tests alive

Nothing here is a reason to write a new unit test. It applies only to repairing the ones that are left until a twin replaces them. A unit test breaks whenever upstream renames something the test knows about, so a unit test knows only what its bug requires. It:

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
