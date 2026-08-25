# Writing regression tests in this suite

Read this before adding or repairing a test here.

## What this suite is

Source-level tests that import the Open WebUI backend directly from a local checkout. They do
not talk to a running instance. `unit/` is the substantial part and is expected to be green.
`integration/` and `e2e/` need a live Open WebUI at `localhost:8080` with seeded accounts, and
are skipped or red without one.

Point the suite at a checkout with `OPEN_WEBUI_SOURCE_DIR`. Always set `DATA_DIR` and
`STATIC_DIR` to scratch paths as well: importing `open_webui.config` creates `DATA_DIR` and
deletes tracked files under `STATIC_DIR`, so an unset pair mutates the tree under test.

```
OPEN_WEBUI_SOURCE_DIR=/path/to/checkout/backend \
DATA_DIR=/tmp/scratch/data STATIC_DIR=/tmp/scratch/static WEBUI_SECRET_KEY=test \
  python -m pytest unit -q
```

## The three layers

Every fix gets all three. One of them alone is not coverage.

1. **Narrow.** Exactly this bug, exactly this fix. Must FAIL on the ref before the fix and PASS
   on the ref after it.
2. **Broad.** The invariant the bug was an instance of. If one endpoint gained an ownership
   check, assert the property across its siblings so the next instance is caught too.
3. **Nearby.** Adjacent behaviour that is currently correct: the positive path, boundaries,
   empty and None inputs, the admin-versus-user split. These SHOULD pass on both refs. They
   prove the fix did not over-correct.

Layer 3 passing on the old ref is correct and expected. Only layer 1 discriminates.

## Rules that exist because they were violated

- **Prove discrimination, do not assume it.** Run the narrow tests against the pre-fix ref and
  confirm they fail. An unproven regression test is decoration.
- **Prefer a behavioural failure** (wrong value, missing exception, a call that should not have
  happened) over a `TypeError` from a changed signature. A signature-only failure is weak
  evidence dressed as a guard.
- **Never write a test that hangs, crashes or exhausts memory on the pre-fix ref.** Denial-of-
  service fixes are guarded by tests whose pre-fix behaviour is, by definition, unbounded. Bound
  them by construction, or drive them out of process with a hard timeout, or guard them with a
  capability check that skips on a checkout lacking the fix. A test that wedges CI is worse than
  no test.
- **Never loosen an assertion to make a test pass.** When a rename breaks a test, retarget it at
  the new shape and keep it pinning the original bug. Softening until green destroys the only
  thing the test was for.
- **Mock only the I/O boundary.** Drive the real production function. Do not reimplement the
  logic in the test and then assert against your reimplementation.
- **Do not assert on your own mock.** If the thing you patched is the thing that makes the
  decision, the test measures the mock and not the code.
- **Do not write to a real config store or database.** Patch what the code reads instead. Rows
  written to the shared store survive the session and poison later runs.
- **Never evict a module from `sys.modules`.** Re-executing a backend module hands the test a
  second module object while the routers still hold the first, so patches silently miss, and
  re-executing one that declares ORM tables raises "Table is already defined". Use the
  `owui_module` fixture.
- **Skip narrowly or not at all.** A blanket `except Exception: pytest.skip(...)` turns real
  breakage into a green run. Skip only for a genuinely absent target, and name the reason.
- **A known-unfixed upstream bug is an `xfail`, not an allowlist.** An `xfail` flips to XPASS on
  its own when upstream fixes it. An allowlist sits there forever until someone remembers.

## Style

Terse comments: at most one short line, and only where the WHY is non-obvious. Never narrate the
mechanism. Descriptive names, flat control flow. No em-dashes. No Oxford comma.

Module docstring states what regressed, the fix commit and PR or issue numbers, the mechanism in
a sentence or two, and closes with a line reading
`Discriminates: passes on <fixed ref>, fails on <buggy ref> (<why>).`
Keep that line accurate when you change the test.

`pytestmark = pytest.mark.regression` at module level. Async tests use `@pytest.mark.asyncio`.
Run `ruff check` before you finish; the config lives in `pyproject.toml`.

## Reporting

Say what you did, the pass counts on both refs, and which tests discriminate and why. If
something could not be tested, say so plainly rather than writing a test that only looks like
coverage. A finding you cannot substantiate is worse than no finding, because it makes someone
else disprove it.
