"""Opt-in line coverage of the backend, switched on with `OWUI_TEST_COVERAGE=1`.

Every scratch instance starts coverage before it imports the backend (through
`COVERAGE_PROCESS_START`, which the instances inherit), and the pytest process measures itself
for the tests that import the backend directly. At the end of the session the data files are
combined into an HTML and a JSON report under `OWUI_TEST_COVERAGE_DIR` (`coverage-report/`).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

ENABLED = os.getenv("OWUI_TEST_COVERAGE", "").lower() in ("1", "true", "yes")
REPORT_DIR = Path(os.getenv("OWUI_TEST_COVERAGE_DIR", "coverage-report")).resolve()
RCFILE = REPORT_DIR / "coveragerc"

_in_process = None


def start() -> None:
    """Write the config, point the instances at it and measure this process too."""
    global _in_process
    if not ENABLED:
        return
    import coverage

    from harness.instance import resolve_backend

    backend = resolve_backend()
    # by path, so modules the unit tests load from a file under another name count too
    source = f"source = {backend / 'open_webui'}" if backend else "source_pkgs = open_webui"
    shutil.rmtree(REPORT_DIR / "data", ignore_errors=True)
    (REPORT_DIR / "data").mkdir(parents=True)
    RCFILE.write_text(
        "[run]\n"
        f"{source}\n"
        "parallel = true\n"
        "sigterm = true\n"
        # SQLAlchemy async switches greenlets under every await on the database
        "concurrency = thread, greenlet\n"
        "disable_warnings = no-data-collected, module-not-imported\n"
        f"data_file = {REPORT_DIR / 'data' / '.coverage'}\n",
        encoding="utf-8",
    )
    os.environ["COVERAGE_PROCESS_START"] = str(RCFILE)
    _in_process = coverage.Coverage(config_file=str(RCFILE))
    _in_process.start()


def report() -> float | None:
    """Combine every process's data into the HTML and JSON reports; returns the total percent."""
    if not ENABLED:
        return None
    import coverage

    if _in_process is not None:
        _in_process.stop()
        _in_process.save()
    combined = coverage.Coverage(config_file=str(RCFILE))
    combined.combine()
    combined.save()
    combined.json_report(outfile=str(REPORT_DIR / "coverage.json"))
    return combined.html_report(directory=str(REPORT_DIR / "html"))
