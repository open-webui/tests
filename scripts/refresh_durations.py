"""Refresh a pytest-split durations file from the JUnit reports of a CI run.

python scripts/refresh_durations.py durations/e2e.json reports/e2e-report-*/junit.xml

Every passing test a report measured gets the mean of that time and its previous one (setup, call
and teardown, so a module's instance boot counts against its first test). A failed, errored or
skipped test is left out, since a red run times the failure, not the test, so it keeps its old
entry; tests no report ran keep theirs too. Shards split on these times, so times measured on the
CI runners balance the shards the CI runners run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parent.parent


def node_id(classname: str, name: str) -> str | None:
    """`integration.x.test_y` and `test_z[a]` as `integration/x/test_y.py::test_z[a]`."""
    parts = classname.split(".")
    for end in range(len(parts), 0, -1):
        module = Path(*parts[:end]).with_suffix(".py")
        if (ROOT / module).is_file():
            return "::".join([module.as_posix(), *parts[end:], name])
    return None


def measured(reports: list[Path]) -> dict[str, float]:
    times: dict[str, float] = {}
    for report in reports:
        for case in ElementTree.parse(report).getroot().iter("testcase"):
            if any(case.find(outcome) is not None for outcome in ("failure", "error", "skipped")):
                continue
            test = node_id(case.get("classname", ""), case.get("name", ""))
            if test:
                times[test] = round(float(case.get("time", 0)), 3)
    return times


def main() -> None:
    target, *reports = (Path(argument) for argument in sys.argv[1:])
    durations = json.loads(target.read_text()) if target.is_file() else {}
    times = measured(reports)
    # the same module runs up to twice as long from one run to the next; averaging with the
    # previous time keeps one slow run from reshuffling every shard
    for test, seconds in times.items():
        previous = durations.get(test)
        durations[test] = round((previous + seconds) / 2, 3) if previous else seconds
    target.write_text(json.dumps(durations, indent=4, sort_keys=True) + "\n")
    print(f"{len(times)} measured times from {len(reports)} reports written to {target}")


if __name__ == "__main__":
    main()
