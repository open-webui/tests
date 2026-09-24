"""Render a pytest JUnit report as Markdown for a GitHub job summary.

python scripts/junit_summary.py e2e-junit.xml "Browser suite" >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path


def summarise(report: Path, title: str) -> str:
    if not report.is_file():
        return f"### {title}\n\nNo report was written; the suite did not get as far as running.\n"
    cases = list(ElementTree.parse(report).getroot().iter("testcase"))
    outcomes = {"passed": 0, "failed": 0, "skipped": 0}
    failures: list[str] = []
    for case in cases:
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            outcomes["failed"] += 1
            message = problem.get("message") or ""
            first_line = message.splitlines()[0][:200] if message else ""
            failures.append(f"- `{case.get('classname')}::{case.get('name')}`: {first_line}")
        elif case.find("skipped") is not None:
            outcomes["skipped"] += 1
        else:
            outcomes["passed"] += 1
    counts = ", ".join(f"{count} {outcome}" for outcome, count in outcomes.items())
    lines = [f"### {title}", "", counts, ""]
    if failures:
        lines += ["Failed:", *failures, ""]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarise(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "Test results"))
