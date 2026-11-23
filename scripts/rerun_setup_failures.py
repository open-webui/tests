"""Rerun the failed jobs of a CI run once when one of them failed before its tests ran.

python scripts/rerun_setup_failures.py 45153867826   # after the run's first attempt
python scripts/rerun_setup_failures.py --guard       # first step of a job in the rerun

A job's tests ran once a step named "Run the ..." started. A job that failed or timed out before
that hit the runner or an outside service, so the failed jobs get one rerun. GitHub reruns every
failed job or a single one, so a job whose tests ran and failed comes along; the guard fails it
again straight away, which leaves its first result standing. Needs `gh` with a token.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

TEST_STEP = "Run the "


def jobs(run_id: str, attempt: int) -> list[dict]:
    run = f"repos/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{run_id}"
    url = f"{run}/attempts/{attempt}/jobs?per_page=50"
    command = ["gh", "api", "--paginate", url, "--jq", ".jobs[]"]
    pages = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in pages.splitlines()]


def tests_ran(job: dict) -> bool:
    return any(
        step["name"].startswith(TEST_STEP) and step["conclusion"] not in (None, "skipped")
        for step in job["steps"]
    )


def rerun(run_id: str) -> None:
    failed = [job for job in jobs(run_id, 1) if job["conclusion"] in ("failure", "timed_out")]
    setup_failures = [job["name"] for job in failed if not tests_ran(job)]
    if not setup_failures:
        print(f"No job failed before its tests ran; {len(failed)} failed in them.")
        return
    print(f"Failed before their tests ran: {', '.join(setup_failures)}. Rerunning the failed jobs.")
    subprocess.run(
        ["gh", "run", "rerun", run_id, "--failed", "-R", os.environ["GITHUB_REPOSITORY"]],
        check=True,
    )


def guard() -> None:
    run_id, attempt = os.environ["GITHUB_RUN_ID"], int(os.environ["GITHUB_RUN_ATTEMPT"])
    name = next(
        job["name"]
        for job in jobs(run_id, attempt)
        if job["runner_name"] == os.environ["RUNNER_NAME"]
    )
    earlier = [job for job in jobs(run_id, attempt - 1) if job["name"] == name]
    if any(job["conclusion"] != "success" and tests_ran(job) for job in earlier):
        print(f"::error::The tests of {name} failed in attempt {attempt - 1}; that result stands.")
        sys.exit(1)


if __name__ == "__main__":
    if sys.argv[1] == "--guard":
        guard()
    else:
        rerun(sys.argv[1])
