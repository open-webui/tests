"""Record the slowest finished wall time of every job a CI run ran in .github/job-times.json.

python scripts/refresh_job_times.py .github/job-times.json regression 45153867826

The file holds seconds per workflow and job, shards of a matrix job counted as one job, and the
workflow sets each job's timeout to twice that. Only jobs that finished, green or red, count: a
cancelled or timed-out job measured the hang, not the job. The time runs from when a runner picked
the job up, as its timeout does, so time spent queued never counts. Needs `gh` with a token.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def finished_jobs(run_id: str) -> dict[str, int]:
    # every attempt's jobs; GitHub answers 502 to a page of 100 of them
    url = f"repos/open-webui/tests/actions/runs/{run_id}/jobs?filter=all&per_page=50"
    command = ["gh", "api", "--paginate", url, "--jq", ".jobs[]"]
    pages = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    times: dict[str, int] = {}
    for line in pages.splitlines():
        job = json.loads(line)
        if job["conclusion"] not in ("success", "failure"):
            continue
        # "integration (3)" is a shard of the integration job
        name = job["name"].split(" (")[0]
        started, completed = (
            datetime.fromisoformat(job[key]) for key in ("started_at", "completed_at")
        )
        times[name] = max(times.get(name, 0), int((completed - started).total_seconds()))
    return times


def main() -> None:
    target, workflow, *run_ids = sys.argv[1:]
    path = Path(target)
    recorded = json.loads(path.read_text()) if path.is_file() else {}
    jobs = recorded.setdefault(workflow, {})
    for run_id in run_ids:
        for name, seconds in finished_jobs(run_id).items():
            jobs[name] = max(jobs.get(name, 0), seconds)
    path.write_text(json.dumps(recorded, indent=4, sort_keys=True) + "\n")
    print(f"{workflow}: {jobs}")


if __name__ == "__main__":
    main()
