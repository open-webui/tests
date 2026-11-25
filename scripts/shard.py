"""Print the test modules of one CI shard of a suite.

python scripts/shard.py integration 10 1 durations/integration.json

Whole modules are dealt to the shards, so a module's instance boots once per run. A module with no
timings yet is dealt first, one per shard, each counted at the average module time; the rest follow
from the slowest down, each to the shard with the least time so far. Contiguous chunks of the test
order put a whole folder of new slow tests, all counted at the average test time, into one shard.
"""

from __future__ import annotations

import heapq
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def module_times(suite: str, durations: dict[str, float]) -> dict[str, float | None]:
    modules = sorted(path.relative_to(ROOT).as_posix() for path in (ROOT / suite).rglob("test_*.py"))
    times: dict[str, float | None] = dict.fromkeys(modules)
    for test, seconds in durations.items():
        module = test.split("::")[0]
        if module in times:
            times[module] = (times[module] or 0.0) + seconds
    return times


def shards(times: dict[str, float | None], splits: int) -> list[list[str]]:
    known = [seconds for seconds in times.values() if seconds is not None]
    average = sum(known) / len(known) if known else 1.0
    new = [module for module, seconds in times.items() if seconds is None]
    timed = sorted((module for module in times if times[module] is not None), key=lambda module: (-times[module], module))
    groups: list[list[str]] = [[] for _ in range(splits)]
    heap = [(0.0, group) for group in range(splits)]
    for module in new + timed:
        total, group = heapq.heappop(heap)
        groups[group].append(module)
        heapq.heappush(heap, (total + (times[module] or average), group))
    return [sorted(group) for group in groups]


def main() -> None:
    suite, splits, group, durations = sys.argv[1:]
    times = module_times(suite, json.loads((ROOT / durations).read_text()))
    print(" ".join(shards(times, int(splits))[int(group) - 1]))


if __name__ == "__main__":
    main()
