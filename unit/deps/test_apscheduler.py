"""Dependency contract: APScheduler (import name ``apscheduler``).

APScheduler is a *declared* requirement of the Open WebUI backend
(``APScheduler==3.11.2`` in requirements.txt / requirements-min.txt) but
is not imported by the application code today — the backend runs its own
lightweight asyncio ``scheduler_worker_loop`` (utils/automations.py) for
time-based work. Because it is a pinned dependency that other
integrations / future code may rely on, this module pins the *core public
surface* of APScheduler 3.x so a silent major bump (the 4.x rewrite moves
nearly everything) is caught here rather than breaking an install that
imports it.

The pinned surface is the classic 3.x layout: the schedulers
(``BackgroundScheduler`` / ``AsyncIOScheduler`` / ``BlockingScheduler``),
the triggers (``cron`` / ``interval`` / ``date``), and the job-store /
executor entry points — plus offline behavioural contracts that build a
scheduler, register and remove jobs, and actually fire one job on a
background thread (no network).

Pattern mirrors test_requests.py. Uses ``depcheck`` from conftest.py.
"""

from __future__ import annotations

import inspect
import time

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "apscheduler"
DIST_NAME = "APScheduler"

# Submodule import paths that make up the classic 3.x public API.
SCHEDULER_SUBMODULES = [
    "apscheduler.schedulers.background",
    "apscheduler.schedulers.asyncio",
    "apscheduler.schedulers.blocking",
    "apscheduler.schedulers.base",
]
TRIGGER_SUBMODULES = [
    "apscheduler.triggers.cron",
    "apscheduler.triggers.interval",
    "apscheduler.triggers.date",
]
STORE_EXECUTOR_SUBMODULES = [
    "apscheduler.jobstores.memory",
    "apscheduler.executors.pool",
    "apscheduler.job",
]


# ---------------------------------------------------------------------------
# Import + version
# ---------------------------------------------------------------------------


def test_import(depcheck):
    """`apscheduler` must import (skip cleanly if absent)."""
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "apscheduler"


def test_version_reported(depcheck):
    """The installed distribution version must resolve."""
    assert depcheck.dist_version(DIST_NAME) is not None


def test_is_apscheduler_3x(depcheck):
    """The pinned surface here is the 3.x layout. Guard the major version so a
    4.x bump (which removes BackgroundScheduler.add_job-style API) fails with a
    clear message instead of cryptic import errors elsewhere."""
    ver = depcheck.dist_version(DIST_NAME)
    if ver is None:
        pytest.skip("apscheduler version not resolvable")
    major = int(ver.split(".")[0])
    assert major == 3, (
        f"APScheduler major version is {major}; this contract pins the 3.x API. "
        "A 4.x bump reorganises the public surface — update consumers + this test."
    )


# ---------------------------------------------------------------------------
# Submodule import checks (the classic 3.x public API is package-structured).
# ---------------------------------------------------------------------------


def test_scheduler_submodules_import(depcheck):
    for name in SCHEDULER_SUBMODULES:
        mod = depcheck.load(name)
        assert mod.__name__ == name


def test_trigger_submodules_import(depcheck):
    for name in TRIGGER_SUBMODULES:
        mod = depcheck.load(name)
        assert mod.__name__ == name


def test_store_and_executor_submodules_import(depcheck):
    for name in STORE_EXECUTOR_SUBMODULES:
        mod = depcheck.load(name)
        assert mod.__name__ == name


# ---------------------------------------------------------------------------
# Symbol-existence checks (the load-bearing classes).
# ---------------------------------------------------------------------------


def test_scheduler_classes_exist(depcheck):
    depcheck.load(IMPORT_NAME)
    assert hasattr(depcheck.load("apscheduler.schedulers.background"), "BackgroundScheduler")
    assert hasattr(depcheck.load("apscheduler.schedulers.asyncio"), "AsyncIOScheduler")
    assert hasattr(depcheck.load("apscheduler.schedulers.blocking"), "BlockingScheduler")


def test_trigger_classes_exist(depcheck):
    depcheck.load(IMPORT_NAME)
    assert hasattr(depcheck.load("apscheduler.triggers.cron"), "CronTrigger")
    assert hasattr(depcheck.load("apscheduler.triggers.interval"), "IntervalTrigger")
    assert hasattr(depcheck.load("apscheduler.triggers.date"), "DateTrigger")


def test_jobstore_and_executor_classes_exist(depcheck):
    depcheck.load(IMPORT_NAME)
    assert hasattr(depcheck.load("apscheduler.jobstores.memory"), "MemoryJobStore")
    assert hasattr(depcheck.load("apscheduler.executors.pool"), "ThreadPoolExecutor")
    assert hasattr(depcheck.load("apscheduler.job"), "Job")


def test_scheduler_state_constants_exist(depcheck):
    """Consumers gate behaviour on STATE_STOPPED/RUNNING/PAUSED; pin them."""
    base = depcheck.load("apscheduler.schedulers.base")
    for name in ("STATE_STOPPED", "STATE_RUNNING", "STATE_PAUSED"):
        assert hasattr(base, name), f"apscheduler.schedulers.base.{name} missing"


# ---------------------------------------------------------------------------
# Signature contracts — the core scheduling API.
# ---------------------------------------------------------------------------


def test_add_job_signature(depcheck):
    """BackgroundScheduler.add_job(func, trigger, ..., id=, name=, ...) — pin
    the primary scheduling parameters."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    depcheck.assert_params(bg.add_job, ["func", "trigger", "id"])


def test_cron_trigger_from_crontab_exists(depcheck):
    """CronTrigger.from_crontab('* * * * *') is the canonical way to build a
    cron schedule from a crontab string; pin it."""
    cron = depcheck.load("apscheduler.triggers.cron").CronTrigger
    assert hasattr(cron, "from_crontab")
    assert callable(cron.from_crontab)


def test_scheduler_lifecycle_methods_exist(depcheck):
    """The scheduler lifecycle (start/shutdown/pause/resume) and job management
    (add_job/get_job/get_jobs/remove_job) must all be present."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    for name in (
        "start",
        "shutdown",
        "pause",
        "resume",
        "add_job",
        "get_job",
        "get_jobs",
        "remove_job",
    ):
        assert hasattr(bg, name), f"BackgroundScheduler.{name} missing"


# ---------------------------------------------------------------------------
# Behavioural contracts (OFFLINE) — build a scheduler, manage jobs, fire one.
# ---------------------------------------------------------------------------


def test_behaviour_add_get_remove_job(depcheck):
    """Register a job (not started -> no firing), look it up by id, then remove
    it. Pin the in-memory job registry semantics."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    sched = bg()
    try:
        job = sched.add_job(lambda: None, "interval", seconds=100, id="contract-job")
        assert job.id == "contract-job"
        assert sched.get_job("contract-job") is not None
        assert any(j.id == "contract-job" for j in sched.get_jobs())
        sched.remove_job("contract-job")
        assert sched.get_job("contract-job") is None
    finally:
        # Never started, so nothing to shut down — but be defensive.
        if sched.state != 0:  # STATE_STOPPED
            sched.shutdown(wait=False)


def test_behaviour_cron_trigger_from_crontab(depcheck):
    """from_crontab must parse a 5-field crontab into a usable CronTrigger and
    compute a next fire time from a reference instant."""
    import datetime

    cron = depcheck.load("apscheduler.triggers.cron").CronTrigger
    # Pin the trigger to UTC so the computed fire time is comparable to a
    # UTC `now` (from_crontab otherwise adopts the machine-local tz, which
    # mixes naive/aware datetimes).
    trigger = cron.from_crontab("0 * * * *", timezone=datetime.timezone.utc)
    assert trigger is not None
    # get_next_fire_time(previous, now) is the trigger contract the scheduler
    # uses; it must return a future datetime.
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = trigger.get_next_fire_time(None, now)
    assert nxt is not None
    assert isinstance(nxt, datetime.datetime)
    assert nxt > now


def test_behaviour_interval_trigger_next_fire(depcheck):
    """IntervalTrigger(seconds=N).get_next_fire_time advances by the interval."""
    interval = depcheck.load("apscheduler.triggers.interval").IntervalTrigger
    import datetime

    trig = interval(seconds=30)
    now = datetime.datetime.now(datetime.timezone.utc)
    nxt = trig.get_next_fire_time(None, now)
    assert nxt is not None and nxt >= now


def test_behaviour_background_scheduler_fires_job(depcheck):
    """End-to-end: a started BackgroundScheduler must actually invoke a job on
    its short interval. Bounded busy-wait (no fixed sleep), short shutdown."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    sched = bg(job_defaults={"misfire_grace_time": 30})
    hits: list[int] = []
    sched.add_job(lambda: hits.append(1), "interval", seconds=0.05, id="tick")
    sched.start()
    try:
        deadline = time.time() + 5
        while len(hits) < 1 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        sched.shutdown(wait=False)
    assert hits, "BackgroundScheduler did not fire the scheduled job within 5s"


def test_behaviour_scheduler_accepts_memory_jobstore(depcheck):
    """A MemoryJobStore is the default backend; constructing a scheduler with
    one explicitly (the documented configuration path) must work offline."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    memstore = depcheck.load("apscheduler.jobstores.memory").MemoryJobStore
    sched = bg(jobstores={"default": memstore()})
    try:
        sched.add_job(lambda: None, "interval", seconds=100, id="ms")
        assert sched.get_job("ms") is not None
    finally:
        if sched.state != 0:
            sched.shutdown(wait=False)


def test_behaviour_add_job_signature_inspectable(depcheck):
    """Sanity: add_job remains a normal introspectable method (regression guard
    against it becoming a C-accelerated opaque callable in a bump)."""
    bg = depcheck.load("apscheduler.schedulers.background").BackgroundScheduler
    sig = inspect.signature(bg.add_job)
    assert "func" in sig.parameters
