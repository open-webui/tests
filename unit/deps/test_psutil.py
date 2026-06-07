"""Dependency contract: psutil.

psutil is a **pinned direct requirement** of the Open WebUI backend
(``psutil==7.2.2`` in requirements.txt) but is *not* imported anywhere in the
``open_webui`` package source today — it is available as a system-introspection
utility for the runtime/operational tooling around the server (process and
resource metrics: CPU, memory, disk, process state). Because there is no
in-tree call site, the meaningful contract is psutil's stable *core* public
surface — the cross-platform functions and the ``Process`` class plus its
exception hierarchy — which any health/metrics consumer would reach for.

This module pins that surface and exercises it for real: psutil reads *local*
machine state only (no network, no external services), so the behavioural
checks are fully offline and assert structure + sane value ranges (e.g. a
percent in 0..100, totals > 0, the current process's pid matches os.getpid())
rather than exact numbers. A psutil major bump that removed/renamed a core
function, changed a result namedtuple's fields, or reshaped the exception
hierarchy fails loudly here.

Pattern mirrors the unit/deps/ exemplar: symbol-existence + signature checks,
plus offline behavioural contracts against the local host. Uses the `depcheck`
fixture from unit/deps/conftest.py.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.depcheck

IMPORT_NAME = "psutil"
DIST_NAME = "psutil"

# Core cross-platform module-level functions a metrics/health consumer relies on.
CORE_FUNCTIONS = [
    "cpu_percent",
    "cpu_count",
    "cpu_times",
    "virtual_memory",
    "swap_memory",
    "disk_usage",
    "disk_partitions",
    "net_io_counters",
    "boot_time",
    "pids",
    "process_iter",
    "Process",
]

# The exception types psutil consumers guard process access with.
EXCEPTIONS = ["Error", "NoSuchProcess", "AccessDenied", "ZombieProcess", "TimeoutExpired"]


# --------------------------------------------------------------------------- #
# Import + version + API surface
# --------------------------------------------------------------------------- #
def test_import(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    assert mod.__name__ == "psutil"


def test_version_reported(depcheck):
    """The installed distribution version must be resolvable so bump tooling
    and this suite agree on what's under test."""
    depcheck.load(IMPORT_NAME)
    assert depcheck.dist_version(DIST_NAME) is not None


def test_core_functions_exist(depcheck):
    """Every core psutil function a metrics consumer reaches for must exist."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, CORE_FUNCTIONS)


def test_core_functions_callable(depcheck):
    mod = depcheck.load(IMPORT_NAME)
    for name in CORE_FUNCTIONS:
        assert callable(getattr(mod, name)), f"psutil.{name} not callable"


def test_exception_hierarchy(depcheck):
    """psutil's process exceptions must all subclass ``psutil.Error`` so a broad
    ``except psutil.Error`` guard keeps catching them."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_symbols(mod, EXCEPTIONS)
    base = mod.Error
    for name in ("NoSuchProcess", "AccessDenied", "ZombieProcess", "TimeoutExpired"):
        exc = getattr(mod, name)
        assert issubclass(exc, base), f"{name} no longer subclasses psutil.Error"


# --------------------------------------------------------------------------- #
# Signatures — the common call shapes
# --------------------------------------------------------------------------- #
def test_cpu_percent_signature(depcheck):
    """``cpu_percent(interval=None, percpu=False)`` — pin both keywords."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.cpu_percent, ["interval", "percpu"])


def test_cpu_count_signature(depcheck):
    """``cpu_count(logical=True)`` — pin the logical keyword."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.cpu_count, ["logical"])


def test_disk_usage_signature(depcheck):
    """``disk_usage(path)`` — first positional is the filesystem path."""
    mod = depcheck.load(IMPORT_NAME)
    depcheck.assert_params(mod.disk_usage, ["path"])


# --------------------------------------------------------------------------- #
# Behavioural: CPU
# --------------------------------------------------------------------------- #
def test_cpu_count_positive(depcheck):
    """``cpu_count()`` returns a positive integer (logical CPU count)."""
    mod = depcheck.load(IMPORT_NAME)
    n = mod.cpu_count()
    assert isinstance(n, int)
    assert n >= 1


def test_cpu_percent_is_percentage(depcheck):
    """``cpu_percent()`` returns a float in [0, 100]. (interval=None returns the
    non-blocking instantaneous value — no sleep, stays offline/fast.)"""
    mod = depcheck.load(IMPORT_NAME)
    pct = mod.cpu_percent(interval=None)
    assert isinstance(pct, float)
    assert 0.0 <= pct <= 100.0


def test_cpu_times_namedtuple_fields(depcheck):
    """``cpu_times()`` returns a namedtuple with at least user/system fields —
    the canonical CPU-time accounting consumers read."""
    mod = depcheck.load(IMPORT_NAME)
    ct = mod.cpu_times()
    assert hasattr(ct, "user")
    assert hasattr(ct, "system")
    assert isinstance(ct.user, float)


# --------------------------------------------------------------------------- #
# Behavioural: memory
# --------------------------------------------------------------------------- #
def test_virtual_memory_fields_and_ranges(depcheck):
    """``virtual_memory()`` returns a namedtuple exposing total/available/
    percent/used — the fields any memory-health check reads. total must be
    positive and percent a valid percentage."""
    mod = depcheck.load(IMPORT_NAME)
    vm = mod.virtual_memory()
    for field in ("total", "available", "percent", "used"):
        assert hasattr(vm, field), f"virtual_memory result missing {field}"
    assert vm.total > 0
    assert 0.0 <= vm.percent <= 100.0
    assert vm.available <= vm.total


def test_swap_memory_fields(depcheck):
    """``swap_memory()`` exposes total/used/free/percent (swap may be 0 on some
    hosts, so only structure + ranges are asserted, not positivity of total)."""
    mod = depcheck.load(IMPORT_NAME)
    sm = mod.swap_memory()
    for field in ("total", "used", "free", "percent"):
        assert hasattr(sm, field), f"swap_memory result missing {field}"
    assert sm.total >= 0
    assert 0.0 <= sm.percent <= 100.0


# --------------------------------------------------------------------------- #
# Behavioural: disk
# --------------------------------------------------------------------------- #
def test_disk_usage_of_cwd(depcheck):
    """``disk_usage(path)`` returns total/used/free/percent for the filesystem
    holding ``path``. Query the current working directory (always exists) and
    assert a sane shape."""
    mod = depcheck.load(IMPORT_NAME)
    du = mod.disk_usage(os.getcwd())
    for field in ("total", "used", "free", "percent"):
        assert hasattr(du, field), f"disk_usage result missing {field}"
    assert du.total > 0
    assert 0.0 <= du.percent <= 100.0
    assert du.used + du.free <= du.total + du.total  # sanity, no overflow nonsense


def test_disk_partitions_returns_list(depcheck):
    """``disk_partitions()`` returns a list of partition namedtuples; each has a
    mountpoint. (all=False may be empty in odd sandboxes, so only the container
    type and per-item shape are asserted.)"""
    mod = depcheck.load(IMPORT_NAME)
    parts = mod.disk_partitions(all=True)
    assert isinstance(parts, list)
    for p in parts:
        assert hasattr(p, "mountpoint")
        assert hasattr(p, "device")


# --------------------------------------------------------------------------- #
# Behavioural: network + misc
# --------------------------------------------------------------------------- #
def test_net_io_counters_fields(depcheck):
    """``net_io_counters()`` returns a namedtuple with bytes_sent/bytes_recv —
    the counters a throughput consumer reads. (May be None if no NICs; accept
    that, else assert the fields.)"""
    mod = depcheck.load(IMPORT_NAME)
    counters = mod.net_io_counters()
    if counters is None:
        pytest.skip("no network counters available in this environment")
    assert hasattr(counters, "bytes_sent")
    assert hasattr(counters, "bytes_recv")
    assert counters.bytes_sent >= 0
    assert counters.bytes_recv >= 0


def test_boot_time_is_past_timestamp(depcheck):
    """``boot_time()`` returns a positive UNIX timestamp (the host booted at some
    point in the past)."""
    mod = depcheck.load(IMPORT_NAME)
    bt = mod.boot_time()
    assert isinstance(bt, float)
    assert bt > 0


def test_pids_contains_current_process(depcheck):
    """``pids()`` returns the list of running PIDs and must include this very
    process — the basic liveness invariant."""
    mod = depcheck.load(IMPORT_NAME)
    pids = mod.pids()
    assert isinstance(pids, list)
    assert os.getpid() in pids


# --------------------------------------------------------------------------- #
# Behavioural: Process class — introspect the current process
# --------------------------------------------------------------------------- #
def test_process_current_pid(depcheck):
    """``Process()`` with no arg targets the current process; its ``.pid`` must
    equal ``os.getpid()`` (``pid`` is an attribute, not a method)."""
    mod = depcheck.load(IMPORT_NAME)
    p = mod.Process()
    assert p.pid == os.getpid()


def test_process_method_surface(depcheck):
    """A Process exposes the introspection methods consumers call: name(),
    status(), memory_info(), memory_percent(), cpu_percent(), cpu_times()."""
    mod = depcheck.load(IMPORT_NAME)
    p = mod.Process()
    for name in (
        "name",
        "status",
        "memory_info",
        "memory_percent",
        "cpu_percent",
        "cpu_times",
    ):
        assert callable(getattr(p, name, None)), f"Process.{name} missing/not callable"


def test_process_memory_info(depcheck):
    """``Process().memory_info()`` returns a namedtuple with rss/vms — the
    resident/virtual memory of this process, both non-negative."""
    mod = depcheck.load(IMPORT_NAME)
    mi = mod.Process().memory_info()
    assert hasattr(mi, "rss")
    assert hasattr(mi, "vms")
    assert mi.rss >= 0
    assert mi.vms >= 0


def test_process_name_is_str(depcheck):
    """``Process().name()`` returns the process executable name as a non-empty
    string."""
    mod = depcheck.load(IMPORT_NAME)
    name = mod.Process().name()
    assert isinstance(name, str)
    assert name  # non-empty


def test_nonexistent_process_raises_nosuchprocess(depcheck):
    """Constructing/accessing a Process for an impossible PID must raise
    ``NoSuchProcess`` — the failure mode a process-monitor guards. Use a PID far
    outside any plausible range."""
    mod = depcheck.load(IMPORT_NAME)
    impossible_pid = 2_000_000_000
    with pytest.raises(mod.NoSuchProcess):
        # Either construction or the first attribute access raises; name() forces it.
        mod.Process(impossible_pid).name()


def test_process_iter_yields_processes(depcheck):
    """``process_iter()`` yields Process objects; iterating must surface this
    process's pid among them (offline scan of the local process table)."""
    mod = depcheck.load(IMPORT_NAME)
    my_pid = os.getpid()
    found = False
    for proc in mod.process_iter(["pid"]):
        if proc.info.get("pid") == my_pid:
            found = True
            break
    assert found, "process_iter did not yield the current process"
