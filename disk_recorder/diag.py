"""Process diagnostics and Windows power management for multi-day runs.

An unattended re-recording session runs for days, so slow resource leaks and
the OS putting the machine to sleep are the two failure modes that matter more
than anything the pipeline itself does wrong:

* :func:`process_stats` samples RSS / handle (fd) count / thread count so a
  leak is visible in the log hours before it becomes an outage.
* :func:`keep_system_awake` stops Windows from sleeping, blanking the display
  or suspending USB while a run is in progress — a suspended USB audio device
  is indistinguishable from a hung one from inside the process.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading

logger = logging.getLogger("disk_recorder.diag")

_IS_WINDOWS = sys.platform == "win32"

# SetThreadExecutionState flags (winbase.h).
_ES_CONTINUOUS = 0x80000000
_ES_SYSTEM_REQUIRED = 0x00000001
_ES_DISPLAY_REQUIRED = 0x00000002


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _windows_stats() -> tuple[float, int]:
    """Return ``(rss_mb, handle_count)`` for the current Windows process."""
    kernel32 = ctypes.windll.kernel32
    process = kernel32.GetCurrentProcess()

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    rss_mb = 0.0
    if ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb):
        rss_mb = counters.WorkingSetSize / (1024.0 * 1024.0)

    handles = ctypes.c_uint32(0)
    if not kernel32.GetProcessHandleCount(process, ctypes.byref(handles)):
        handles.value = -1
    return rss_mb, int(handles.value)


def _posix_stats() -> tuple[float, int]:
    """Return ``(rss_mb, open_fd_count)`` for the current POSIX process."""
    rss_mb = 0.0
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    rss_mb = int(line.split()[1]) / 1024.0
                    break
    except OSError:
        pass
    try:
        handles = len(os.listdir("/proc/self/fd"))
    except OSError:
        handles = -1
    return rss_mb, handles


def process_stats() -> dict[str, float | int]:
    """Sample the current process: RSS in MB, OS handles/fds and threads.

    Values that cannot be obtained on the host are reported as ``-1`` rather
    than raising — diagnostics must never break the run.
    """
    try:
        rss_mb, handles = _windows_stats() if _IS_WINDOWS else _posix_stats()
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.debug("process stats unavailable: %s", exc)
        rss_mb, handles = -1.0, -1
    return {"rss_mb": rss_mb, "handles": handles,
            "threads": threading.active_count()}


def format_stats(stats: dict[str, float | int]) -> str:
    """One-line rendering of :func:`process_stats` for the heartbeat log."""
    return (f"rss={stats['rss_mb']:.0f}MB handles={stats['handles']} "
            f"threads={stats['threads']}")


def keep_system_awake(display: bool = True) -> bool:
    """Ask Windows not to sleep while a run is in progress.

    The flags stay in effect for this thread until :func:`release_system_awake`
    is called, so it must be called from a thread that outlives the run (the
    GUI main thread). A no-op returning ``False`` on non-Windows hosts.
    """
    if not _IS_WINDOWS:
        return False
    flags = _ES_CONTINUOUS | _ES_SYSTEM_REQUIRED
    if display:
        flags |= _ES_DISPLAY_REQUIRED
    try:
        if ctypes.windll.kernel32.SetThreadExecutionState(flags) == 0:
            logger.warning("SetThreadExecutionState failed — the machine may sleep")
            return False
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("cannot inhibit sleep: %s", exc)
        return False
    logger.info("sleep/display timeout inhibited for this session")
    return True


def release_system_awake() -> None:
    """Drop the sleep inhibition requested by :func:`keep_system_awake`."""
    if not _IS_WINDOWS:
        return
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(_ES_CONTINUOUS)
    except Exception:  # pragma: no cover - platform dependent
        pass
