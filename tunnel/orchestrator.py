"""Fleet orchestration for background vLLM instances via pidfiles.

Each instance runs as `python -m tunnel.cli serve <id>` in its own process
group, logging to `logs/<id>.log` with its pid in `.tunnel/<id>.pid`.
Stops SIGTERM the whole group (vLLM spawns workers), escalating to SIGKILL.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from tunnel.registry import InstanceConfig

PID_DIR = Path(".tunnel")
LOG_DIR = Path("logs")


def _write_pidfile(inst_id: str, pid: int) -> None:
    """Record a pid as the tracked process for an instance.

    Args:
        inst_id: The instance ID whose pidfile to write.
        pid: The pid to record.
    """
    PID_DIR.mkdir(parents=True, exist_ok=True)
    (PID_DIR / f"{inst_id}.pid").write_text(str(pid))


def launch_instance(inst: InstanceConfig) -> int:
    """Spawn a vLLM instance in the background and record its pid.

    Args:
        inst: The instance to launch.

    Returns:
        The pid of the spawned process.
    """
    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / f"{inst.id}.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_path.open("a") as logfile:
        logfile.write(f"=== tunnel up {inst.id} @ {timestamp} ===\n")
        logfile.flush()
        proc = subprocess.Popen(
            [sys.executable, "-m", "tunnel.cli", "serve", inst.id],
            stdout=logfile,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    _write_pidfile(inst.id, proc.pid)
    return proc.pid


def adopt_instance(inst_id: str, pid: int) -> None:
    """Track an already-running, untracked process as an instance's pid.

    Used by `tunnel up` when a port is already being listened on by a
    process with no (or a stale) pidfile — e.g. started manually via
    `make serve`, or left behind after a duplicate launch crashed and
    clobbered the original pidfile. Adopting avoids launching a second
    process onto the same port/GPU.

    Args:
        inst_id: The instance ID to adopt.
        pid: The pid of the untracked process already listening on the
            instance's port.
    """
    _write_pidfile(inst_id, pid)


def find_listening_pids(ports: set[int]) -> dict[int, int]:
    """Map each port in `ports` to the pid listening on it, in one system scan.

    Args:
        ports: TCP ports to look up.

    Returns:
        Dict of port -> pid for ports with a LISTEN-state process; ports with
        no listener (or a listener whose pid is not visible) are absent.
    """
    listeners: dict[int, int] = {}
    for conn in psutil.net_connections(kind="tcp"):
        if (
            conn.status == psutil.CONN_LISTEN
            and conn.laddr.port in ports
            and conn.pid is not None
        ):
            listeners[conn.laddr.port] = conn.pid
    return listeners


def find_listening_pid(port: int) -> int | None:
    """Find the pid of the process listening on a local TCP port.

    Args:
        port: The TCP port to check.

    Returns:
        The pid in LISTEN state on `port`, or None if no visible listener.
    """
    return find_listening_pids({port}).get(port)


def is_alive(pid: int) -> bool:
    """Check whether a process with the given pid is running.

    Args:
        pid: The process id to check.

    Returns:
        True if the process exists (including if we lack permission to
        signal it, which still implies it exists), False otherwise.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pid(inst_id: str) -> int | None:
    """Read the tracked pid for an instance, cleaning up a corrupt pidfile.

    Args:
        inst_id: The instance ID whose pidfile to read.

    Returns:
        The pid, or None if the pidfile is missing or unparseable (in which
        case a corrupt pidfile is removed as stale).
    """
    pid_path = PID_DIR / f"{inst_id}.pid"
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text().strip())
    except ValueError:
        pid_path.unlink()
        return None


def stop_instance(inst_id: str, term_wait_s: float = 10.0) -> str:
    """Stop a tracked instance cleanly, escalating to SIGKILL if needed.

    Args:
        inst_id: The instance ID to stop.
        term_wait_s: Seconds to wait for a graceful SIGTERM exit before
            escalating to SIGKILL.

    Returns:
        One of "stopped" (SIGTERM sufficed), "killed" (needed SIGKILL),
        "stale" (pidfile pointed at a dead process), or "absent" (no
        pidfile). Adopted pids may not be process-group leaders, so killpg
        falls back to a plain kill.
    """
    pid_path = PID_DIR / f"{inst_id}.pid"
    pidfile_existed = pid_path.exists()
    pid = read_pid(inst_id)

    if pid is None:
        return "absent" if not pidfile_existed else "stale"

    if not is_alive(pid):
        pid_path.unlink(missing_ok=True)
        return "stale"

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        os.kill(pid, signal.SIGTERM)

    deadline = time.monotonic() + term_wait_s
    still_alive = is_alive(pid)
    while still_alive and time.monotonic() < deadline:
        time.sleep(0.5)
        still_alive = is_alive(pid)

    if still_alive:
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        outcome = "killed"
    else:
        outcome = "stopped"

    pid_path.unlink(missing_ok=True)
    return outcome
