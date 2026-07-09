"""
tests/unit/test_orchestrator.py
================================
Tests for tunnel/orchestrator.py.

PID_DIR and LOG_DIR are relative paths, so every test chdirs into a tmp_path
to stay hermetic. No real subprocesses or real vLLM instances are spawned —
subprocess.Popen and os.kill/os.killpg are monkeypatched where needed.
"""
from __future__ import annotations

import os
import signal

import pytest

from tunnel.orchestrator import (
    LOG_DIR,
    PID_DIR,
    is_alive,
    launch_instance,
    read_pid,
    stop_instance,
)
from tunnel.registry import InstanceConfig


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _minimal_instance(**overrides) -> InstanceConfig:
    data = {"id": "test-model", "model": "org/test-model",
            "port": 8000, "gpu_memory_utilization": 0.40, **overrides}
    return InstanceConfig.model_validate(data)


def test_read_pid_missing_file_returns_none():
    assert read_pid("nope") is None


def test_read_pid_valid_pidfile_returns_int():
    PID_DIR.mkdir(parents=True)
    (PID_DIR / "m1.pid").write_text("12345")
    assert read_pid("m1") == 12345


def test_read_pid_corrupt_pidfile_returns_none_and_removes_file():
    PID_DIR.mkdir(parents=True)
    pid_path = PID_DIR / "m1.pid"
    pid_path.write_text("garbage")
    assert read_pid("m1") is None
    assert not pid_path.exists()


def test_is_alive_true_for_current_process():
    assert is_alive(os.getpid()) is True


def test_is_alive_false_when_os_kill_raises_process_lookup_error(monkeypatch):
    def _raise(pid, sig):
        raise ProcessLookupError

    monkeypatch.setattr(os, "kill", _raise)
    assert is_alive(99999) is False


def test_is_alive_true_when_os_kill_raises_permission_error(monkeypatch):
    def _raise(pid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "kill", _raise)
    assert is_alive(1) is True


def test_stop_instance_absent_when_no_pidfile():
    assert stop_instance("nope") == "absent"


def test_stop_instance_stale_when_pid_dead(monkeypatch):
    PID_DIR.mkdir(parents=True)
    (PID_DIR / "m1.pid").write_text("54321")
    monkeypatch.setattr("tunnel.orchestrator.is_alive", lambda pid: False)

    assert stop_instance("m1") == "stale"
    assert not (PID_DIR / "m1.pid").exists()


def test_stop_instance_stopped_after_sigterm(monkeypatch):
    PID_DIR.mkdir(parents=True)
    (PID_DIR / "m1.pid").write_text("54321")

    alive_sequence = iter([True, False])
    monkeypatch.setattr("tunnel.orchestrator.is_alive", lambda pid: next(alive_sequence))

    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr("time.sleep", lambda s: None)

    outcome = stop_instance("m1", term_wait_s=1.0)

    assert outcome == "stopped"
    assert killpg_calls == [(54321, signal.SIGTERM)]
    assert not (PID_DIR / "m1.pid").exists()


def test_stop_instance_killed_when_sigterm_insufficient(monkeypatch):
    PID_DIR.mkdir(parents=True)
    (PID_DIR / "m1.pid").write_text("54321")

    monkeypatch.setattr("tunnel.orchestrator.is_alive", lambda pid: True)

    killpg_calls = []
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killpg_calls.append((pid, sig)))
    monkeypatch.setattr("time.sleep", lambda s: None)

    outcome = stop_instance("m1", term_wait_s=0.1)

    assert outcome == "killed"
    assert (54321, signal.SIGTERM) in killpg_calls
    assert (54321, signal.SIGKILL) in killpg_calls
    assert not (PID_DIR / "m1.pid").exists()


def test_launch_instance_writes_pidfile_and_log(monkeypatch):
    class _StubProc:
        pid = 42

    captured = {}

    def _fake_popen(cmd, stdout, stderr, start_new_session):
        captured["cmd"] = cmd
        captured["start_new_session"] = start_new_session
        stdout.write("child output\n")
        return _StubProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    inst = _minimal_instance()
    pid = launch_instance(inst)

    assert pid == 42
    assert captured["start_new_session"] is True
    assert captured["cmd"][-2:] == ["serve", "test-model"]

    pid_path = PID_DIR / "test-model.pid"
    assert pid_path.read_text() == "42"

    log_path = LOG_DIR / "test-model.log"
    contents = log_path.read_text()
    assert "=== tunnel up test-model @" in contents
    assert "child output" in contents
