"""
Tests for tunnel/cli.py — build_serve_command() and cmd_up()'s launch/adopt/skip
decision. Pure function tests: no process launching, no filesystem writes
besides tmp_path. cmd_up tests mock orchestrator calls and wait_for_all/cmd_proxy
so no real subprocess, socket, or exec happens.
"""
from tunnel import cli
from tunnel.cli import build_serve_command, cmd_up
from tunnel.registry import InstanceConfig
from tunnel.startup import StartupResult


def _minimal_instance(**overrides) -> InstanceConfig:
    data = {"id": "test-model", "model": "org/test-model",
            "port": 8000, "gpu_memory_utilization": 0.40, "max_model_len": 16384, **overrides}
    return InstanceConfig.model_validate(data)


def test_base_command_for_minimal_instance():
    cmd = build_serve_command(_minimal_instance())
    assert cmd == [
        "vllm", "serve", "org/test-model",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "16384",
        "--dtype", "auto",
        "--default-chat-template-kwargs", '{"enable_thinking": false}',
    ]


def test_quantization_flag_present_when_set():
    cmd = build_serve_command(_minimal_instance(quantization="fp8"))
    assert "--quantization" in cmd
    assert cmd[cmd.index("--quantization") + 1] == "fp8"


def test_quantization_flag_absent_when_none():
    cmd = build_serve_command(_minimal_instance())
    assert "--quantization" not in cmd


def test_served_model_name_flag_present_when_set():
    cmd = build_serve_command(_minimal_instance(served_model_name="custom-name"))
    assert "--served-model-name" in cmd
    assert cmd[cmd.index("--served-model-name") + 1] == "custom-name"


def test_served_model_name_flag_absent_when_none():
    cmd = build_serve_command(_minimal_instance())
    assert "--served-model-name" not in cmd


def test_tool_parser_produces_both_flags():
    cmd = build_serve_command(_minimal_instance(tool_parser="hermes"))
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd
    assert cmd[cmd.index("--tool-call-parser") + 1] == "hermes"


def test_tool_parser_flags_absent_when_none():
    cmd = build_serve_command(_minimal_instance())
    assert "--enable-auto-tool-choice" not in cmd
    assert "--tool-call-parser" not in cmd


def test_reasoning_parser_flag_present_when_set():
    cmd = build_serve_command(_minimal_instance(reasoning_parser="qwen3"))
    assert "--reasoning-parser" in cmd
    assert cmd[cmd.index("--reasoning-parser") + 1] == "qwen3"


def test_reasoning_parser_flag_absent_when_none():
    cmd = build_serve_command(_minimal_instance())
    assert "--reasoning-parser" not in cmd


def test_chat_template_included_when_path_exists(tmp_path):
    template = tmp_path / "template.jinja2"
    template.write_text("{# template #}")
    cmd = build_serve_command(_minimal_instance(chat_template=str(template)))
    assert "--chat-template" in cmd
    assert cmd[cmd.index("--chat-template") + 1] == str(template)


def test_chat_template_omitted_when_path_missing():
    cmd = build_serve_command(_minimal_instance(chat_template="/no/such/path.jinja2"))
    assert "--chat-template" not in cmd


def test_extra_args_come_last():
    cmd = build_serve_command(_minimal_instance(
        quantization="fp8",
        tool_parser="hermes",
        reasoning_parser="qwen3",
        served_model_name="custom-name",
        extra_args=["--foo", "bar"],
    ))
    assert cmd[-2:] == ["--foo", "bar"]


class _FakeRegistry:
    def __init__(self, instances):
        self.instances = instances


async def _fake_wait_for_all(registry, timeout_s):
    return StartupResult(ready=True, elapsed_s=0.1, failed_instances=[])


def _patch_up_scaffolding(monkeypatch, registry):
    """Patch everything cmd_up touches besides the launch/adopt/skip decision."""
    monkeypatch.setattr(cli, "load_registry", lambda path: registry)
    monkeypatch.setattr(cli, "wait_for_all", _fake_wait_for_all)
    monkeypatch.setattr(cli, "cmd_proxy", lambda: None)


def test_cmd_up_skips_tracked_live_instance(monkeypatch, capsys):
    inst = _minimal_instance()
    registry = _FakeRegistry([inst])
    _patch_up_scaffolding(monkeypatch, registry)

    monkeypatch.setattr(cli, "read_pid", lambda inst_id: 123)
    monkeypatch.setattr(cli, "is_alive", lambda pid: True)
    monkeypatch.setattr(
        cli, "find_listening_pid",
        lambda port: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    launch_calls = []
    monkeypatch.setattr(cli, "launch_instance", lambda i: launch_calls.append(i) or 999)
    adopt_calls = []
    monkeypatch.setattr(
        cli, "adopt_instance", lambda inst_id, pid: adopt_calls.append((inst_id, pid))
    )

    cmd_up([])

    assert launch_calls == []
    assert adopt_calls == []
    assert "already running (pid 123)" in capsys.readouterr().err


def test_cmd_up_adopts_untracked_listening_process(monkeypatch, capsys):
    inst = _minimal_instance()
    registry = _FakeRegistry([inst])
    _patch_up_scaffolding(monkeypatch, registry)

    monkeypatch.setattr(cli, "read_pid", lambda inst_id: None)
    monkeypatch.setattr(cli, "find_listening_pid", lambda port: 4242)
    launch_calls = []
    monkeypatch.setattr(cli, "launch_instance", lambda i: launch_calls.append(i) or 999)
    adopt_calls = []
    monkeypatch.setattr(
        cli, "adopt_instance", lambda inst_id, pid: adopt_calls.append((inst_id, pid))
    )

    cmd_up([])

    assert launch_calls == []
    assert adopt_calls == [(inst.id, 4242)]
    err = capsys.readouterr().err
    assert "adopted untracked process on :8000 (pid 4242)" in err


def test_cmd_up_launches_when_port_free(monkeypatch, capsys):
    inst = _minimal_instance()
    registry = _FakeRegistry([inst])
    _patch_up_scaffolding(monkeypatch, registry)

    monkeypatch.setattr(cli, "read_pid", lambda inst_id: None)
    monkeypatch.setattr(cli, "find_listening_pid", lambda port: None)
    launch_calls = []
    monkeypatch.setattr(cli, "launch_instance", lambda i: launch_calls.append(i) or 555)
    adopt_calls = []
    monkeypatch.setattr(
        cli, "adopt_instance", lambda inst_id, pid: adopt_calls.append((inst_id, pid))
    )

    cmd_up([])

    assert launch_calls == [inst]
    assert adopt_calls == []
    assert "launched (pid 555)" in capsys.readouterr().err
