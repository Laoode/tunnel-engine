"""
Tests for tunnel/cli.py — build_serve_command().
Pure function tests: no process launching, no filesystem writes besides tmp_path.
"""
from tunnel.cli import build_serve_command
from tunnel.registry import InstanceConfig


def _minimal_instance(**overrides) -> InstanceConfig:
    data = {"id": "test-model", "model": "org/test-model",
            "port": 8000, "gpu_memory_utilization": 0.40, **overrides}
    return InstanceConfig.model_validate(data)


def test_base_command_for_minimal_instance():
    cmd = build_serve_command(_minimal_instance())
    assert cmd == [
        "vllm", "serve", "org/test-model",
        "--port", "8000",
        "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.4",
        "--max-model-len", "65536",
        "--dtype", "auto",
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
