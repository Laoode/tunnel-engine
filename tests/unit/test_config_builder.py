import tempfile
from pathlib import Path
import pytest
import yaml
from tunnel.registry import TunnelRegistry
from tunnel.gateway.config_builder import build_litellm_config, write_litellm_config


def _reg(n=2):
    return TunnelRegistry.model_validate({"instances": [
        {"id": f"model-{i}", "model": f"org/model-{i}",
         "port": 8000+i, "gpu_memory_utilization": 0.40}
        for i in range(n)
    ], "litellm": {"port": 4000, "master_key": "sk-test",
                   "routing_strategy": "least-busy"}})


def test_model_list_has_all_instances():
    names = {m["model_name"] for m in build_litellm_config(_reg(3))["model_list"]}
    assert names == {"model-0", "model-1", "model-2"}

def test_api_base_matches_port():
    cfg = build_litellm_config(_reg())
    e = next(m for m in cfg["model_list"] if m["model_name"] == "model-0")
    assert e["litellm_params"]["api_base"] == "http://localhost:8000/v1"

def test_model_prefixed_with_openai():
    cfg = build_litellm_config(_reg(1))
    assert cfg["model_list"][0]["litellm_params"]["model"].startswith("openai/")

def test_retry_settings_present():
    rs = build_litellm_config(_reg())["router_settings"]
    assert "num_retries" in rs and "cooldown_time" in rs

def test_write_creates_file_and_parent_dirs():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "deep" / "config.yaml"
        write_litellm_config(_reg(), out)
        assert out.exists()
        assert "AUTO-GENERATED" in out.read_text()
