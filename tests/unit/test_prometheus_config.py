"""
Tests for tunnel/observability/prometheus_config.py.
"""
import tempfile
from pathlib import Path

from tunnel.observability.prometheus_config import (
    build_prometheus_config,
    write_prometheus_config,
)
from tunnel.registry import TunnelRegistry


def _reg() -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "model-a", "model": "org/model-a", "port": 8000,
             "gpu_memory_utilization": 0.10},
            {"id": "model-b", "model": "org/model-b", "port": 8001,
             "gpu_memory_utilization": 0.10},
        ],
        "remote_models": [
            {"id": "remote-x", "upstream_model": "x",
             "api_base": "https://api.example.com", "api_key_env": "X_KEY"},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test"},
    })


def test_one_target_per_local_instance():
    cfg = build_prometheus_config(_reg())
    statics = cfg["scrape_configs"][0]["static_configs"]
    assert [s["targets"] for s in statics] == [["localhost:8000"], ["localhost:8001"]]


def test_targets_labeled_with_instance_id_and_model():
    statics = build_prometheus_config(_reg())["scrape_configs"][0]["static_configs"]
    assert statics[0]["labels"] == {"instance_id": "model-a", "model": "org/model-a"}


def test_remote_models_not_scraped():
    text = str(build_prometheus_config(_reg()))
    assert "remote-x" not in text


def test_single_vllm_job():
    jobs = [j["job_name"] for j in build_prometheus_config(_reg())["scrape_configs"]]
    assert jobs == ["vllm"]


def test_write_creates_file_with_header():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "prom" / "prometheus.yml"
        write_prometheus_config(_reg(), out)
        assert out.read_text().startswith("# AUTO-GENERATED")
