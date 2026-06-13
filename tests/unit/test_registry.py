"""
Tests for tunnel/registry.py.
Contract tests — every validation guard has a failing test case.
"""
import tempfile
from pathlib import Path

import pytest
import yaml

from tunnel.registry import load_registry


def _write(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def _minimal_instance(**overrides) -> dict:
    return {"id": "test-model", "model": "org/test-model",
            "port": 8000, "gpu_memory_utilization": 0.40, **overrides}


def _minimal_registry(*instances) -> dict:
    return {"instances": list(instances) or [_minimal_instance()]}


# Happy path
def test_load_valid_registry():
    reg = load_registry(_write(_minimal_registry(
        _minimal_instance(id="a", port=8000),
        _minimal_instance(id="b", port=8001),
    )))
    assert len(reg.instances) == 2

def test_api_base_property():
    reg = load_registry(_write(_minimal_registry(_minimal_instance(port=8000))))
    assert reg.instances[0].api_base == "http://localhost:8000/v1"

def test_get_instance_found():
    reg = load_registry(_write(_minimal_registry(_minimal_instance(id="target"))))
    assert reg.get_instance("target") is not None

def test_get_instance_not_found():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.get_instance("does-not-exist") is None

# Defaults merging
def test_defaults_fill_missing_fields():
    data = {"defaults": {"max_model_len": 4096},
            "instances": [_minimal_instance()]}
    reg = load_registry(_write(data))
    assert reg.instances[0].max_model_len == 4096

def test_instance_overrides_default():
    data = {"defaults": {"gpu_memory_utilization": 0.5},
            "instances": [_minimal_instance(gpu_memory_utilization=0.35)]}
    reg = load_registry(_write(data))
    assert reg.instances[0].gpu_memory_utilization == 0.35

# Collision guards
def test_duplicate_ports_rejected():
    with pytest.raises(Exception, match="[Dd]uplicate.*port"):
        load_registry(_write(_minimal_registry(
            _minimal_instance(id="a", port=8000),
            _minimal_instance(id="b", port=8000),
        )))

def test_duplicate_ids_rejected():
    with pytest.raises(Exception, match="[Dd]uplicate.*[Ii][Dd]"):
        load_registry(_write(_minimal_registry(
            _minimal_instance(id="same", port=8000),
            _minimal_instance(id="same", port=8001),
        )))

def test_instance_port_collides_with_litellm():
    with pytest.raises(Exception, match="collides"):
        load_registry(_write({
            "instances": [_minimal_instance(port=4000)],
            "litellm": {"port": 4000, "master_key": "sk-test"},
        }))

# Field validation
def test_port_below_range_rejected():
    with pytest.raises(Exception, match="[Pp]ort"):
        load_registry(_write(_minimal_registry(_minimal_instance(port=80))))

def test_gpu_mem_above_one_rejected():
    with pytest.raises(Exception):
        load_registry(_write(_minimal_registry(_minimal_instance(gpu_memory_utilization=1.5))))

def test_lora_enabled_without_modules_rejected():
    with pytest.raises(Exception, match="[Ll]o[Rr][Aa]"):
        load_registry(_write(_minimal_registry(
            {**_minimal_instance(), "lora": {"enabled": True, "modules": []}}
        )))

def test_invalid_lmcache_backend_rejected():
    with pytest.raises(Exception, match="backend"):
        load_registry(_write(_minimal_registry(
            {**_minimal_instance(),
             "lmcache": {"enabled": True, "backend": "nvme",
                         "max_cache_size_gb": 10, "chunk_size": 256}}
        )))

def test_invalid_routing_strategy_rejected():
    with pytest.raises(Exception, match="routing_strategy"):
        load_registry(_write({
            "instances": [_minimal_instance()],
            "litellm": {"port": 4000, "routing_strategy": "round-robin-invalid"},
        }))
