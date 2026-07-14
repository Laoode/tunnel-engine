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

def test_valid_fallback_accepted():
    data = _minimal_registry(
        _minimal_instance(id="primary", port=8000, fallbacks=["backup"]),
        _minimal_instance(id="backup", port=8001),
    )
    reg = load_registry(_write(data))
    assert reg.instances[0].fallbacks == ["backup"]


def test_fallback_unknown_id_rejected():
    data = _minimal_registry(
        _minimal_instance(id="a", port=8000, fallbacks=["does-not-exist"]),
    )
    with pytest.raises(Exception, match="unknown IDs"):
        load_registry(_write(data))


def test_fallback_self_reference_rejected():
    data = _minimal_registry(
        _minimal_instance(id="a", port=8000, fallbacks=["a"]),
    )
    with pytest.raises(Exception, match="itself"):
        load_registry(_write(data))


def test_fallback_defaults_to_empty():
    reg = load_registry(_write(_minimal_registry(_minimal_instance())))
    assert reg.instances[0].fallbacks == []


# New per-instance fields
def test_new_instance_fields_default_to_none():
    reg = load_registry(_write(_minimal_registry(_minimal_instance())))
    inst = reg.instances[0]
    assert inst.tool_parser is None
    assert inst.reasoning_parser is None
    assert inst.quantization is None
    assert inst.served_model_name is None


def test_new_instance_fields_round_trip_when_set():
    data = _minimal_registry(_minimal_instance(
        tool_parser="hermes",
        reasoning_parser="qwen3",
        quantization="fp8",
        served_model_name="my-model",
    ))
    reg = load_registry(_write(data))
    inst = reg.instances[0]
    assert inst.tool_parser == "hermes"
    assert inst.reasoning_parser == "qwen3"
    assert inst.quantization == "fp8"
    assert inst.served_model_name == "my-model"


# GPU budget validation
def test_gpu_budget_valid_sum_passes():
    data = _minimal_registry(
        _minimal_instance(id="a", port=8000, gpu_memory_utilization=0.35),
        _minimal_instance(id="b", port=8001, gpu_memory_utilization=0.45),
    )
    reg = load_registry(_write(data))
    assert reg.gpu.budget == 0.90


def test_gpu_budget_over_budget_rejected_with_instance_ids():
    data = _minimal_registry(
        _minimal_instance(id="qwen-0.8b", port=8000, gpu_memory_utilization=0.35),
        _minimal_instance(id="minicpm-1b", port=8001, gpu_memory_utilization=0.45),
        _minimal_instance(id="big", port=8002, gpu_memory_utilization=0.30),
    )
    with pytest.raises(Exception, match="qwen-0.8b=0.35"):
        load_registry(_write(data))


def test_gpu_budget_out_of_range_rejected():
    data = {**_minimal_registry(_minimal_instance()), "gpu": {"budget": 1.5}}
    with pytest.raises(Exception, match="gpu.budget"):
        load_registry(_write(data))


def test_gpu_budget_defaults_when_block_absent():
    reg = load_registry(_write(_minimal_registry(_minimal_instance())))
    assert reg.gpu.budget == 0.90


# Remote models (DeepSeek and other hosted OpenAI-compatible upstreams)
def _remote(**overrides) -> dict:
    return {"id": "deepseek-v4-pro", "upstream_model": "deepseek-v4-pro",
            "api_base": "https://api.deepseek.com",
            "api_key_env": "DEEPSEEK_API_KEY", **overrides}


def test_remote_models_default_to_empty():
    reg = load_registry(_write(_minimal_registry(_minimal_instance())))
    assert reg.remote_models == []


def test_remote_model_loads_without_port_or_gpu():
    data = {"instances": [_minimal_instance()], "remote_models": [_remote()]}
    reg = load_registry(_write(data))
    assert reg.remote_models[0].provider == "openai"       # default prefix
    assert reg.remote_models[0].upstream_model == "deepseek-v4-pro"


def test_remote_does_not_count_against_gpu_budget():
    # Instance already at budget; adding a remote model must not tip it over.
    data = {"instances": [_minimal_instance(gpu_memory_utilization=0.90)],
            "remote_models": [_remote()]}
    reg = load_registry(_write(data))
    assert len(reg.remote_models) == 1


def test_remote_id_colliding_with_instance_rejected():
    data = {"instances": [_minimal_instance(id="shared")],
            "remote_models": [_remote(id="shared")]}
    with pytest.raises(Exception, match="[Dd]uplicate.*ID"):
        load_registry(_write(data))


def test_local_can_fall_back_to_remote_model():
    data = {"instances": [_minimal_instance(id="local", fallbacks=["deepseek-v4-pro"])],
            "remote_models": [_remote()]}
    reg = load_registry(_write(data))
    assert reg.instances[0].fallbacks == ["deepseek-v4-pro"]


def test_invalid_remote_serde_rejected():
    with pytest.raises(Exception, match="remote_serde"):
        load_registry(_write(_minimal_registry(
            {**_minimal_instance(),
             "lmcache": {"enabled": True, "backend": "redis", "remote_serde": "zstd"}}
        )))


# Tiers + services
def _tiered_registry(**overrides) -> dict:
    data = {
        "instances": [_minimal_instance()],
        "tiers": {"free": {"priority": 2, "rpm_limit": 60, "tpm_limit": 20000}},
        "services": [{"id": "dev", "tier": "free", "models": ["test-model"]}],
    }
    data.update(overrides)
    return data


def test_tiers_and_services_load():
    reg = load_registry(_write(_tiered_registry()))
    assert reg.tiers["free"].priority == 2
    assert reg.services[0].models == ["test-model"]


def test_tiers_default_empty():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.tiers == {} and reg.services == []


def test_service_with_unknown_tier_rejected():
    with pytest.raises(Exception, match="unknown tier"):
        load_registry(_write(_tiered_registry(
            services=[{"id": "dev", "tier": "platinum"}])))


def test_service_with_unknown_model_rejected():
    with pytest.raises(Exception, match="unknown ID"):
        load_registry(_write(_tiered_registry(
            services=[{"id": "dev", "tier": "free", "models": ["nope"]}])))


def test_service_may_reference_remote_model():
    data = _tiered_registry(
        remote_models=[{"id": "remote-x", "upstream_model": "x",
                        "api_base": "https://api.example.com",
                        "api_key_env": "X_KEY"}],
        services=[{"id": "dev", "tier": "free", "models": ["remote-x"]}],
    )
    reg = load_registry(_write(data))
    assert reg.services[0].models == ["remote-x"]


def test_duplicate_service_ids_rejected():
    with pytest.raises(Exception, match="[Dd]uplicate service"):
        load_registry(_write(_tiered_registry(
            services=[{"id": "dev", "tier": "free"},
                      {"id": "dev", "tier": "free"}])))


def test_negative_tier_priority_rejected():
    with pytest.raises(Exception, match="priority"):
        load_registry(_write(_tiered_registry(
            tiers={"free": {"priority": -1, "rpm_limit": 60, "tpm_limit": 20000}})))


def test_zero_tier_limits_rejected():
    with pytest.raises(Exception, match="limits"):
        load_registry(_write(_tiered_registry(
            tiers={"free": {"priority": 2, "rpm_limit": 0, "tpm_limit": 20000}})))


# Cost
def test_instance_cost_loads():
    reg = load_registry(_write(_minimal_registry(
        _minimal_instance(cost={"input_per_mtok": 0.05, "output_per_mtok": 0.20}))))
    assert reg.instances[0].cost.output_per_mtok == 0.20


def test_negative_cost_rejected():
    with pytest.raises(Exception, match="cost"):
        load_registry(_write(_minimal_registry(
            _minimal_instance(cost={"input_per_mtok": -1, "output_per_mtok": 0.2}))))


def test_cost_defaults_to_none():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.instances[0].cost is None


def test_service_ids_colliding_after_env_normalization_rejected():
    with pytest.raises(Exception, match="normalization"):
        load_registry(_write(_tiered_registry(
            services=[{"id": "a-b", "tier": "free"},
                      {"id": "a_b", "tier": "free"}])))


def test_scheduling_policy_valid_values():
    reg = load_registry(_write(_minimal_registry(
        _minimal_instance(scheduling_policy="priority"))))
    assert reg.instances[0].scheduling_policy == "priority"


def test_invalid_scheduling_policy_rejected():
    with pytest.raises(Exception, match="scheduling_policy"):
        load_registry(_write(_minimal_registry(
            _minimal_instance(scheduling_policy="weighted-fair"))))


def test_scheduling_policy_defaults_to_none():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.instances[0].scheduling_policy is None


# Guardrails
def _guarded_registry(**guardrail_overrides) -> dict:
    return {
        "instances": [
            _minimal_instance(),
            _minimal_instance(id="guard", model="org/guard", port=8002,
                              gpu_memory_utilization=0.10, internal=True),
        ],
        "guardrails": {"model": "guard", **guardrail_overrides},
    }


def test_guardrails_block_loads_with_defaults():
    reg = load_registry(_write(_guarded_registry()))
    assert reg.guardrails.enabled is True
    assert reg.guardrails.threshold == 0.5
    assert reg.guardrails.check_output is False
    assert reg.guardrails.on_error == "allow"


def test_guardrails_defaults_to_none():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.guardrails is None


def test_guardrails_unknown_model_rejected():
    with pytest.raises(Exception, match="not a"):
        load_registry(_write({
            "instances": [_minimal_instance()],
            "guardrails": {"model": "ghost"},
        }))


def test_guardrails_model_must_be_internal():
    data = _guarded_registry()
    data["instances"][1]["internal"] = False
    with pytest.raises(Exception, match="internal"):
        load_registry(_write(data))


def test_guardrails_invalid_threshold_rejected():
    with pytest.raises(Exception, match="threshold"):
        load_registry(_write(_guarded_registry(threshold=1.5)))


def test_guardrails_invalid_on_error_rejected():
    with pytest.raises(Exception, match="on_error"):
        load_registry(_write(_guarded_registry(on_error="explode")))


def test_service_referencing_internal_instance_rejected():
    data = _guarded_registry()
    data["tiers"] = {"free": {"priority": 2, "rpm_limit": 60, "tpm_limit": 20000}}
    data["services"] = [{"id": "dev", "tier": "free", "models": ["guard"]}]
    with pytest.raises(Exception, match="unknown IDs"):
        load_registry(_write(data))


def test_fallback_to_internal_instance_rejected():
    data = _guarded_registry()
    data["instances"][0]["fallbacks"] = ["guard"]
    with pytest.raises(Exception, match="unknown IDs"):
        load_registry(_write(data))


def test_service_guardrails_optout_loads():
    data = _guarded_registry()
    data["tiers"] = {"free": {"priority": 2, "rpm_limit": 60, "tpm_limit": 20000}}
    data["services"] = [{"id": "dev", "tier": "free", "guardrails": False}]
    reg = load_registry(_write(data))
    assert reg.services[0].guardrails is False


def test_internal_defaults_to_false():
    reg = load_registry(_write(_minimal_registry()))
    assert reg.instances[0].internal is False
