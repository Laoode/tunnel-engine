import tempfile
from pathlib import Path
from tunnel.registry import TunnelRegistry
from tunnel.gateway.config_builder import build_litellm_config, write_litellm_config


def _reg(n=2):
    return TunnelRegistry.model_validate({"instances": [
        {"id": f"model-{i}", "model": f"org/model-{i}",
         "port": 8000+i, "gpu_memory_utilization": 0.10}
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

def _make_registry_with_fallbacks() -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "primary", "model": "org/primary", "port": 8000,
             "gpu_memory_utilization": 0.40, "fallbacks": ["backup"]},
            {"id": "backup", "model": "org/backup", "port": 8001,
             "gpu_memory_utilization": 0.40},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test",
                    "routing_strategy": "least-busy"},
    })


def test_fallbacks_emitted_in_router_settings():
    config = build_litellm_config(_make_registry_with_fallbacks())
    fallbacks = config["router_settings"]["fallbacks"]
    assert {"primary": ["backup"]} in fallbacks


def test_no_fallbacks_key_when_none_configured():
    config = build_litellm_config(_reg(2))
    assert "fallbacks" not in config["router_settings"]


def test_fallback_only_emitted_for_instances_that_define_one():
    config = build_litellm_config(_make_registry_with_fallbacks())
    fallback_keys = [list(f.keys())[0] for f in config["router_settings"]["fallbacks"]]
    assert "primary" in fallback_keys
    assert "backup" not in fallback_keys


def test_prometheus_callback_emitted_when_enabled():
    reg = TunnelRegistry.model_validate({
        "instances": [
            {"id": "m", "model": "org/m", "port": 8000, "gpu_memory_utilization": 0.4}
        ],
        "litellm": {"port": 4000, "master_key": "sk-test",
                    "routing_strategy": "least-busy", "prometheus": True},
    })
    config = build_litellm_config(reg)
    assert "prometheus" in config["litellm_settings"].get("callbacks", [])


def test_prometheus_callback_absent_when_disabled():
    config = build_litellm_config(_reg(1))
    assert "callbacks" not in config["litellm_settings"]


def test_served_model_name_used_when_set():
    reg = TunnelRegistry.model_validate({
        "instances": [
            {"id": "m", "model": "org/m", "port": 8000, "gpu_memory_utilization": 0.4,
             "served_model_name": "custom-name"},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test",
                    "routing_strategy": "least-busy"},
    })
    config = build_litellm_config(reg)
    assert config["model_list"][0]["litellm_params"]["model"] == "openai/custom-name"


def test_served_model_name_unset_falls_back_to_model():
    cfg = build_litellm_config(_reg(1))
    assert cfg["model_list"][0]["litellm_params"]["model"] == "openai/org/model-0"


def _reg_with_remote() -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "local-a", "model": "org/a", "port": 8000,
             "gpu_memory_utilization": 0.4},
        ],
        "remote_models": [
            {"id": "deepseek-v4-pro", "upstream_model": "deepseek-v4-pro",
             "api_base": "https://api.deepseek.com", "api_key_env": "DEEPSEEK_API_KEY"},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test",
                    "routing_strategy": "least-busy"},
    })


def test_remote_model_appended_to_model_list():
    entry = next(
        m for m in build_litellm_config(_reg_with_remote())["model_list"]
        if m["model_name"] == "deepseek-v4-pro"
    )
    assert entry["litellm_params"]["model"] == "openai/deepseek-v4-pro"
    assert entry["litellm_params"]["api_base"] == "https://api.deepseek.com"


def test_remote_model_api_key_is_env_reference_not_secret():
    entry = next(
        m for m in build_litellm_config(_reg_with_remote())["model_list"]
        if m["model_name"] == "deepseek-v4-pro"
    )
    assert entry["litellm_params"]["api_key"] == "os.environ/DEEPSEEK_API_KEY"


def _reg_with_services():
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "model-0", "model": "org/model-0", "port": 8000,
             "gpu_memory_utilization": 0.10,
             "cost": {"input_per_mtok": 0.05, "output_per_mtok": 0.20}},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test"},
        "tiers": {"free": {"priority": 2, "rpm_limit": 60, "tpm_limit": 20000}},
        "services": [{"id": "dev", "tier": "free"}],
    })


def test_database_url_emitted_only_with_services():
    with_db = build_litellm_config(_reg_with_services())["general_settings"]
    assert with_db["database_url"] == "os.environ/DATABASE_URL"
    assert with_db["store_model_in_db"] is False

    without_db = build_litellm_config(_reg(1))["general_settings"]
    assert "database_url" not in without_db
    assert "store_model_in_db" not in without_db


def test_cost_emitted_as_per_token():
    params = build_litellm_config(_reg_with_services())["model_list"][0]["litellm_params"]
    assert params["input_cost_per_token"] == 0.05 / 1_000_000
    assert params["output_cost_per_token"] == 0.20 / 1_000_000


def test_no_cost_params_when_cost_unset():
    params = build_litellm_config(_reg(1))["model_list"][0]["litellm_params"]
    assert "input_cost_per_token" not in params
