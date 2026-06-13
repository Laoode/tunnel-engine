"""
tunnel/gateway/config_builder.py
=================================
Generates configs/litellm/config.yaml from the TunnelRegistry.

Design: pure functions only. build_litellm_config() takes a registry and
returns a plain dict — testable without touching the filesystem.
write_litellm_config() is the I/O wrapper.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tunnel.registry import TunnelRegistry

_AUTO_HEADER = (
    "# AUTO-GENERATED — do not edit manually.\n"
    "# Source of truth : configs/models.yaml\n"
    "# Regenerate      : make generate\n\n"
)


def build_litellm_config(registry: TunnelRegistry) -> dict:
    """Build the LiteLLM proxy config dict from the registry.

    Pure function: no filesystem access, fully testable.

    Args:
        registry: Validated TunnelRegistry.

    Returns:
        Dict matching the LiteLLM proxy config schema.
    """
    model_list = [
        {
            "model_name": inst.id,
            "litellm_params": {
                "model": f"openai/{inst.model}",
                "api_base": inst.api_base,
                "api_key": "none",
            },
        }
        for inst in registry.instances
    ]

    router_settings: dict = {
        "routing_strategy": registry.litellm.routing_strategy,
        "num_retries": 3,
        "retry_after": 5,
        "allowed_fails": 1,    # failures before a model enters cooldown
        "cooldown_time": 60,   # seconds before retrying a cooled-down model
    }

    # fallbacks: [{primary_id: [fallback_id, ...]}] — only for instances that define one
    fallbacks = [
        {inst.id: inst.fallbacks}
        for inst in registry.instances
        if inst.fallbacks
    ]
    if fallbacks:
        router_settings["fallbacks"] = fallbacks

    return {
        "model_list": model_list,
        "router_settings": router_settings,
        "general_settings": {
            "master_key": registry.litellm.master_key,
        },
        "litellm_settings": {
            "request_timeout": 120,
            "drop_params": True,
        },
    }


def write_litellm_config(
    registry: TunnelRegistry,
    output_path: Path | str = "configs/litellm/config.yaml",
) -> Path:
    """Write the generated LiteLLM config to disk.

    Args:
        registry: Validated TunnelRegistry.
        output_path: Destination path. Parent dirs are created if needed.

    Returns:
        Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = build_litellm_config(registry)
    out.write_text(
        _AUTO_HEADER + yaml.dump(config, default_flow_style=False, sort_keys=False)
    )
    return out

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
    config = build_litellm_config(_make_registry(2))
    assert "fallbacks" not in config["router_settings"]


def test_fallback_only_emitted_for_instances_that_define_one():
    config = build_litellm_config(_make_registry_with_fallbacks())
    fallback_keys = [list(f.keys())[0] for f in config["router_settings"]["fallbacks"]]
    assert "primary" in fallback_keys
    assert "backup" not in fallback_keys
