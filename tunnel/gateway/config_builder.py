"""
tunnel/gateway/config_builder.py
=================================
Generates configs/litellm/config.yaml from the TunnelRegistry.

Pure functions only. build_litellm_config() takes a registry and returns
a plain dict — testable without touching the filesystem.
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
    """Pure function: TunnelRegistry → LiteLLM proxy config dict."""
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

    return {
        "model_list": model_list,
        "router_settings": {
            "routing_strategy": registry.litellm.routing_strategy,
            "num_retries": 3,
            "retry_after": 5,
            "allowed_fails": 1,
            "cooldown_time": 60,
        },
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
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = build_litellm_config(registry)
    out.write_text(
        _AUTO_HEADER + yaml.dump(config, default_flow_style=False, sort_keys=False)
    )
    return out
