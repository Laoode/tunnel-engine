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

    litellm_settings: dict = {
        "request_timeout": 120,
        "drop_params": True,
        "require_auth_for_metrics_endpoint": False,
    }

    # fallbacks: [{primary_id: [fallback_id, ...]}] — only for instances that define one
    fallbacks = [
        {inst.id: inst.fallbacks}
        for inst in registry.instances
        if inst.fallbacks
    ]
    if fallbacks:
        router_settings["fallbacks"] = fallbacks

    if registry.litellm.prometheus:
        # Enables /metrics on the proxy port. Requires: pip install prometheus-client
        litellm_settings["callbacks"] = ["prometheus"]

    return {
        "model_list": model_list,
        "router_settings": router_settings,
        "general_settings": {
            "master_key": registry.litellm.master_key,
        },
        "litellm_settings": litellm_settings,

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