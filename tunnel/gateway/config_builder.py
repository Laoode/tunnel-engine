"""Generates configs/litellm/config.yaml from the TunnelRegistry.

build_litellm_config() is a pure registry -> dict function;
write_litellm_config() is the I/O wrapper.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tunnel.registry import ModelCost, TunnelRegistry

_AUTO_HEADER = (
    "# AUTO-GENERATED — do not edit manually.\n"
    "# Source of truth : configs/models.yaml\n"
    "# Regenerate      : make generate\n\n"
)


def _cost_params(cost: ModelCost | None) -> dict:
    """Return LiteLLM per-token cost params for a ModelCost, or {} when unset.

    Args:
        cost: Registry cost block (USD per 1M tokens), or None.

    Returns:
        Dict with input/output_cost_per_token, empty when cost is None.
    """
    if cost is None:
        return {}
    return {
        "input_cost_per_token": cost.input_per_mtok / 1_000_000,
        "output_cost_per_token": cost.output_per_mtok / 1_000_000,
    }


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
                "model": f"openai/{inst.served_model_name or inst.model}",
                "api_base": inst.api_base,
                "api_key": "none",
                **_cost_params(inst.cost),
            },
        }
        for inst in registry.instances
    ]

    # Remote OpenAI-compatible upstreams (e.g. DeepSeek). The key is stored as an
    # os.environ/ reference so no secret is written into the generated config.
    model_list += [
        {
            "model_name": rm.id,
            "litellm_params": {
                "model": f"{rm.provider}/{rm.upstream_model}",
                "api_base": rm.api_base,
                "api_key": f"os.environ/{rm.api_key_env}",
                **_cost_params(rm.cost),
            },
        }
        for rm in registry.remote_models
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

    general_settings: dict = {
        "master_key": registry.litellm.master_key,
    }
    if registry.services:
        # Virtual keys / spend logs need LiteLLM's Postgres layer. Emitted only
        # when services are declared so the no-DB dev path keeps working.
        general_settings["database_url"] = "os.environ/DATABASE_URL"
        general_settings["store_model_in_db"] = False

    return {
        "model_list": model_list,
        "router_settings": router_settings,
        "general_settings": general_settings,
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