"""Generates configs/prometheus/prometheus.yml from the TunnelRegistry.

Each local vLLM instance exposes native metrics on /metrics (TTFT, TPOT,
queue depth, KV-cache usage, preemptions); this emits one static scrape
target per instance. build_prometheus_config() is a pure registry -> dict
function; write_prometheus_config() is the I/O wrapper.
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

SCRAPE_INTERVAL_S = 5  # vLLM latency histograms move fast under load


def build_prometheus_config(registry: TunnelRegistry) -> dict:
    """Build the Prometheus scrape config dict from the registry.

    Args:
        registry: Validated TunnelRegistry.

    Returns:
        Dict matching the prometheus.yml schema: one `vllm` job with a
        static target per local instance, labeled by instance id and model.
    """
    targets = [
        {
            "targets": [f"localhost:{inst.port}"],
            "labels": {"instance_id": inst.id, "model": inst.model},
        }
        for inst in registry.instances
    ]
    return {
        "global": {"scrape_interval": f"{SCRAPE_INTERVAL_S}s"},
        "scrape_configs": [
            {
                "job_name": "vllm",
                "static_configs": targets,
            }
        ],
    }


def write_prometheus_config(
    registry: TunnelRegistry,
    output_path: Path | str = "configs/prometheus/prometheus.yml",
) -> Path:
    """Write the generated Prometheus config to disk.

    Args:
        registry: Validated TunnelRegistry.
        output_path: Destination path. Parent dirs are created if needed.

    Returns:
        Path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = build_prometheus_config(registry)
    out.write_text(
        _AUTO_HEADER + yaml.dump(config, default_flow_style=False, sort_keys=False)
    )
    return out
