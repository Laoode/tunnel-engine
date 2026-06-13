"""
tunnel/cache/lmcache_config.py
================================
Generates per-instance LMCache config files under configs/lmcache/.

LMCache activates by LMCACHE_CONFIG_FILE=<path> env var before vLLM.
The CLI (tunnel/cli.py) sets this automatically.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from tunnel.registry import InstanceConfig, TunnelRegistry

_AUTO_HEADER_TPL = (
    "# AUTO-GENERATED for instance: {instance_id}\n"
    "# Source : configs/models.yaml  |  Regenerate: make generate\n\n"
)


def build_lmcache_config(inst: InstanceConfig) -> dict:
    """Pure function: InstanceConfig → LMCache config dict."""
    cfg = inst.lmcache
    return {
        "chunk_size": cfg.chunk_size,
        "local_cpu": {
            "enabled": cfg.backend == "cpu",
            "max_cache_size": cfg.max_cache_size_gb,
        },
        "local_disk": {
            "enabled": cfg.backend == "disk",
            "path": f"/tmp/lmcache/{inst.id}",
            "max_cache_size": cfg.max_cache_size_gb,
        },
    }


def write_lmcache_configs(
    registry: TunnelRegistry,
    output_dir: Path | str = "configs/lmcache",
) -> list[Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for inst in registry.instances:
        if not inst.lmcache.enabled:
            continue
        config = build_lmcache_config(inst)
        path = out_dir / f"{inst.id}.yaml"
        path.write_text(
            _AUTO_HEADER_TPL.format(instance_id=inst.id)
            + yaml.dump(config, default_flow_style=False)
        )
        written.append(path)

    return written
