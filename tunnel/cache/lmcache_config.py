"""Generates per-instance LMCache config files under configs/lmcache/.

LMCache activates via the LMCACHE_CONFIG_FILE env var, set by tunnel/cli.py.
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
    """Pure function: InstanceConfig -> LMCache config dict.

    Emits LMCache's flat config schema (v0.4.x): local_cpu is a bool with a
    sibling max_local_cpu_size (GB), local_disk is a path string with a sibling
    max_local_disk_size, and remote_url/remote_serde drive the remote backend.

    Backend tiers:
      - cpu:   CPU RAM only.
      - disk:  local disk only.
      - redis: CPU RAM (L1) + Redis (L2). The remote_url is injected at serve
               time from the environment (LMCACHE_REMOTE_URL), since the Redis
               host/port are environment-specific and must never be committed.
    """
    cfg = inst.lmcache
    size_gb = float(cfg.max_cache_size_gb)
    config: dict = {
        "chunk_size": cfg.chunk_size,
        "local_cpu": cfg.backend in ("cpu", "redis"),
        "max_local_cpu_size": size_gb,
    }
    if cfg.backend == "disk":
        config["local_disk"] = f"/tmp/lmcache/{inst.id}"
        config["max_local_disk_size"] = size_gb
    if cfg.backend == "redis":
        config["remote_serde"] = cfg.remote_serde
        # remote_url is set via LMCACHE_REMOTE_URL at serve time (see tunnel/cli.py).
    return config


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
