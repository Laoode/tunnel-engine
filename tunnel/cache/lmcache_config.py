"""Builds the `lmcache server` command for an instance (MP architecture).

Each instance with lmcache.enabled runs its own LMCache server process (ZMQ);
vLLM reaches it through the LMCacheMPConnector with lmcache.mp.port pointing
at inst.lmcache.port. The server owns all cache tiers: L1 is CPU RAM, and the
"disk" / "redis" backends add an L2 adapter. See docs/lmcache.md.
"""
from __future__ import annotations

import json
import os

from tunnel.registry import InstanceConfig

LMCACHE_DISK_ROOT = "/tmp/lmcache"


def build_lmcache_server_command(
    inst: InstanceConfig, env: dict | None = None
) -> list[str]:
    """Pure function: InstanceConfig -> `lmcache server` argv.

    Backend tiers:
      - cpu:   L1 CPU RAM only.
      - disk:  L1 + filesystem L2 adapter under /tmp/lmcache/<id>.
      - redis: L1 + RESP L2 adapter. Host/port come from LMCACHE_REDIS_HOST /
               LMCACHE_REDIS_PORT in `env` (environment-specific, never
               committed); silently degrades to L1-only when the host is
               unset — the caller warns.

    Args:
        inst: Validated InstanceConfig with lmcache.enabled.
        env: Environment mapping for redis host/port lookup. Defaults to
            os.environ.

    Returns:
        Full argv list for the LMCache server process.
    """
    if env is None:
        env = dict(os.environ)
    cfg = inst.lmcache
    cmd = [
        "lmcache", "server",
        "--instance-id", inst.id,
        "--port",        str(cfg.port),
        # The HTTP frontend always binds; the default :8080 would collide
        # across servers, so it gets a derived unique port on localhost.
        "--http-host",   "127.0.0.1",
        "--http-port",   str(cfg.port + 1000),
        "--chunk-size",  str(cfg.chunk_size),
        "--l1-size-gb",  str(cfg.max_cache_size_gb),
        "--eviction-policy", cfg.eviction_policy,
    ]
    if cfg.backend == "disk":
        cmd += ["--l2-adapter",
                json.dumps({"type": "fs", "path": f"{LMCACHE_DISK_ROOT}/{inst.id}"})]
    if cfg.backend == "redis":
        host = env.get("LMCACHE_REDIS_HOST")
        if host:
            port = int(env.get("LMCACHE_REDIS_PORT", "6379"))
            cmd += ["--l2-adapter",
                    json.dumps({"type": "resp", "host": host, "port": port})]
    return cmd
