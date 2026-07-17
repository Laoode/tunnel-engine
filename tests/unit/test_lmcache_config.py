"""Tests for build_lmcache_server_command (LMCache MP server argv)."""
import json

from tunnel.registry import TunnelRegistry
from tunnel.cache.lmcache_config import build_lmcache_server_command


def _reg(instances):
    return TunnelRegistry.model_validate({"instances": instances})


def _inst(backend, **lmcache):
    base = {"enabled": True, "backend": backend, "max_cache_size_gb": 10, "chunk_size": 256}
    base.update(lmcache)
    return {"id": "m", "model": "x", "port": 8000, "gpu_memory_utilization": 0.4, "lmcache": base}


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def test_cpu_backend_l1_only():
    cmd = build_lmcache_server_command(_reg([_inst("cpu")]).instances[0], env={})
    assert cmd[:2] == ["lmcache", "server"]
    assert _flag(cmd, "--port") == "9000"           # instance port + 1000 default
    assert _flag(cmd, "--http-port") == "10000"     # zmq port + 1000
    assert _flag(cmd, "--l1-size-gb") == "10"
    assert _flag(cmd, "--chunk-size") == "256"
    assert _flag(cmd, "--eviction-policy") == "LRU"
    assert "--l2-adapter" not in cmd


def test_explicit_port_wins_over_derived_default():
    cmd = build_lmcache_server_command(
        _reg([_inst("cpu", port=6555)]).instances[0], env={}
    )
    assert _flag(cmd, "--port") == "6555"
    assert _flag(cmd, "--http-port") == "7555"


def test_disk_backend_adds_fs_l2_adapter():
    inst = _reg([{**_inst("disk"), "id": "my-model"}]).instances[0]
    cmd = build_lmcache_server_command(inst, env={})
    adapter = json.loads(_flag(cmd, "--l2-adapter"))
    assert adapter == {"type": "fs", "path": "/tmp/lmcache/my-model"}


def test_redis_backend_adds_resp_l2_adapter_from_env():
    cmd = build_lmcache_server_command(
        _reg([_inst("redis")]).instances[0],
        env={"LMCACHE_REDIS_HOST": "cache.internal", "LMCACHE_REDIS_PORT": "6380"},
    )
    adapter = json.loads(_flag(cmd, "--l2-adapter"))
    assert adapter == {"type": "resp", "host": "cache.internal", "port": 6380}


def test_redis_backend_degrades_to_l1_without_host():
    # Host/port are environment-specific and never committed; without them the
    # server still starts with the local L1 tier only (caller warns).
    cmd = build_lmcache_server_command(_reg([_inst("redis")]).instances[0], env={})
    assert "--l2-adapter" not in cmd


def test_custom_eviction_policy_passthrough():
    cmd = build_lmcache_server_command(
        _reg([_inst("cpu", eviction_policy="IsolatedLRU")]).instances[0], env={}
    )
    assert _flag(cmd, "--eviction-policy") == "IsolatedLRU"
