import tempfile

import pytest

from tunnel.registry import TunnelRegistry
from tunnel.cache.lmcache_config import build_lmcache_config, write_lmcache_configs


def _reg(instances):
    return TunnelRegistry.model_validate({"instances": instances})


def _inst(backend, **lmcache):
    base = {"enabled": True, "backend": backend, "max_cache_size_gb": 10, "chunk_size": 256}
    base.update(lmcache)
    return {"id": "m", "model": "x", "port": 8000, "gpu_memory_utilization": 0.4, "lmcache": base}


def test_cpu_backend_enables_local_cpu_flat_schema():
    cfg = build_lmcache_config(_reg([_inst("cpu")]).instances[0])
    assert cfg["local_cpu"] is True                 # flat bool, not nested dict
    assert cfg["max_local_cpu_size"] == 10.0        # GB as float
    assert "local_disk" not in cfg
    assert "remote_serde" not in cfg


def test_disk_backend_sets_path_and_disables_cpu():
    inst = _reg([{**_inst("disk"), "id": "my-model"}]).instances[0]
    cfg = build_lmcache_config(inst)
    assert cfg["local_cpu"] is False
    assert cfg["local_disk"] == "/tmp/lmcache/my-model"
    assert cfg["max_local_disk_size"] == 10.0


def test_redis_backend_keeps_cpu_tier_and_sets_serde():
    cfg = build_lmcache_config(_reg([_inst("redis", remote_serde="naive")]).instances[0])
    assert cfg["local_cpu"] is True                 # L1 CPU tier stays on
    assert cfg["remote_serde"] == "naive"
    # remote_url is injected at serve time from env, never baked into the file.
    assert "remote_url" not in cfg


def test_write_skips_disabled_instances():
    reg = _reg([
        {**_inst("cpu"), "id": "on", "port": 8000},
        {**_inst("cpu"), "id": "off", "port": 8001, "lmcache": {"enabled": False, "backend": "cpu"}},
    ])
    with tempfile.TemporaryDirectory() as d:
        written = write_lmcache_configs(reg, d)
        assert len(written) == 1
        assert written[0].stem == "on"


@pytest.mark.parametrize("backend", ["cpu", "disk", "redis"])
def test_generated_config_is_accepted_by_lmcache(backend):
    """Guard against schema drift: LMCache 0.4.x must parse what we emit.

    The previous nested schema silently raised TypeError inside LMCache, so KV
    caching never actually ran. This asserts the flat schema loads cleanly.
    """
    lmcache_config = pytest.importorskip("lmcache.v1.config")
    import yaml

    cfg = build_lmcache_config(_reg([_inst(backend)]).instances[0])
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg, f)
        path = f.name
    loaded = lmcache_config.LMCacheEngineConfig.from_file(path)
    assert loaded.chunk_size == 256
