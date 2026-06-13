import tempfile
import yaml
from tunnel.registry import TunnelRegistry
from tunnel.cache.lmcache_config import build_lmcache_config, write_lmcache_configs


def _reg(instances):
    return TunnelRegistry.model_validate({"instances": instances})


def test_cpu_backend_only_cpu_enabled():
    reg = _reg([{"id":"m","model":"x","port":8000,"gpu_memory_utilization":0.4,
                 "lmcache":{"enabled":True,"backend":"cpu","max_cache_size_gb":10,"chunk_size":256}}])
    cfg = build_lmcache_config(reg.instances[0])
    assert cfg["local_cpu"]["enabled"] is True
    assert cfg["local_disk"]["enabled"] is False

def test_disk_path_includes_instance_id():
    reg = _reg([{"id":"my-model","model":"x","port":8000,"gpu_memory_utilization":0.4,
                 "lmcache":{"enabled":True,"backend":"disk","max_cache_size_gb":10,"chunk_size":256}}])
    cfg = build_lmcache_config(reg.instances[0])
    assert "my-model" in cfg["local_disk"]["path"]

def test_write_skips_disabled_instances():
    reg = _reg([
        {"id":"on","model":"x","port":8000,"gpu_memory_utilization":0.4,
         "lmcache":{"enabled":True,"backend":"cpu","max_cache_size_gb":10,"chunk_size":256}},
        {"id":"off","model":"y","port":8001,"gpu_memory_utilization":0.4,
         "lmcache":{"enabled":False,"backend":"cpu","max_cache_size_gb":10,"chunk_size":256}},
    ])
    with tempfile.TemporaryDirectory() as d:
        written = write_lmcache_configs(reg, d)
        assert len(written) == 1
        assert written[0].stem == "on"
