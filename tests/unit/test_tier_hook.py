"""
Tests for tunnel/gateway/tier_hook.py: priority injected for local
priority-scheduled models only, tier metadata honored, keyless fallback.
"""
import asyncio
from types import SimpleNamespace

from tunnel.gateway.tier_hook import TierPriorityHook
from tunnel.registry import TunnelRegistry


def _registry() -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "prio-model", "model": "org/a", "port": 8000,
             "gpu_memory_utilization": 0.10, "scheduling_policy": "priority"},
            {"id": "fcfs-model", "model": "org/b", "port": 8001,
             "gpu_memory_utilization": 0.10, "scheduling_policy": "fcfs"},
        ],
        "remote_models": [
            {"id": "remote-x", "upstream_model": "x",
             "api_base": "https://api.example.com", "api_key_env": "X_KEY"},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test"},
        "tiers": {
            "enterprise": {"priority": 0, "rpm_limit": 100, "tpm_limit": 1000},
            "free": {"priority": 2, "rpm_limit": 10, "tpm_limit": 100},
        },
    })


def _run_hook(hook: TierPriorityHook, model: str, metadata: dict | None) -> dict:
    """Invoke async_pre_call_hook synchronously and return the mutated data.

    Args:
        hook: The hook under test.
        model: Requested gateway model name.
        metadata: Key metadata dict, or None for a keyless request.

    Returns:
        The request payload after the hook ran.
    """
    key = SimpleNamespace(metadata=metadata)
    data = {"model": model, "messages": []}
    return asyncio.run(hook.async_pre_call_hook(key, None, data, "completion"))


def test_priority_injected_from_key_metadata():
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "prio-model", {"priority": 0, "tier": "enterprise"})
    assert data["priority"] == 0


def test_priority_zero_not_treated_as_missing():
    # priority 0 (enterprise, best) is falsy; the hook must not fall back.
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "prio-model", {"priority": 0})
    assert data["priority"] == 0


def test_keyless_request_gets_worst_tier_priority():
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "prio-model", None)
    assert data["priority"] == 2


def test_fcfs_model_not_touched():
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "fcfs-model", {"priority": 0})
    assert "priority" not in data


def test_remote_model_not_touched():
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "remote-x", {"priority": 0})
    assert "priority" not in data


def test_unknown_model_not_touched():
    hook = TierPriorityHook(registry=_registry())
    data = _run_hook(hook, "not-registered", {"priority": 0})
    assert "priority" not in data


def test_garbage_priority_metadata_falls_back_to_default():
    hook = TierPriorityHook(registry=_registry())
    for bad in (None, "not-a-number"):
        data = _run_hook(hook, "prio-model", {"priority": bad})
        assert data["priority"] == 2
