"""
tests/unit/test_startup.py
===========================
Tests for tunnel/startup.py.

wait_for_instance() is tested by mocking httpx.AsyncClient.get directly.
wait_for_all() is tested by patching wait_for_instance — the layer below it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tunnel.registry import TunnelRegistry
from tunnel.startup import StartupResult, wait_for_all, wait_for_instance


def _make_registry(n: int = 2) -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {
                "id": f"m{i}",
                "model": f"org/m{i}",
                "port": 8000 + i,
                "gpu_memory_utilization": 0.1,
            }
            for i in range(n)
        ]
    })


def _mock_response(status_code: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    return resp


@pytest.mark.asyncio
async def test_wait_for_instance_returns_true_on_200() -> None:
    registry = _make_registry(1)
    inst = registry.instances[0]

    mock_get = AsyncMock(return_value=_mock_response(200))
    async with httpx.AsyncClient() as client:
        with patch.object(client, "get", mock_get):
            ok = await wait_for_instance(client, inst, timeout_s=5.0, poll_interval_s=0.01)

    assert ok is True
    mock_get.assert_called_once_with(inst.health_url, timeout=3.0)


@pytest.mark.asyncio
async def test_wait_for_instance_returns_false_on_non_200() -> None:
    registry = _make_registry(1)
    inst = registry.instances[0]

    mock_get = AsyncMock(return_value=_mock_response(503))
    async with httpx.AsyncClient() as client:
        with patch.object(client, "get", mock_get):
            # timeout_s < poll_interval_s so loop exits after one attempt
            ok = await wait_for_instance(client, inst, timeout_s=0.02, poll_interval_s=0.03)

    assert ok is False


@pytest.mark.asyncio
async def test_wait_for_all_returns_ready_when_all_healthy() -> None:
    registry = _make_registry(2)

    async def _always_ready(client, inst, timeout_s, poll_interval_s) -> bool:
        return True

    with patch("tunnel.startup.wait_for_instance", side_effect=_always_ready):
        result = await wait_for_all(registry, timeout_s=5.0, poll_interval_s=0.01)

    assert result.ready is True
    assert result.failed_instances == []


@pytest.mark.asyncio
async def test_wait_for_all_returns_not_ready_when_one_fails() -> None:
    registry = _make_registry(2)
    failing_id = registry.instances[1].id

    async def _selective(client, inst, timeout_s, poll_interval_s) -> bool:
        return inst.id != failing_id

    with patch("tunnel.startup.wait_for_instance", side_effect=_selective):
        result = await wait_for_all(registry, timeout_s=5.0, poll_interval_s=0.01)

    assert result.ready is False
    assert failing_id in result.failed_instances
    assert len(result.failed_instances) == 1


@pytest.mark.asyncio
async def test_wait_for_all_all_fail() -> None:
    registry = _make_registry(3)

    async def _always_fail(client, inst, timeout_s, poll_interval_s) -> bool:
        return False

    with patch("tunnel.startup.wait_for_instance", side_effect=_always_fail):
        result = await wait_for_all(registry, timeout_s=1.0, poll_interval_s=0.01)

    assert result.ready is False
    assert len(result.failed_instances) == len(registry.instances)


@pytest.mark.asyncio
async def test_wait_for_all_failed_instances_are_correct_ids() -> None:
    registry = _make_registry(3)
    failing = {registry.instances[0].id, registry.instances[2].id}

    async def _selective(client, inst, timeout_s, poll_interval_s) -> bool:
        return inst.id not in failing

    with patch("tunnel.startup.wait_for_instance", side_effect=_selective):
        result = await wait_for_all(registry, timeout_s=5.0, poll_interval_s=0.01)

    assert set(result.failed_instances) == failing


def test_startup_result_ready_flag() -> None:
    ok = StartupResult(ready=True, elapsed_s=4.2)
    assert ok.ready is True
    assert ok.failed_instances == []

    fail = StartupResult(ready=False, elapsed_s=300.0, failed_instances=["m1", "m2"])
    assert fail.ready is False
    assert "m1" in fail.failed_instances
