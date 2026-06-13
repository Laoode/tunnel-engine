"""
tunnel/health/checker.py
=========================
Concurrently polls all registered vLLM instances.

All checks fire in parallel (asyncio.gather) — N instances ≈ same wall time as 1.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx

from tunnel.registry import TunnelRegistry


class InstanceStatus(str, Enum):
    OK = "ok"
    DOWN = "down"


@dataclass
class InstanceHealth:
    id: str
    port: int
    model: str
    status: InstanceStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None

    @property
    def is_healthy(self) -> bool:
        return self.status == InstanceStatus.OK


async def _check_one(
    client: httpx.AsyncClient,
    instance_id: str,
    port: int,
    model: str,
    timeout: float,
) -> InstanceHealth:
    url = f"http://localhost:{port}/health"
    t0 = time.monotonic()
    try:
        response = await client.get(url, timeout=timeout)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        if response.status_code == 200:
            return InstanceHealth(
                id=instance_id, port=port, model=model,
                status=InstanceStatus.OK, latency_ms=latency_ms,
            )
        return InstanceHealth(
            id=instance_id, port=port, model=model,
            status=InstanceStatus.DOWN, error=f"HTTP {response.status_code}",
        )
    except httpx.TimeoutException:
        return InstanceHealth(
            id=instance_id, port=port, model=model,
            status=InstanceStatus.DOWN, error="timeout",
        )
    except Exception as exc:
        return InstanceHealth(
            id=instance_id, port=port, model=model,
            status=InstanceStatus.DOWN, error=str(exc),
        )


async def check_all(
    registry: TunnelRegistry,
    timeout: float = 5.0,
) -> list[InstanceHealth]:
    """Concurrently poll every registered vLLM instance."""
    async with httpx.AsyncClient() as client:
        return list(
            await asyncio.gather(
                *[
                    _check_one(client, inst.id, inst.port, inst.model, timeout)
                    for inst in registry.instances
                ]
            )
        )


def format_report(results: list[InstanceHealth]) -> str:
    lines = ["", "── Tunnel Engine Health ──────────────────────────────────"]
    for h in results:
        icon = "✓" if h.is_healthy else "✗"
        lat = f"{h.latency_ms}ms" if h.latency_ms is not None else "—"
        err = f"  ← {h.error}" if h.error else ""
        lines.append(
            f"  {icon}  {h.id:<24}  :{h.port:<6}  {lat:<10}{err}"
        )
    up = sum(1 for h in results if h.is_healthy)
    lines.append(f"\n  {up}/{len(results)} instances healthy")
    lines.append("────────────────────────────────────────────────────────────\n")
    return "\n".join(lines)
