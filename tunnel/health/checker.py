"""Concurrent health polling of registered vLLM instances + optional GPU stats.

Checks run in parallel and return typed dataclasses; format_report() is
presentation-only; collect_gpu_stats() degrades gracefully without pynvml.
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


@dataclass
class GpuStats:
    """Memory stats for a single GPU."""

    index: int
    used_gb: float
    total_gb: float
    utilization_pct: float  # 0.0-1.0

    @property
    def is_near_oom(self) -> bool:
        """True when >90% of VRAM is consumed — actionable OOM warning threshold."""
        return self.utilization_pct > 0.90


def collect_gpu_stats() -> list[GpuStats]:
    """Return memory stats for all visible GPUs.

    Uses pynvml (installed with vLLM). Returns an empty list rather than
    raising if the library is unavailable or NVML fails to initialise —
    callers should treat an empty list as "stats not available".

    Returns:
        List of GpuStats, one per GPU. Empty if pynvml is unavailable.
    """
    try:
        import pynvml  # optional: nvidia-ml-py, pulled in by vllm
        pynvml.nvmlInit()
        stats = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            stats.append(GpuStats(
                index=i,
                used_gb=round(info.used / 1024 ** 3, 2),
                total_gb=round(info.total / 1024 ** 3, 2),
                utilization_pct=round(info.used / info.total, 3),
            ))
        return stats
    except Exception:
        return []


async def _check_one(
    client: httpx.AsyncClient,
    instance_id: str,
    port: int,
    model: str,
    timeout: float,
) -> InstanceHealth:
    url = f"http://localhost:{port}/health"
    t0 = time.monotonic()
    latency_ms = None
    try:
        response = await client.get(url, timeout=timeout)
        latency_ms = round((time.monotonic() - t0) * 1000, 1)
        error = None if response.status_code == 200 else f"HTTP {response.status_code}"
    except httpx.TimeoutException:
        error = "timeout"
    except Exception as exc:
        error = str(exc)

    return InstanceHealth(
        id=instance_id, port=port, model=model,
        status=InstanceStatus.OK if error is None else InstanceStatus.DOWN,
        latency_ms=latency_ms if error is None else None,
        error=error,
    )


async def check_all(
    registry: TunnelRegistry,
    timeout: float = 5.0,
) -> list[InstanceHealth]:
    """Concurrently poll every registered vLLM instance.

    All checks fire in parallel — total time ~ slowest single check.

    Args:
        registry: The loaded TunnelRegistry.
        timeout: Per-instance request timeout in seconds.

    Returns:
        List of InstanceHealth results, one per instance.
    """
    async with httpx.AsyncClient() as client:
        return list(
            await asyncio.gather(
                *[
                    _check_one(client, inst.id, inst.port, inst.model, timeout)
                    for inst in registry.instances
                ]
            )
        )


def format_report(
    results: list[InstanceHealth],
    gpu_stats: list[GpuStats] | None = None,
) -> str:
    """Render a human-readable health report.

    Args:
        results: Per-instance health results from check_all().
        gpu_stats: Optional GPU memory stats from collect_gpu_stats().

    Returns:
        Multi-line string suitable for printing to stdout.
    """
    lines = ["", "-- Tunnel Engine Health --------------------------------------------------"]
    for h in results:
        icon = "[ok]" if h.is_healthy else "[!!]"
        lat = f"{h.latency_ms}ms" if h.latency_ms is not None else "-"
        err = f"  <- {h.error}" if h.error else ""
        lines.append(f"  {icon}  {h.id:<24}  :{h.port:<6}  {lat:<10}{err}")

    up = sum(1 for h in results if h.is_healthy)
    lines.append(f"\n  {up}/{len(results)} instances healthy")

    if gpu_stats:
        lines.append("")
        for gpu in gpu_stats:
            warn = "  ** NEAR OOM **" if gpu.is_near_oom else ""
            lines.append(
                f"  GPU {gpu.index}  {gpu.used_gb}/{gpu.total_gb} GB"
                f"  ({gpu.utilization_pct:.1%}){warn}"
            )
    elif gpu_stats is not None:
        lines.append("  GPU stats unavailable (pynvml not initialised)")

    lines.append("--------------------------------------------------------------------------\n")
    return "\n".join(lines)
