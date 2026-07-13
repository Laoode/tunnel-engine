"""Health-gated startup: poll vLLM instances until all respond healthy.

Prevents LiteLLM's cooldown failure mode: started before vLLM finishes a cold
load (~30-120s), LiteLLM marks the model failed and enters a 60s cooldown, so
the first minute of traffic errors even after vLLM comes up.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
import structlog

from tunnel.orchestrator import is_alive
from tunnel.registry import InstanceConfig, TunnelRegistry

log = structlog.get_logger(__name__)

DEFAULT_TIMEOUT_S: float = 300.0    # 5 min — cold model loads can take 2-3 min
DEFAULT_POLL_INTERVAL_S: float = 5.0


@dataclass
class StartupResult:
    """Result of a wait_for_all() call."""

    ready: bool
    elapsed_s: float
    failed_instances: list[str] = field(default_factory=list)


async def wait_for_instance(
    client: httpx.AsyncClient,
    inst: InstanceConfig,
    timeout_s: float,
    poll_interval_s: float,
    pid: int | None = None,
) -> bool:
    """Poll one instance's /health endpoint until it returns 200 or timeout expires.

    Args:
        client: Shared async HTTP client.
        inst: The instance to poll.
        timeout_s: Max seconds to wait before declaring failure.
        poll_interval_s: Seconds to sleep between poll attempts.
        pid: If given, checked for liveness each poll iteration. When the
            process has died, the wait aborts immediately (returns False)
            instead of polling health until timeout_s expires — otherwise a
            crashed engine blocks the gate for the full timeout even though
            it will never answer /health again.

    Returns:
        True if the instance responded 200 within timeout_s, False otherwise.
    """
    deadline = time.monotonic() + timeout_s
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        if pid is not None and not is_alive(pid):
            log.error("instance_died", instance=inst.id, pid=pid, attempt=attempt)
            return False
        try:
            resp = await client.get(inst.health_url, timeout=3.0)
            if resp.status_code == 200:
                log.info("instance_ready", instance=inst.id, attempt=attempt)
                return True
            log.debug(
                "instance_not_ready",
                instance=inst.id, attempt=attempt, http_status=resp.status_code,
            )
        except Exception as exc:
            log.debug(
                "instance_unreachable",
                instance=inst.id, attempt=attempt, error=str(exc),
            )
        await asyncio.sleep(poll_interval_s)

    log.error(
        "instance_startup_timeout",
        instance=inst.id, timeout_s=timeout_s, attempts=attempt,
    )
    return False


async def wait_for_all(
    registry: TunnelRegistry,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    pids: dict[str, int] | None = None,
) -> StartupResult:
    """Wait for all registered instances to become healthy, polling concurrently.

    Each instance gets its own timeout_s budget. All run concurrently, so total
    wall time is bounded by the slowest instance, not the sum of all timeouts.

    Args:
        registry: The loaded TunnelRegistry.
        timeout_s: Per-instance timeout in seconds.
        poll_interval_s: Seconds between health polls per instance.
        pids: Optional map of instance id -> pid. When an instance's id is
            present, its wait aborts immediately if that pid dies instead of
            polling until timeout_s. Callers with no pids to track (e.g.
            `cmd_start`, which waits on processes it didn't launch) can omit
            this and keep the original poll-until-timeout behavior.

    Returns:
        StartupResult with ready=True only when every instance passed.
    """
    log.info(
        "startup_waiting",
        instance_count=len(registry.instances),
        timeout_s=timeout_s,
    )
    t0 = time.monotonic()
    pids = pids or {}

    async with httpx.AsyncClient() as client:
        outcomes: list[bool] = list(
            await asyncio.gather(
                *[
                    wait_for_instance(
                        client, inst, timeout_s, poll_interval_s,
                        pid=pids.get(inst.id),
                    )
                    for inst in registry.instances
                ]
            )
        )

    failed = [
        inst.id
        for inst, ok in zip(registry.instances, outcomes)
        if not ok
    ]
    elapsed = round(time.monotonic() - t0, 1)

    if failed:
        log.error("startup_failed", failed=failed, elapsed_s=elapsed)
    else:
        log.info("startup_complete", elapsed_s=elapsed)

    return StartupResult(ready=not failed, elapsed_s=elapsed, failed_instances=failed)


async def wait_for_one(
    inst: InstanceConfig,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
    pid: int | None = None,
) -> bool:
    """Wait for a single instance to become healthy, opening its own HTTP client.

    Used by `cmd_up`'s sequential launch mode to health-gate each instance
    before launching the next one, avoiding the concurrent-startup GPU
    memory profiling corruption that `wait_for_all`'s all-at-once polling
    would otherwise race into.

    Args:
        inst: The instance to poll.
        timeout_s: Max seconds to wait before declaring failure.
        poll_interval_s: Seconds to sleep between poll attempts.
        pid: If given, abort immediately once this pid is no longer alive
            instead of polling health until timeout_s.

    Returns:
        True if the instance responded 200 within timeout_s, False otherwise.
    """
    async with httpx.AsyncClient() as client:
        return await wait_for_instance(client, inst, timeout_s, poll_interval_s, pid=pid)
