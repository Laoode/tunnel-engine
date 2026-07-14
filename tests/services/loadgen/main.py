"""Open-loop load generator against the Tunnel Engine gateway (not pytest).

Poisson arrivals at RATE req/s for DURATION seconds, traffic drawn from the
LISO/LILO/SILO/SISO mixes, per-request streaming timings recorded to
results/results.jsonl (one JSON object per request: mix, tier, isl, osl,
ttft_s, itl_s series, total_s, status).

  make loadtest                                # defaults below
  RATE=2 DURATION=120 MIX=LILO make loadtest   # single mix
  TIER_MIX="tools-bench:4,development:2,klaudia:1" make loadtest

Env vars:
  MODEL     target gateway model id (default: first local instance)
  RATE      mean arrivals per second (default 1.0)
  DURATION  seconds to keep launching (default 60)
  MIX       one mix name, or "all" round-robin (default all)
  TIER_MIX  weighted service:weight list; keys read from .tunnel/keys.env.
            Unset = master key only (no tier limits, pure engine behavior).
  SEED      workload RNG seed (default 7)

Run plots afterwards: make loadtest-plots.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from workload import MIXES, make_request  # noqa: E402
from tunnel.cli import _load_env  # noqa: E402
from tunnel.gateway.keys import key_env_name, read_secrets  # noqa: E402
from tunnel.registry import load_registry  # noqa: E402

_load_env()  # resolve LITELLM_MASTER_KEY the same way the CLI does

RESULTS_PATH = Path(__file__).parent / "results" / "results.jsonl"


def _tier_keys(registry, tier_mix: str) -> list[tuple[str, str, str]]:
    """Resolve TIER_MIX into weighted (service, tier, key) choices.

    Args:
        registry: Loaded registry (maps service -> tier).
        tier_mix: "service:weight,service:weight" string.

    Returns:
        A list with each (service, tier, key) repeated `weight` times,
        suitable for random.choice.
    """
    secrets = read_secrets()
    tiers = {s.id: s.tier for s in registry.services}
    weighted: list[tuple[str, str, str]] = []
    for part in tier_mix.split(","):
        service, _, weight = part.strip().partition(":")
        key = secrets.get(key_env_name(service))
        if key is None:
            sys.exit(f"ERROR: no key for service '{service}' in .tunnel/keys.env "
                     "(run `make keys-sync`)")
        weighted += [(service, tiers[service], key)] * int(weight or 1)
    return weighted


async def _one(client: httpx.AsyncClient, url: str, key: str, model: str,
               req, tier: str, results: list) -> None:
    """Send one streaming request and record its timings."""
    row = {"mix": req.mix, "tier": tier, "model": model, "status": None,
           "isl": None, "osl": None, "ttft_s": None, "itl_s": [], "total_s": None}
    t0 = time.perf_counter()
    last = t0
    try:
        async with client.stream(
            "POST", url,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": req.prompt}],
                  "max_tokens": req.max_tokens, "stream": True,
                  "stream_options": {"include_usage": True}},
        ) as resp:
            row["status"] = resp.status_code
            if resp.status_code != 200:
                results.append(row)
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                chunk = json.loads(line[6:])
                usage = chunk.get("usage")
                if usage:
                    row["isl"] = usage.get("prompt_tokens")
                    row["osl"] = usage.get("completion_tokens")
                    continue
                if row["ttft_s"] is None:
                    row["ttft_s"] = now - t0
                else:
                    row["itl_s"].append(now - last)
                last = now
        row["total_s"] = time.perf_counter() - t0
    except httpx.HTTPError as exc:
        row["status"] = f"error: {exc.__class__.__name__}"
    results.append(row)


async def main() -> None:
    registry = load_registry()
    model = os.environ.get("MODEL", registry.instances[0].id)
    rate = float(os.environ.get("RATE", "1.0"))
    duration = float(os.environ.get("DURATION", "60"))
    mix_env = os.environ.get("MIX", "all")
    mixes = list(MIXES) if mix_env == "all" else [mix_env]
    rng = random.Random(int(os.environ.get("SEED", "7")))
    salt = f"{time.time():.0f}"
    url = f"http://localhost:{registry.litellm.port}/v1/chat/completions"

    tier_mix = os.environ.get("TIER_MIX")
    if tier_mix:
        callers = _tier_keys(registry, tier_mix)
    else:
        master = registry.litellm.resolved_master_key
        if not master:
            sys.exit("ERROR: LITELLM_MASTER_KEY unset and no TIER_MIX given")
        callers = [("master", "none", master)]

    results: list[dict] = []
    tasks: list[asyncio.Task] = []
    print(f"loadgen: model={model} rate={rate}/s duration={duration}s "
          f"mixes={mixes} callers={sorted({c[0] for c in callers})}",
          file=sys.stderr)

    async with httpx.AsyncClient(timeout=600) as client:
        t_end = time.monotonic() + duration
        i = 0
        while time.monotonic() < t_end:
            service, tier, key = rng.choice(callers)
            req = make_request(mixes[i % len(mixes)], rng, salt)
            tasks.append(asyncio.create_task(
                _one(client, url, key, model, req, tier, results)))
            i += 1
            await asyncio.sleep(rng.expovariate(rate))
        print(f"loadgen: {i} requests launched, draining...", file=sys.stderr)
        await asyncio.gather(*tasks)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as fh:
        for row in results:
            fh.write(json.dumps(row) + "\n")

    ok = sum(1 for r in results if r["status"] == 200)
    limited = sum(1 for r in results if r["status"] == 429)
    print(f"loadgen: done. {ok} ok, {limited} rate-limited, "
          f"{len(results) - ok - limited} other -> {RESULTS_PATH}", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
