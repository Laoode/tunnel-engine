"""Guardrail benchmark: XGuard latency + accuracy on the judged dataset.

Sends every kept sample to the internal guard instance exactly as the gateway
guard hook does (direct vLLM call, max_tokens=1, first-token logprobs) and
records per-request latency and the predicted verdict. Reports:
  - latency distribution (p50/p90/p99, mean) of the guard check itself,
  - confusion matrix vs the judged labels (safe/unsafe) at the config threshold,
  - per-category recall on unsafe samples.

This is the production hot-path cost: the number the gateway pays per request
when guardrails are on. Runs the guard model directly, so it needs the fleet
up (`make up`) but not the proxy.

  make guard-bench                    # reads dataset.judged.yaml
  CONCURRENCY=8 make guard-bench      # simulate load; latency reflects queueing

Env vars:
  CONCURRENCY   in-flight guard requests (default 1 = pure per-call latency)
  THRESHOLD     block cutoff override (default: registry guardrails.threshold)
  WARMUP        warmup requests before timing (default 3)
"""
from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tunnel.cli import _load_env  # noqa: E402
from tunnel.gateway.guard_hook import SAFE_LABEL, risk_scores_from_response  # noqa: E402
from tunnel.registry import load_registry  # noqa: E402

_load_env()

HERE = Path(__file__).parent
JUDGED_PATH = HERE / "results" / "dataset.judged.yaml"
REPORT_PATH = HERE / "RESULTS.md"


def _percentile(values: list[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of values via nearest-rank."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * len(ordered)) - 1))
    return ordered[rank]


async def _check(client: httpx.AsyncClient, url: str, model: str,
                 prompt: str) -> tuple[float, dict[str, float]]:
    """Run one guard classification; returns (latency_s, risk_scores)."""
    t0 = time.perf_counter()
    resp = await client.post(url, json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10,
    })
    resp.raise_for_status()
    latency = time.perf_counter() - t0
    return latency, risk_scores_from_response(resp.json())


def _predict(scores: dict[str, float], threshold: float) -> tuple[bool, str, float]:
    """Return (is_blocked, top_category, top_score) at the given threshold."""
    risky = {c: p for c, p in scores.items() if c != SAFE_LABEL}
    if not risky:
        return False, SAFE_LABEL, scores.get(SAFE_LABEL, 0.0)
    top_cat, top_score = max(risky.items(), key=lambda kv: kv[1])
    return top_score >= threshold, top_cat, top_score


async def _run(samples: list[dict], url: str, model: str, threshold: float,
               concurrency: int, warmup: int) -> list[dict]:
    """Benchmark all samples with bounded concurrency; returns per-sample rows."""
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=30.0) as client:
        for sample in samples[:warmup]:  # warm CUDA graphs / caches, untimed
            await _check(client, url, model, sample["prompt"])

        async def _one(sample: dict) -> dict:
            async with sem:
                latency, scores = await _check(client, url, model, sample["prompt"])
            blocked, category, score = _predict(scores, threshold)
            return {
                "id": sample["id"], "expected": sample["expected"],
                "true_category": sample["category"], "latency_s": latency,
                "blocked": blocked, "pred_category": category, "pred_score": score,
            }

        return await asyncio.gather(*[_one(s) for s in samples])


def _confusion(rows: list[dict]) -> dict[str, int]:
    """Tally TP/FP/TN/FN treating 'blocked' as the positive (unsafe) prediction."""
    tally = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        unsafe = row["expected"] == "unsafe"
        if row["blocked"] and unsafe:
            tally["tp"] += 1
        elif row["blocked"] and not unsafe:
            tally["fp"] += 1
        elif not row["blocked"] and not unsafe:
            tally["tn"] += 1
        else:
            tally["fn"] += 1
    return tally


def _render_report(rows: list[dict], latencies: list[float], threshold: float,
                   model: str, concurrency: int) -> str:
    """Build the RESULTS.md content from the benchmark rows."""
    c = _confusion(rows)
    total = len(rows)
    accuracy = (c["tp"] + c["tn"]) / total if total else 0.0
    precision = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    fpr = c["fp"] / (c["fp"] + c["tn"]) if (c["fp"] + c["tn"]) else 0.0

    ms = [v * 1000 for v in latencies]
    lat_lines = (
        f"- mean {statistics.mean(ms):.1f} ms | p50 {_percentile(ms, 50):.1f} | "
        f"p90 {_percentile(ms, 90):.1f} | p99 {_percentile(ms, 99):.1f} | "
        f"max {max(ms):.1f}"
    )

    per_cat: dict[str, list[int]] = {}
    for row in rows:
        if row["expected"] == "unsafe":
            hit = per_cat.setdefault(row["true_category"], [0, 0])
            hit[1] += 1
            hit[0] += int(row["blocked"])
    cat_lines = "\n".join(
        f"| {cat} | {hit}/{n} | {hit / n:.0%} |"
        for cat, (hit, n) in sorted(per_cat.items(), key=lambda kv: kv[1][0] / kv[1][1])
    )

    return f"""# Guardrail Benchmark — XGuard 0.6B

Model: `{model}` · threshold {threshold} · concurrency {concurrency} · {total} samples
(dataset: DeepSeek v4 Pro generated, Sonnet 5 judged; Indonesian).

## Latency (guard check, the per-request hot-path cost)
{lat_lines}

## Accuracy (blocked = predicted unsafe)
| Metric | Value |
|--------|-------|
| Accuracy | {accuracy:.1%} |
| Precision | {precision:.1%} |
| Recall (unsafe caught) | {recall:.1%} |
| F1 | {f1:.3f} |
| False-positive rate (safe blocked) | {fpr:.1%} |

Confusion: TP {c['tp']} · FP {c['fp']} · TN {c['tn']} · FN {c['fn']}

## Per-category recall (unsafe)
| Category | Caught | Recall |
|----------|--------|--------|
{cat_lines}
"""


def main() -> None:
    """Run the benchmark and write RESULTS.md."""
    if not JUDGED_PATH.exists():
        sys.exit(f"ERROR: {JUDGED_PATH} not found. Run `make guard-judge` first.")
    data = yaml.safe_load(JUDGED_PATH.read_text())
    samples = data["kept"]
    if not samples:
        sys.exit("ERROR: no kept samples to benchmark")

    registry = load_registry()
    if registry.guardrails is None:
        sys.exit("ERROR: no guardrails block in the registry")
    guard = registry.get_instance(registry.guardrails.model)
    url = f"{guard.api_base}/chat/completions"
    model = guard.served_model_name or guard.model
    threshold = float(os.environ.get("THRESHOLD", registry.guardrails.threshold))
    concurrency = int(os.environ.get("CONCURRENCY", "1"))
    warmup = int(os.environ.get("WARMUP", "3"))

    print(f"Benchmarking {len(samples)} samples against {guard.id} "
          f"@ {url} (threshold {threshold}, concurrency {concurrency}) ...",
          file=sys.stderr)
    try:
        rows = asyncio.run(
            _run(samples, url, model, threshold, concurrency, warmup))
    except httpx.HTTPError as exc:
        sys.exit(f"ERROR: guard instance unreachable at {url}: {exc}\n"
                 "  Is the fleet up? `make up`")

    latencies = [r["latency_s"] for r in rows]
    report = _render_report(rows, latencies, threshold, model, concurrency)
    REPORT_PATH.write_text(report)
    print(report, file=sys.stderr)
    print(f"\n  -> {REPORT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
