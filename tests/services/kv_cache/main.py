"""Standalone KV-cache benchmark against the Tunnel Engine gateway (not pytest).

Run once the fleet is up (make up):

  python tests/services/kv_cache/main.py                 # every registered target
  python tests/services/kv_cache/main.py minicpm-1b      # only these ids
  N_PARAS=85 N_PROMPTS=10 python tests/services/kv_cache/main.py

Sends N_PROMPTS unique long-prefix prompts twice per target (cold then warm)
and measures TTFT; a working KV cache makes the warm round's TTFT collapse.
Targets come from the registry (local instances + remote_models whose
api_key_env is set). Results -> stdout + RESULTS.md. See docs/lmcache.md.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tunnel.registry import TunnelRegistry, load_registry  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = str(_ROOT / "configs" / "models.yaml")
RESULTS_PATH = Path(__file__).resolve().parent / "RESULTS.md"

N_PROMPTS = int(os.environ.get("N_PROMPTS", "10"))
N_PARAS = int(os.environ.get("N_PARAS", "60"))  # ~6k-token prefix; safe for template overhead
MAX_TOKENS = 48
REQUEST_TIMEOUT_S = 120.0

from dataset import PromptCase, build_prompts  # noqa: E402


def _select_targets(registry: TunnelRegistry, argv: list[str]) -> list[tuple[str, str]]:
    """Resolve which model ids to benchmark and a human label for each.

    Args:
        registry: The loaded registry.
        argv: Optional explicit list of ids from the command line.

    Returns:
        List of (model_id, backend_label) tuples, in registry order.
    """
    labels: dict[str, str] = {}
    for inst in registry.instances:
        if inst.lmcache.enabled:
            labels[inst.id] = f"local LMCache:{inst.lmcache.backend}"
        else:
            labels[inst.id] = "local vLLM-native cache"
    for rm in registry.remote_models:
        if os.environ.get(rm.api_key_env):
            labels[rm.id] = f"remote {rm.provider}"

    if argv:
        return [(mid, labels.get(mid, "unknown")) for mid in argv]
    return list(labels.items())


def _one_call(client: httpx.Client, base_url: str, key: str | None,
              model: str, case: PromptCase) -> dict:
    """Stream one completion, returning ttft/total latency and usage.

    Raises:
        httpx.HTTPStatusError: on a non-2xx response (e.g. context exceeded).
    """
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": case.system},
            {"role": "user", "content": case.user},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    t0 = time.perf_counter()
    ttft: float | None = None
    usage: dict = {}
    with client.stream("POST", f"{base_url}/v1/chat/completions",
                       json=body, headers=headers) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            if ttft is None and chunk.get("choices"):
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content") or delta.get("reasoning_content"):
                    ttft = time.perf_counter() - t0
    total = time.perf_counter() - t0
    return {"ttft": ttft if ttft is not None else total, "total": total, "usage": usage}


def _cached_tokens(usage: dict) -> int:
    """Best-effort cache-hit token count across provider usage shapes."""
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens", 0) or usage.get("prompt_cache_hit_tokens", 0)


def _run_round(client, base_url, key, model, cases) -> dict | None:
    """Run one round over all cases. Returns None if the model can't serve them."""
    ttfts, cached, prompt_tok = [], 0, 0
    for case in cases:
        try:
            row = _one_call(client, base_url, key, model, case)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            if "ContextWindowExceeded" in body or exc.response.status_code == 400:
                return {"skipped": "context window exceeded (lower N_PARAS)"}
            return {"skipped": f"HTTP {exc.response.status_code}: {body[:120]}"}
        except httpx.HTTPError as exc:
            return {"skipped": f"request error: {exc}"}
        ttfts.append(row["ttft"])
        cached += _cached_tokens(row["usage"])
        prompt_tok += row["usage"].get("prompt_tokens", 0)
    return {
        "ttft_mean": statistics.mean(ttfts),
        "ttft_median": statistics.median(ttfts),
        "cached_tokens": cached,
        "prompt_tokens": prompt_tok,
    }


def _warmup(client, base_url, key, model, warmup_cases) -> None:
    """Pay one-time startup costs (CUDA graph capture, compile specialization for
    the long-sequence path) on throwaway prompts so the cold round measures
    steady-state prefill, not first-request warmup. Uses a different salt than
    the real cases, so it warms the compute path without caching the test prompts.
    """
    for case in warmup_cases:
        try:
            _one_call(client, base_url, key, model, case)
        except httpx.HTTPError:
            return  # a model that can't serve warmup will be caught/skipped in the round


def _bench(client, base_url, key, model, label, cases, warmup_cases) -> dict:
    """Warmup, then cold + warm rounds for one model."""
    print(f"\n=== {model}  ({label}) ===")
    print("  warmup (discarded)...")
    _warmup(client, base_url, key, model, warmup_cases)
    print("  cold round (populate cache)...")
    cold = _run_round(client, base_url, key, model, cases)
    if "skipped" in cold:
        print(f"  SKIPPED: {cold['skipped']}")
        return {"model": model, "label": label, "skipped": cold["skipped"]}
    print("  warm round (replay identical prompts)...")
    warm = _run_round(client, base_url, key, model, cases)
    if "skipped" in warm:
        return {"model": model, "label": label, "skipped": warm["skipped"]}
    speedup = cold["ttft_mean"] / warm["ttft_mean"] if warm["ttft_mean"] else 0.0
    print(f"  cold TTFT mean={cold['ttft_mean']*1000:8.1f}ms | "
          f"warm TTFT mean={warm['ttft_mean']*1000:8.1f}ms | speedup={speedup:.2f}x")
    return {"model": model, "label": label, "cold": cold, "warm": warm, "speedup": speedup}


def _render_markdown(results: list[dict], meta: dict) -> str:
    """Render the results table + detail into a Markdown report."""
    lines = [
        "# KV-Cache Benchmark Results",
        "",
        "> Auto-generated by `tests/services/kv_cache/main.py`. Re-run to refresh.",
        "",
        "## Run configuration",
        "",
        f"- Timestamp (UTC): {meta['timestamp']}",
        f"- Registry: `{meta['registry']}`",
        f"- Gateway: `{meta['gateway']}`",
        f"- Prompts per model: {meta['n_prompts']}  |  paragraphs/prefix: {meta['n_paras']}"
        f"  |  max_tokens: {MAX_TOKENS}",
        f"- Salt: `{meta['salt']}` (makes the cold round genuinely cold)",
        "",
        "Method: a warmup pass (discarded) pays one-time CUDA-graph/compile costs, then",
        "each model gets N unique long-prefix prompts sent twice -- a **cold** round (KV",
        "computed) then a **warm** round (identical prompts, KV reused). We measure",
        "time-to-first-token (TTFT), which is dominated by prefill. A working KV cache",
        "collapses the warm TTFT; without warmup, cold-start noise would inflate speedup.",
        "",
        "## Summary",
        "",
        "| Model | Backend | Cold TTFT | Warm TTFT | Speedup | Warm cached tokens |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {r['model']} | {r['label']} | — | — | skipped | {r['skipped']} |")
            continue
        c, w = r["cold"], r["warm"]
        cached = f"{w['cached_tokens']}/{w['prompt_tokens']}" if w["prompt_tokens"] else "n/a"
        lines.append(
            f"| {r['model']} | {r['label']} | {c['ttft_mean']*1000:.0f} ms | "
            f"{w['ttft_mean']*1000:.0f} ms | **{r['speedup']:.2f}x** | {cached} |"
        )
    lines += [
        "",
        "## How to read this",
        "",
        "- **Speedup > 1** on a local LMCache model = KV reuse is working: the warm round",
        "  skips recomputing the long prefix. The gain grows with prefix length.",
        "- **Hybrid-attention models** (Qwen3.5 Mamba+Full, Gemma 4 SWA+Full) are cached",
        "  via the LMCache MP connector too; a speedup ~= 1 there means the cache is broken.",
        "- **Remote models** are network-bound so TTFT barely moves, but *warm cached tokens*",
        "  shows the provider's server-side prompt cache hitting (a direct cost saving).",
        "- **Local `cached tokens` reads 0 even when LMCache is hitting**: LMCache hits are",
        "  internal to vLLM and not reported in the OpenAI `usage` field. For local models the",
        "  TTFT speedup is the signal; confirm hits in the instance log (`LMCache hit tokens`).",
        "",
        f"Sample prompt (truncated): _{meta['sample'][:240]}..._",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    registry = load_registry(REGISTRY_PATH)
    base_url = f"http://localhost:{registry.litellm.port}"
    key = registry.litellm.resolved_master_key
    salt = os.environ.get("SALT", f"r{int(time.time())}")
    cases = build_prompts(N_PROMPTS, N_PARAS, salt)
    # A couple of same-length prompts with a different salt: warm the compute path
    # without seeding the cache for the real cases.
    warmup_cases = build_prompts(2, N_PARAS, f"{salt}-warmup")

    targets = _select_targets(registry, sys.argv[1:])
    if not targets:
        print("No targets. Is the registry empty, or are remote api keys unset?", file=sys.stderr)
        return 1

    results = []
    with httpx.Client(timeout=REQUEST_TIMEOUT_S) as client:
        for model, label in targets:
            results.append(_bench(client, base_url, key, model, label, cases, warmup_cases))

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "registry": REGISTRY_PATH, "gateway": base_url,
        "n_prompts": N_PROMPTS, "n_paras": N_PARAS, "salt": salt,
        "sample": cases[0].system.replace("\n", " "),
    }
    RESULTS_PATH.write_text(_render_markdown(results, meta))

    print("\n" + "=" * 70)
    print(f"{'MODEL':<20} {'COLD':>10} {'WARM':>10} {'SPEEDUP':>9}")
    print("-" * 70)
    for r in results:
        if "skipped" in r:
            print(f"{r['model']:<20} {'skipped: ' + r['skipped']:>41}")
        else:
            print(f"{r['model']:<20} {r['cold']['ttft_mean']*1000:>8.0f}ms "
                  f"{r['warm']['ttft_mean']*1000:>8.0f}ms {r['speedup']:>8.2f}x")
    print(f"\nWrote {RESULTS_PATH.relative_to(_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
