# KV Cache Performance: LMCache MP on Hybrid Attention

Research recap, 2026-07-17. Hardware: Lightning AI, 1x L4 24GB, 31GB RAM.
Model under test: Qwen/Qwen3.5-0.8B (Mamba GDN + Full attention hybrid).

## What changed

- lmcache 0.4.2 -> 0.5.1, connector `LMCacheConnectorV1` -> `LMCacheMPConnector`.
- The old in-process connector crashed vLLM startup on hybrid-attention
  models, so qwen-0.8b ran with LMCache disabled. The MP connector manages
  each KV cache group (full attn, sliding window, Mamba) independently, so
  hybrid models now cache.
- Architecture: one `lmcache server` process per instance (ZMQ on instance
  port + 1000). vLLM connects via `kv_connector_extra_config.lmcache.mp.port`.
  The server owns L1 (CPU RAM) and optional L2 (fs / redis RESP adapter).

## Required config for Mamba hybrids

vLLM prints the unified attention block size N at startup:

```
Setting attention block size to 544 tokens ...
```

Registry rules (`configs/models.yaml`):

- `lmcache.chunk_size` = N (544 for Qwen3.5-0.8B)
- `lmcache.mamba_align: true` emits `--mamba-cache-mode align`,
  `--enable-prefix-caching`, `--max-num-batched-tokens 2N-1` (1087)

Gemma 4 (SWA + full) needs neither; the MP connector handles heterogeneous
block sizes out of the box.

## Results

Method: 600-sentence prompt (~7k tokens), streaming chat, TTFT measured
client-side. LMCache isolation: populate cache, restart the vLLM instance
(GPU prefix cache wiped, lmcache server survives), absorb first-request
warmup with a throwaway prompt, then replay.

| Path | TTFT |
|---|---:|
| Cold prefill (never seen) | 1212 ms |
| vLLM local prefix cache hit | 107 ms |
| LMCache-only hit (after engine restart) | 145 ms |

- LMCache vs cold: **8.4x**. Server log confirms `15/15 retained keys (15 L1)`.
- Store cost: 544-token chunks stored in 4-7 ms each, off the hot path.
- Cache survives engine restarts by design: `tunnel stop <id>` leaves the
  lmcache server running; `tunnel down` stops everything.

## Gate results (make perf, 2026-07-17)

| Scenario | Result |
|---|---|
| smoke (aiperf, conc 4, ISL 512/OSL 64) | PASS. TTFT p99 273 ms, 336 tok/s output |
| kv-longdoc (gate: TTFT gain >= 1.5x) | PASS. Gain 4.17x, latency gain 1.66x |
| goodput via gateway (TTFT<=2s, ITL<=100ms, conc 8) | PASS. 1.95 of 2.18 req/s in SLO (~89%) |

## Caveats / open items

- TTFT tail at concurrency 8 through the gateway: max ~16s. Cause:
  `--max-num-batched-tokens 1087` (the 2N-1 rule) caps prefill batching, so
  concurrent long prefills queue. Next experiment: larger multiples of N.
- Cached vs uncached generation is not bit-identical (LMCache caveat);
  compare eval metrics, not token strings.
- LFM2.5 (conv+attn hybrid) is not in LMCache's validated list; left
  disabled in the prod registry until verified.
- Mamba prefix caching is still experimental in vLLM.

## Reproduce

```bash
make up            # auto-starts tunnel-pg if needed
make perf          # gated suite: smoke + kv-longdoc
make perf SCENARIOS="goodput mixed ttft-sweep"
```

Artifacts land in `tests/services/performbench/results/<ts>/`; summary in
`tests/services/performbench/RESULTS.md`. See also `docs/lmcache.md`.
