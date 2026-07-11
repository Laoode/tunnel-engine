# LMCache KV-Cache Offload

Maintenance reference for the LMCache integration: what it does, how it is wired
into vLLM, how to configure each backend, which models it works with, and how to
benchmark it. LMCache stores the transformer KV cache in tiers below GPU HBM
(CPU RAM, local disk, or Redis) so a repeated prompt prefix is *loaded* instead
of *recomputed*, cutting time-to-first-token (TTFT) and freeing GPU compute.

## How it is wired

Two things must both be true for LMCache to actually run (either alone is a
silent no-op):

1. **Config file** — `make generate` writes `configs/lmcache/<id>.yaml` per
   LMCache-enabled instance (from `build_lmcache_config` in
   `tunnel/cache/lmcache_config.py`). `cmd_serve` sets
   `LMCACHE_CONFIG_FILE=configs/lmcache/<id>.yaml` before launching vLLM.
2. **KV connector** — `build_serve_command` passes
   `--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`
   whenever `lmcache.enabled`. Without this flag vLLM ignores the config file and
   only its in-GPU prefix cache runs.

Enabling the connector turns **off** vLLM's own hybrid KV manager and native
prefix cache; LMCache becomes the sole KV-reuse layer (GPU HBM is L0, then the
LMCache tiers). LMCache keys are prefixed with `model_name`, so multiple models
(or replicas of one model) can safely share one remote store without collisions.

## Registry configuration

Per instance, under `lmcache:` (merged from `defaults:` in `configs/models.yaml`):

```yaml
lmcache:
  enabled: true          # false => no LMCache, no connector flag
  backend: cpu           # cpu | disk | redis
  max_cache_size_gb: 20  # size of the tier (CPU RAM or disk)
  chunk_size: 256         # KV chunk granularity (tokens)
  remote_serde: naive    # naive | cachegen  (redis backend only)
```

Generated flat LMCache config (`configs/lmcache/<id>.yaml`) per backend:

| backend | emitted keys |
|---|---|
| `cpu` | `local_cpu: true`, `max_local_cpu_size: <gb>` |
| `disk` | `local_cpu: false`, `local_disk: /tmp/lmcache/<id>`, `max_local_disk_size: <gb>` |
| `redis` | `local_cpu: true` (L1) + `remote_serde`; `remote_url` injected at serve time |

### Redis backend

The Redis host/port are environment-specific, so they are never committed. Set
them in `.env`; `cmd_serve` assembles `redis://host:port` and exports it as
`LMCACHE_REMOTE_URL` (LMCache merges env over the config file):

```
LMCACHE_REDIS_HOST=localhost
LMCACHE_REDIS_PORT=6379
LMCACHE_REMOTE_SERDE=naive
```

URL scheme is `redis://` (also `rediss://`, `redis-sentinel://`). If
`LMCACHE_REDIS_HOST` is unset, the instance warns and degrades to the local CPU
tier. Redis gives KV *persistence* across restarts and spillover beyond host RAM;
same-model replica sharing is the natural extension.

## Model compatibility (important)

LMCache's connector requires a single unified KV cache type, so it is
**incompatible with hybrid-attention / SSM models**. Qwen3.5
(`Qwen3_5ForConditionalGeneration`, gated delta-net / linear attention) fails at
startup with:

```
ValueError: Hybrid KV cache manager is disabled but failed to convert the KV
cache specs to one unified type.
```

Set `lmcache.enabled: false` for such models (they fall back to vLLM's native
prefix cache, which yields ~0% hit rate on hybrid layers anyway). Standard dense
transformers (e.g. MiniCPM5-1B) work.

| Model | Works with LMCache? |
|---|---|
| MiniCPM5-1B (dense) | yes |
| Qwen3.5 family (hybrid GDN) | no — set `lmcache.enabled: false` |

## Verifying it works

Watch the vLLM instance log for LMCache init and per-request hit stats:

```
Created backend: LocalCPUBackend
Reqid: ..., Total tokens 5045, ... LMCache hit tokens: 4864, need to load: 0
```

`LMCache hit tokens` on the warm request is the direct proof of KV reuse.

## Benchmark

Reusable benchmark: `tests/services/kv_cache/main.py` (dataset in
`dataset.py`). It sends N unique long-prefix prompts twice (cold then warm) at
every registered target and reports the TTFT collapse. Results are written to
`tests/services/kv_cache/RESULTS.md`.

```bash
make up                                   # bring the fleet up
make bench-cache                          # or: python tests/services/kv_cache/main.py
python tests/services/kv_cache/main.py minicpm-1b        # specific ids
N_PARAS=85 N_PROMPTS=10 make bench-cache                 # bigger prefix => bigger win
```

Targets are registry-driven: every local instance, plus any `remote_models`
entry whose `api_key_env` is set (so DeepSeek's server-side cache is compared
too). The prefix length auto-caps via graceful skip if a model's chat template
pushes a prompt past its context window (lower `N_PARAS` and re-run).

### Observed on L4 24GB (2026-07-11)

KV-reuse speedup scales with prefix length; measured cold-vs-warm TTFT:

| Model | Backend | ~5k-tok | ~8.5k-tok | ~9.2k-tok |
|---|---|---|---|---|
| minicpm-1b | LMCache CPU | 3.9x | 5.8x | 8.6x |
| qwen-0.8b | hybrid, LMCache off | ~1x (0% hit) | — | — |
| deepseek-v4-flash | remote | ~1x TTFT, 98% prompt tokens server-cached | | |

LMCache logs confirmed 4864/5045 (96%) of the prefix KV served from cache on
warm requests. See `tests/services/kv_cache/RESULTS.md` for the latest run.
