# Tools Bench on the Prod Fleet — Bring-up Recap (2026-07-18)

How the 7-model tool-calling bench stack went live on the RTX Pro 6000 (96 GB),
every error hit on the way, and how to use KV cache and GPU memory correctly so
the next bring-up doesn't repeat them. Registry: `configs/models-prod.yaml`
(branch `feat/prod-tools-bench`).

## Using the stack

One endpoint, one key, seven models:

```bash
KEY=$(grep TOOLS_BENCH .tunnel/keys.env | cut -d= -f2)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen3.5-4b", "messages": [...], "tools": [...]}'
```

Models: `qwen3.5-0.8b`, `qwen3.5-2b`, `qwen3.5-4b`, `gemma-4-e2b`, `minicpm5-1b`
(vLLM, S3-streamed from Wasabi) and `bonsai-27b-1bit`, `ternary-bonsai-27b`
(PrismML llama.cpp, `make bonsai-up`). The bench never sees which backend serves
a model — everything is OpenAI-compatible through :4000.

Bench request rules learned the hard way:

- **Cap prompt + max_tokens at 16384** per request. That is `max_model_len` on all
  vLLM instances and `--ctx-size` on the llama.cpp servers. Oversize requests get a
  400 from vLLM but a **500 from llama.cpp**, and LiteLLM treats 500 as a deployment
  failure: with `allowed_fails: 1, cooldown_time: 60` the model then 429s
  ("No deployments available") for 60s while the server itself is perfectly healthy.
  Check `.bonsai/<id>.log` before assuming a crash.
- **Pass sampling explicitly for Ternary Bonsai** (`temperature: 0.7, top_p: 0.95`,
  the model card values). At llama-server defaults its tool-calling is flaky
  (returned prose instead of a tool call in repeated trials).
- Bonsai servers run `--reasoning-budget 0` to match the fleet's
  `enable_thinking: false`. Without it the 1-bit model burns the entire completion
  budget inside `<think>` and returns empty messages.

## GPU memory: how to size `gpu_memory_utilization`

`gpu_memory_utilization` is an absolute fraction of the 96 GB card, and vLLM
(≥0.21, CUDA-graph profiling on by default) spends it as:

```
util × 95.6 GiB = weights + activation peak + CUDA graphs (~1.6 GiB) + KV cache
```

The original paper layout (0.03–0.14 ratios) ignored the CUDA-graph term and every
instance failed with negative KV ("No available memory for the cache blocks").
Deployed values, all verified live:

| Instance | util | ≈ GB | KV at boot |
|---|---|---|---|
| qwen3.5-0.8b | 0.10 | 9.6 | 5.61 GiB |
| qwen3.5-2b | 0.12 | 11.5 | 4.61 GiB |
| qwen3.5-4b | 0.16 | 15.3 | 4.02 GiB |
| gemma-4-e2b | 0.18 | 17.2 | 2.79 GiB |
| minicpm5-1b | 0.08 | 7.6 | 4.55 GiB |

Sum 0.64 of a 0.90 budget. The slack is not free: the two llama.cpp servers
(~15 GB combined) are `remote_models`, invisible to `make check`'s budget math —
leave them headroom by hand. Steady state with all 7 models: ~83/96 GB.

**Mamba hybrids need a seq cap.** Every decode sequence on a Qwen3.5 (GDN + full
attention) instance holds one Mamba cache block. vLLM's default `max_num_seqs`
1024 demanded more blocks than the whole KV allocation held and aborted CUDA graph
capture ("max_num_seqs (1024) exceeds available Mamba cache blocks (66)"). All
three Qwen instances run `--max-num-seqs 64`, which is ample for bench concurrency
and shrinks graph memory too.

## KV cache: LMCache alignment on hybrid models

Each vLLM instance pairs with an `lmcache server` (MP mode, port +1000) holding L1
cache in CPU RAM (20 GB per instance — confirm host RAM before densifying). For
Mamba hybrids, `lmcache.chunk_size` MUST equal the unified attention block size N
that vLLM prints at startup ("Setting attention block size to N tokens"), because
`mamba_align` derives `--max-num-batched-tokens 2N-1` from it and the engine
enforces `block_size <= max_num_batched_tokens < 2*block_size`.

**N is per-variant, not per-family:** 0.8B and 2B print 544, but **4B prints 528**.
Booting the 4B with the family's 544 failed hard (`max_num_batched_tokens=1087,
block_size=528`). Never copy a sibling's chunk_size; read it from the target
variant's own log. Values in the registry are now marked verified.

Gemma 4 (SWA + full hybrid) needs no alignment; its sliding window is also why
2.79 GiB KV is enough at 16K context. The Bonsai models do their own KV handling
in llama.cpp (hybrid attention, context checkpoints — visible in `.bonsai/*.log`).

## Boot errors encountered, in order

| Error (log signature) | Cause | Fix |
|---|---|---|
| `Available KV cache memory: -4.33 GiB` → "No available memory for the cache blocks" | util 0.03 didn't cover weights + 1.6 GiB CUDA graphs | resize utils (table above) |
| `max_num_seqs (1024) exceeds available Mamba cache blocks (66)` | Mamba: 1 cache block per seq, default 1024 | `--max-num-seqs 64` on Qwen instances |
| `Mamba-hybrid models with LMCache require block_size <= max_num_batched_tokens < 2 * block_size; got 1087, block_size=528` | 4B's N is 528, registry said 544 | per-variant chunk_size from startup log |
| Healthy :8000/:8001 died mid-boot with async_llm tracebacks | killed the backgrounded `make up`; instances are its process-group children | `setsid nohup make up-timeout ... &`, never TaskStop it; fix stragglers after it exits |
| Proxy: `Unable to find Prisma binaries` | fresh env, prisma client never generated | `cd <site-packages>/litellm/proxy && prisma generate`, then `make start` |
| Bench 429 `No deployments available ... cooldown_list` | >16K request → llama.cpp 500 → LiteLLM cooldown | cap requests at 16K; see follow-ups |

Ops pattern that caught all of these fast: monitor per-instance logs during boot
instead of waiting on the aggregate health gate —

```bash
tail -F logs/*.log | grep -E "EngineCore failed|No available memory|ValueError|Application startup complete|Available KV cache memory"
```

## Follow-ups (open)

- Long-context Bonsai scenarios: `make bonsai-down && CTX_SIZE=131072 make bonsai-up`
  (model supports 262K; watch `nvidia-smi`, ~13 GB were free at steady state).
- The router cooldown (`allowed_fails: 1`, hardcoded in
  `tunnel/gateway/config_builder.py`) is a hair-trigger for bench traffic where
  malformed requests are expected; consider raising it for the prod registry.
- `make perf` (performbench gates) has not yet been run against this fleet.
- Bonsai promotion to a first-class `backend: llamacpp` instance type stays
  deferred until bench numbers justify it (research-phase decision, 2026-07-18).
