## Hybrid Attention Models

LMCache supports hybrid-attention architectures through `LMCacheMPConnector`.

Supported architectures include:

| Model | Attention |
|--------|-----------|
| Gemma 3 / 4 | Sliding Window + Full |
| GPT-OSS | Sliding Window + Full |
| Qwen3.5 / Qwen3.6 | Mamba (GDN) + Full |
| DeepSeek-V4 Flash | Sparse MLA |
| GLM 5.1 / 5.2 | Dynamic Sparse |
| MiniMax M3 | Sparse + Lightning Indexer |

LMCache automatically detects every KV cache group and manages them independently. No additional hybrid-specific flags are required.

---

## Object Group Separation

LMCache separates cache objects by attention type.

Default:

```bash
lmcache server \
    --chunk-size 256 \
    --l1-size-gb 100
```

Disable separation:

```bash
lmcache server \
    --chunk-size 256 \
    --l1-size-gb 100 \
    --no-separate-object-groups
```

This only changes storage layout. Cache correctness is unaffected.

---

## Qwen3.5 / Qwen3.6 (Mamba Hybrid)

Validated:

- Qwen/Qwen3.5-0.8B
- Qwen/Qwen3.6-27B

Architecture:

```
Full Attention
       │
       ▼
Mamba / Gated DeltaNet
       │
       ▼
Full Attention
       │
       ▼
Mamba / Gated DeltaNet
```

LMCache transparently converts Mamba recurrent states into cacheable pages, enabling prefix caching without model-specific code.

---

## Find the Unified Block Size

Each model has a unified attention block size **N**.

vLLM prints it during startup:

```text
INFO Setting attention block size to 544 tokens...
```

Example:

```bash
vllm serve Qwen/Qwen3.5-4B \
    --enable-prefix-caching \
    --mamba-cache-mode align
```

Typical values:

| Model | Block Size |
|--------|-----------:|
| Qwen3.5-0.8B | 544 |
| Qwen3.6-27B | 784 |

---

## Required Configuration

LMCache:

```bash
lmcache server \
    --chunk-size N \
    --l1-size-gb 100
```

vLLM:

```bash
vllm serve <model> \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --max-num-batched-tokens $((2*N-1)) \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

Rules:

- `--chunk-size = N`
- `--max-num-batched-tokens = 2N-1` (recommended)
- `--mamba-cache-mode align` is required.

Using `N` also works but reduces scheduler throughput under concurrent requests.

---

## Caveats

- Generation is not bit-identical between cached and uncached runs.
- Compare evaluation metrics instead of generated tokens.
- Cache entries cannot be shared across different attention backends.
- Image/video KV caching is not validated.
- Mamba prefix caching is still experimental in vLLM.

---

## Gemma 4

Validated:

- google/gemma-4-31B-it
- google/gemma-4-12B-it
- google/gemma-4-E4B-it

Architecture:

```
Sliding Window
       │
       ▼
Full Attention
       │
       ▼
Sliding Window
```

Gemma 4 is supported out of the box.

Start LMCache:

```bash
lmcache server \
    --l1-size-gb 100
```

Single GPU:

```bash
vllm serve google/gemma-4-12B-it \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

Multi GPU:

```bash
vllm serve google/gemma-4-31B-it \
    --tensor-parallel-size 2 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

Adjust `--tensor-parallel-size` for your hardware.

---

## Gemma 4 Notes

- Hybrid KV cache with heterogeneous block sizes is fully supported.
- LMCache automatically handles different block sizes for sliding-window and full-attention groups.
- Cross-layer KV sharing is preserved automatically.
- No additional configuration is required.
- CacheGen compression has not been validated.

---

## Verification

To verify KV reuse:

1. Run an evaluation to populate LMCache.
2. Clear only vLLM's local prefix cache.

```bash
curl -X POST http://localhost:8000/reset_prefix_cache
```

3. Run the evaluation again.

Expected behavior:

```
vLLM Local Cache  -> MISS
LMCache           -> HIT
Evaluation Score  -> Same
```

## Uniform Attention Models

Unlike hybrid architectures, these models use a single attention mechanism across every layer. No unified block-size tuning or hybrid KV management is required.

LMCache works out of the box.

---

## Qwen3 MoE

Validated:

- Qwen/Qwen3-235B-A22B
- Qwen/Qwen3-30B-A3B
- Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8
- Qwen/Qwen3-Coder-30B-A3B-Instruct

Architecture:

```
Attention
   │
   ▼
MoE
   │
   ▼
Attention
   │
   ▼
MoE
```

Start LMCache:

```bash
lmcache server \
    --l1-size-gb 100 \
    --eviction-policy LRU
```

### Single GPU

```bash
vllm serve Qwen/Qwen3-30B-A3B \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

### Multi GPU (Expert Parallel)

```bash
vllm serve Qwen/Qwen3-235B-A22B \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'
```

Qwen3-Coder models use the `qwen3_coder` parser.

Adjust `--tensor-parallel-size` to match your hardware.

No hybrid-specific configuration is required.

---

## KV Cache SDK

LMCache SDK provides direct programmatic access to KV cache objects.

Instead of only reading cached KV, you can retrieve it, modify it on CPU, and write it back before decoding resumes.

Typical use cases:

- Token Dropping
- KV Compression
- KV Pruning
- Custom KV Transformations
- Research experiments

---

## Pipeline

```
Prompt
   │
   ▼
Prefill (vLLM)
   │
   ▼
LMCache
   │
   ▼
CPU KV Editor
   │
   ▼
Modified KV
   │
   ▼
Decode
```

Workflow:

1. Prefill prompt.
2. Store KV into LMCache.
3. Retrieve KV to CPU.
4. Apply custom edit function.
5. Store modified KV.
6. Continue decoding.

---

## Start LMCache

Enable shared-memory transfer.

```bash
lmcache server \
    --l1-size-gb 150 \
    --chunk-size 256 \
    --eviction-policy LRU \
    --port 6555 \
    --http-port 8080 \
    --shm-name lmcache_kvcache_sdk \
    --no-l1-use-lazy
```

---

## Start vLLM

```bash
vllm serve Qwen/Qwen3-8B \
    --port 8000 \
    --enforce-eager \
    --gpu-memory-utilization 0.65 \
    --return-tokens-as-token-ids \
    --kv-transfer-config '{
        "kv_connector":"LMCacheMPConnector",
        "kv_role":"kv_both",
        "kv_connector_extra_config":{
            "lmcache.mp.port":6555
        }
    }'
```

---

## Python SDK

```python
import lmcache.sdk.kvcache as lmc_sdk

ctx = lmc_sdk.connect(
    url="tcp://localhost:6555",
    http_url="http://localhost:8080",
    model_name="Qwen/Qwen3-8B",
)

...

lmc_sdk.close(ctx)
```

---

## Custom KV Editing

Your edit function receives:

- KV tensor
- Token IDs

Returns:

- Modified KV
- Modified Token IDs

```
LMCache
    │
    ▼
Edit Function
    │
    ├── Drop Tokens
    ├── Compress KV
    ├── Prune Layers
    └── Custom Logic
    │
    ▼
Store Back
```

The SDK automatically processes every request in the batch.

---

## SDK APIs

| API | Description |
|------|-------------|
| `connect()` | Create SDK context |
| `close()` | Release resources |
| `create_request()` | Create request stream |
| `LMCacheBatchedStream()` | Create batch |
| `batch.add()` | Add request |
| `batch.prefill()` | Generate KV cache |
| `batch.modify()` | Edit KV cache |
| `batch.decode()` | Continue generation |

Metrics returned include:

- Prefill throughput
- Decode throughput
- Input tokens
- Output tokens
- Modify latency

---

## Notes

- KV tensors are edited entirely on CPU.
- Shared memory is recommended for maximum throughput.
- Token IDs are required to correctly identify cached KV entries.
- The SDK is intended for advanced KV cache research and optimization.