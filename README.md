<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=600&size=47&duration=1&pause=1000&color=000000&background=4add9c&center=true&vCenter=true&repeat=false&width=1200&lines=Tunnel+Engine+:+A+Unified+Gateway+for+All+LLM+Services" alt="Title" />
</h1>

<div align="center">
  <img src="https://github.com/Laoode/Tunnel-Engine/blob/main/assets/the-tunnel.png" alt="The Tunnel">
</div>

<p align="center">
  <b>📟 LLM inference is fundamentally different. Standard backend logic won't cut it.</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&pause=1800&color=4add9c&center=true&vCenter=true&width=1000&lines=Unified+API+Gateway+for+Multiple+LLM+Models;Blazing+Fast+Inference+via+vLLM+Continuous+Batching;Distributed+KV-Cache+Sharing+with+LMCache;Smart+Load+Balancing+%26+Automatic+Fallbacks;Built+for+Production-Ready+AI+Microservices" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=yellow" />
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=green" />
  <img src="https://img.shields.io/badge/vLLM-Model%20Serving-30a2ff" />
  <img src="https://img.shields.io/badge/LMCache-Global%20Cache-599aac" />
  <img src="https://img.shields.io/badge/LiteLLM-Orchestration-white" />
  <img src="https://img.shields.io/badge/NeMo_Guardrails-AI_Safety-76B900?logo=nvidia" />
  <img src="https://img.shields.io/badge/APISIX-Edge_Gateway-e8433f?logo=apacheapisix&logoColor=white" />
  <img src="https://img.shields.io/badge/Kubernetes-Service_Mesh-326CE5?logo=kubernetes&logoColor=white" />
  <img src="https://img.shields.io/badge/KEDA-Autoscaling-FF4500?logo=keda&logoColor=white" />
  <img src="https://img.shields.io/badge/Redis-Distributed%20Cache-DC382D?logo=redis&logoColor=red" />
  <img src="https://img.shields.io/badge/MinIO-Model_Storage-darkred?logo=minio&logoColor=darkred" />
  <img src="https://img.shields.io/badge/Prometheus-Metrics-E6522C?logo=prometheus&logoColor=orange" />
  <img src="https://img.shields.io/badge/Grafana-Observability-F46800?logo=grafana&logoColor=orange" />
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=blue" />
</p>

---

Real production workloads require:<br>
➢ Multiple endpoints<br>
➢ Multiple model families<br>
➢ Parallel async inference<br>
➢ High availability<br>
➢ Load balancing<br>
➢ Fault tolerance<br>
➢ Efficient GPU memory sharing<br>
➢ Caching<br>

That's where Tunnel Engine comes in. It provides a single endpoint link to access multiple LLM models. By simply changing the model name, we can easily maintain all model providers.

vLLM enables efficient multi-model serving with continuous batching, PagedAttention, precise GPU memory utilization, KV-cache, and separate URL endpoints per instance.

LMCache acts as the extender for vLLM. To store the KV-cache precisely, we need LMCache. It takes the KV-cache from vLLM and saves it to cheaper memory (local RAM or storage, instead of GPU memory only) so it can be reused later. LMCache also supports distributed cache synchronization, allowing multiple vLLM nodes to share caches.

LiteLLM is used for intelligent load balancing. While vLLM runs on multiple separate ports (e.g., ports 8000, 8001, 8002), LiteLLM wraps all of these ports into a single endpoint URL (e.g., port 4000) for all our microservices to call. Additionally, if a model in a specific service crashes (e.g., Out of Memory), we can set up alternative fallback models to handle the requests seamlessly without pausing the service.

Architecture (what we need):
```
   Our services        ┌───────────────────────────┐
   call this     ────▶ │   Apache APISIX Gateway   │  ← Edge TLS, Global API Key Auth, WAF
                       └─────────────┬─────────────┘
                                     │ (Internal VPC)
                                     ▼
                        ┌───────────────────────────┐  
   Rate limiting ────▶  │    LiteLLM Proxy :4000    │ ──(Tracks Latency/Token Counts)  ────▶  [ Prometheus ]
                        │ (Routing / Load-Balancer) │                                                ▲
                        └─────────────┬─────────────┘                                                │
                        ▲             │(Calls Hook)                                                  │
      (Checks/Sanitizes)│   ┌─────────▼─────────┐                                                    │
                        └───│  NeMo Guardrails  │  ← Async or synchronous validation block           │
                            └─────────┬─────────┘                                                    │
                                      │ (Validated request)                                          │
                        ┌─────────────▼─────────────┐                                                │
                        │  Kubernetes Service Mesh  │  ← K8s Load Balancer    | [Optional rn]        │
                        └─────────────┬─────────────┘                                                │
┌─────────────────────┐               │                                                              │
│ MinIO Model Storage │               │                                                              │
└───▲─────────────────┘               │                                                              │
    |               ┌─────────────────▼─────────────────┐                                            │
    |               │                                   │                                            │
    |  ┌────────────▼──────────┐             ┌──────────▼────────────┐                               │
    |  │   vLLM Pod 1 :8000    │             │   vLLM Pod 2 :8001    │ ← Autoscaled via KEDA         │
    └─ │    (Model A v2)       │             │     (Model A v2)      │ ──(vllm:num_requests_waiting)─┘
       └────────────┬──────────┘             └──────────┬────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
                    ┌───────────────────────────────────┐
                    │      LMCache + Distributed Redis  │  ← Ultra-low TTFT across pods
                    └───────────────────────────────────┘

```
    
Running models manual via vLLM:
```bash
# Instance 1: Qwen 0.8B 
vllm serve Qwen/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.35 \
  --max-model-len 65536

# Instance 2: MiniCPM 1B 
vllm serve openbmb/MiniCPM5-1B \
  --port 8001 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.45 \
  --max-model-len 65536
```

## Quick Start

The whole workflow is driven by `make`. Run `make help` any time to list every target.

### 1. One-time setup

```bash
make install                 # install all dependencies

cp .env.example .env         # then edit .env and set at least:
                             #   HF_TOKEN=hf_...           (for gated models)
                             #   LITELLM_MASTER_KEY=...    (the gateway auth key)
```

Secrets live only in `.env`. They are never written into any file under `configs/`
(the registry references the key via `os.environ/LITELLM_MASTER_KEY`, resolved at boot).

### 2. Configure your models

`configs/models.yaml` is the single source of truth. Add or swap a model there, then:

```bash
make check       # validate the registry: YAML parses + GPU budget not exceeded (writes nothing)
make generate    # rebuild derived configs (configs/litellm/ + configs/lmcache/)
make list        # show registered instances, their ports, and the proxy port
```

Always run `make check` first after editing `models.yaml` - it catches YAML mistakes
and over-budget GPU splits before you ever launch a model.

### 3. Run the engine (pick ONE style)

**Option A - single model, foreground (dev / benchmarking):**

```bash
# terminal 1: load one model (stays in foreground, streams its logs)
make serve ID=<instance-id>     # use an id from `make list`

# terminal 2: once the model is loaded, health-gate and start the proxy on :4000
make start
```

**Option B - whole fleet, background (one command):**

```bash
make up          # launches every instance in the background, waits until all are
                 # healthy, then starts the proxy. Use `make down` to stop it all.
```

### 4. Verify and test

```bash
make health      # poll every instance + show GPU memory usage
make test        # full test suite
```

### 5. Stop and clean everything

```bash
make down         # stops ALL instances (started via `serve` OR `up`) and the proxy,
                  # then frees the GPU. Safe to run any time before switching models.
```

`make down` sweeps both tracked background instances and untracked foreground
`make serve` processes, plus the LiteLLM proxy - so nothing is left holding the GPU
or a port.

### 6. Retire ONE model while others keep serving

To stop a single model without touching the rest of the fleet or the gateway - for
example, decommissioning one model the next day while another is still serving live
production traffic:

```bash
make stop ID=<instance-id>       # stops just that instance, frees its GPU slice
```

This is safe with zero downtime for the surviving models: every instance is a
separate process on its own port, and the LiteLLM proxy is a separate process, so
stopping one never interrupts the others. The proxy keeps running and the remaining
models keep answering.

> [!NOTE]
>  The proxy still lists the stopped model in `/v1/models`, so calls routed to it
will error until you bring it back (`make serve ID=<id>`). To also stop the gateway
from advertising it, remove its block from `configs/models.yaml`, run `make generate`,
and restart the proxy during a maintenance window (a proxy restart briefly interrupts
the surviving models too).

## Golden rules (avoid conflicts)

1. **Edit only `configs/models.yaml`, then `make generate`.** Never hand-edit files
   under `configs/litellm/` or `configs/lmcache/` - they are auto-generated and get
   overwritten.
2. **Always `make down` before swapping models or re-running.** A leftover vLLM keeps
   the GPU memory and the port bound; the next launch will fail with an out-of-memory
   or port-in-use error. When unsure, confirm the GPU is clear with `nvidia-smi`
   (used memory should read 0 MiB).
3. **Every instance needs a unique `id` and `port`.** The registry rejects duplicate
   ids, duplicate ports, and any port that collides with the proxy - `make check`
   surfaces these instantly.
4. **Do not mix `make serve` and `make up` for the same model.** `serve` is foreground
   (one terminal per model); `up` is background (the whole fleet). Running both binds
   the same port twice.

## Command reference

| Command | What it does |
| --- | --- |
| `make install` | Install all dependencies |
| `make check` | Validate the registry without writing anything |
| `make generate` | Rebuild derived LiteLLM + LMCache configs |
| `make list` | List registered instances and ports |
| `make serve ID=<id>` | Launch one instance in the foreground |
| `make up` | Launch the whole fleet in the background + proxy |
| `make start` | Health-gate running instances, then start the proxy |
| `make health` | Poll instance health + GPU memory |
| `make stop ID=<id>` | Stop ONE instance, leave the others and the proxy running |
| `make down` | Stop every instance and the proxy, free the GPU |
| `make test` | Run the full test suite |
| `make view-models` | List locally cached HuggingFace models |

## Calling Tunnel Engine from your service

Tunnel Engine exposes **one OpenAI-compatible endpoint** for every model. Your service
always points at the same URL and key; to use a different model you change only the
`model` field to a registered instance id. No new endpoint per model, no client code
changes - that is the whole point of the gateway.

| What your client needs | Value |
| --- | --- |
| LLM endpoint (`base_url`) | `http://<tunnel-host>:4000/v1` (e.g. `http://localhost:4000/v1`) |
| LLM API key | your `LITELLM_MASTER_KEY` (the value in `.env`) |
| Model provider | `openai` - the gateway speaks the OpenAI API |
| LLM model | a registered instance id, e.g. `lfm2.5-8b-a1b` - **swap this to switch models** |

Discover the available model ids with `make list`, or `GET /v1/models` on the gateway.

### curl

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2.5-8b-a1b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:4000/v1",   # the ONE endpoint
    api_key="sk-...",                       # your LITELLM_MASTER_KEY
)

resp = client.chat.completions.create(
    model="lfm2.5-8b-a1b",                  # swap this id to use another model
    messages=[{"role": "user", "content": "Hello"}],
)
print(resp.choices[0].message.content)
```

To route to a different model, change only `model="..."` to another id from
`make list`. The URL, key, and code stay identical. Tool calling works through the
standard OpenAI `tools` parameter for any instance that has a `tool_parser` set in
`models.yaml`.

### Framework config (e.g. a tool-bench service)

If your service takes provider settings, map them like this:

```
provider   = openai
base_url    = http://<tunnel-host>:4000/v1
api_key     = <LITELLM_MASTER_KEY>
model       = <instance-id>   # e.g. lfm2.5-8b-a1b, swap per call
```

So yes: one endpoint, one key, and you swap models by name.