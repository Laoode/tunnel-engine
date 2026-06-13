<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=600&size=47&duration=1&pause=1000&color=000000&background=4add9c&center=true&vCenter=true&repeat=false&width=1200&lines=Tunnel+Engine+:+A+Unified+Gateway+for+All+LLM+Services" alt="Title" />
</h1>

<div align="center">
  <img src="https://github.com/Laoode/Tunnel-Engine/blob/main/assets/the-tunnel.png" alt="The Tunnel">
</div>

<p align="center">
  <b>LLM inference is fundamentally different. Standard backend logic won't cut it.</b>
</p>

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&pause=1800&color=4add9c&center=true&vCenter=true&width=1000&lines=Unified+API+Gateway+for+Multiple+LLM+Models;Blazing+Fast+Inference+via+vLLM+Continuous+Batching;Distributed+KV-Cache+Sharing+with+LMCache;Smart+Load+Balancing+%26+Automatic+Fallbacks;Built+for+Production-Ready+AI+Microservices" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12.11-3776AB?logo=python&logoColor=yellow" />
  <img src="https://img.shields.io/badge/CUDA-13.0-76B900?logo=nvidia&logoColor=green" />
  <img src="https://img.shields.io/badge/vLLM-Model%20Serving-4add9c" />
  <img src="https://img.shields.io/badge/LMCache-Global%20Cache-orange" />
  <img src="https://img.shields.io/badge/LiteLLM-Orchestration-white" />
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
                    ┌───────────────────────────────────┐
   Our services     │        LiteLLM Proxy :4000        │  ← single unified endpoint
   call this  ────▶ │  routing · fallback · balancing   │
                    └────────────┬────────────┬─────────┘
                                 │            │
                    ┌────────────▼──┐  ┌──────▼────────────┐
                    │  vLLM :8000   │  │   vLLM :8001      │
                    │  + LMCache    │  │   + LMCache       │
                    │  model-A      │  │   model-B         │
                    └───────────────┘  └───────────────────┘
                           │                    │
                    ┌──────▼────────────────────▼────────┐
                    │        RAM KV-Cache Store          │
                    │  (LMCache local, upgrades to Redis)│
                    └────────────────────────────────────┘
```

Install:
```bash
uv pip install -r tunnel-engine/requirements/dev.txt --torch-backend=auto
# Uninstall
# uv pip uninstall -r tunnel-engine/requirements/dev.txt -y
```

Running models:
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

Makefile:
```bash
# Validate the registry parses correctly
make check

# Generate derived configs (LiteLLM + LMCache yamls)
make generate

# Verify both instances (already running)
make health
```