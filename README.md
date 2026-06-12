<h1 align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Chakra+Petch&weight=600&size=47&duration=1&pause=1000&color=000000&background=4add9c&center=true&vCenter=true&repeat=false&width=1200&lines=Tunnel+Engine+:+A+Unified+Gateway+for+All+LLM+Services" alt="Title" />
</h1>

<div align="center">
  <img src="https://github.com/Laoode/Tunnel-Engine/blob/main/assets/the-tunnel.png" alt="The Tunnel">
</div>

<p align="center">
  <img src="https://img.shields.io/badge/vLLM-Model%20Serving-1f4b99" />
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