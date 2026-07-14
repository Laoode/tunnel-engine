"""Traffic mixes for the loadgen harness: LISO / LILO / SILO / SISO.

Long input ~4-8k tokens (prefill-bound), short ~50-200. Long output
max_tokens 512-1024 (decode-bound), short 32-64. Prompts are unique per
request (salted) so vLLM's prefix cache cannot short-circuit prefill.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

_TOPICS = [
    "distributed systems", "cache coherence", "vector databases", "GPU scheduling",
    "network protocols", "consensus algorithms", "memory hierarchies", "load balancing",
]

# (input paragraph range, max_tokens range); one paragraph ~= 100 tokens
MIXES: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {
    "LISO": ((40, 80), (32, 64)),
    "LILO": ((40, 80), (512, 1024)),
    "SILO": ((1, 2), (512, 1024)),
    "SISO": ((1, 2), (32, 64)),
}


@dataclass(frozen=True)
class Request:
    """One generated load request."""
    mix: str
    prompt: str
    max_tokens: int


def _paragraph(rng: random.Random, i: int, salt: str) -> str:
    """One ~100-token filler paragraph, unique per (salt, rng draw, i)."""
    topic = rng.choice(_TOPICS)
    a, b = rng.randint(10, 99), rng.randint(100, 999)
    return (
        f"[run {salt}] Section {i}: In the study of {topic}, document {a}-{i} "
        f"describes trade-offs unique to configuration {b}. The subsystem "
        f"allocates {a + i} units of budget and tracks {i + 5} counters. Observed "
        f"latency for path {a}:{i} was {b % 87} milliseconds under nominal load, "
        f"degrading to {b % 331} milliseconds at saturation. Operators tune the "
        f"{topic} layer toward throughput while preserving tail latency. "
    )


def make_request(mix: str, rng: random.Random, salt: str) -> Request:
    """Build one request for a mix.

    Args:
        mix: One of MIXES.
        rng: Seeded random source (reproducible workloads).
        salt: Run-unique string keeping prompts prefix-cache-cold.

    Returns:
        A Request with a unique prompt and sampled max_tokens.
    """
    (p_lo, p_hi), (t_lo, t_hi) = MIXES[mix]
    n_paras = rng.randint(p_lo, p_hi)
    body = "".join(_paragraph(rng, i, salt) for i in range(n_paras))
    prompt = f"{body}\nSummarize the operational trade-offs described above."
    return Request(mix=mix, prompt=prompt, max_tokens=rng.randint(t_lo, t_hi))
