"""
tests/services/kv_cache/dataset.py
==================================
Deterministic long-prefix prompt generator for the KV-cache benchmark.

Each "test case" is a long, unique document (the shared *prefix* whose KV is
expensive to compute) followed by a short question. The benchmark sends each
case twice: a cold round populates the cache, a warm round replays the exact
same prompt so the KV of the long prefix can be reused instead of recomputed.

Design choices that make the benchmark meaningful:
  - Long prefix so prefill compute dominates and KV reuse clearly pays off.
  - Each document is unique (per seed) so there is no accidental cross-prompt
    prefix sharing within a round.
  - A ``salt`` makes every run's prompts unique, guaranteeing the cold round is
    genuinely cold even though LMCache persists KV inside the running vLLM
    process across benchmark runs.
"""
from __future__ import annotations

from dataclasses import dataclass

_TOPICS = [
    "distributed systems", "cache coherence", "vector databases", "GPU scheduling",
    "network protocols", "consensus algorithms", "memory hierarchies", "load balancing",
    "query planning", "stream processing",
]


@dataclass(frozen=True)
class PromptCase:
    """One KV-cache test case: a long shared prefix plus a short question."""
    id: str
    system: str      # the long, cache-worthy prefix
    user: str        # the short trailing question
    answer_hint: str  # the value a correct answer should contain (sanity check)


def _paragraph(seed: int, i: int, salt: str) -> str:
    """One deterministic, unique-per-document paragraph (~100 tokens)."""
    topic = _TOPICS[(seed + i) % len(_TOPICS)]
    return (
        f"[run {salt}] Section {i}: In the study of {topic}, engineers repeatedly find "
        f"that document {seed}-{i} describes trade-offs unique to configuration "
        f"{seed * 7 + i}. The subsystem allocates {100 + seed + i} units of budget and "
        f"tracks {50 + i} distinct counters. Observed latency for path {seed}:{i} was "
        f"{3 * seed + i} milliseconds under nominal load, degrading to {9 * seed + i} "
        f"milliseconds at saturation. Operators tune the {topic} layer toward throughput "
        f"while preserving tail-latency guarantees for tenant {seed}. "
    )


def build_prompts(n_prompts: int, n_paras: int, salt: str) -> list[PromptCase]:
    """Build ``n_prompts`` unique long-prefix test cases.

    Args:
        n_prompts: Number of distinct documents (each sent cold then warm).
        n_paras: Paragraphs per document (~100 tokens each). Larger => longer
            prefix => bigger KV-reuse win, but must fit the model's context
            window after any chat-template overhead.
        salt: Run-unique string mixed into every paragraph so the cold round is
            genuinely cold.

    Returns:
        A list of PromptCase, one per document.
    """
    cases: list[PromptCase] = []
    for seed in range(n_prompts):
        doc = "".join(_paragraph(seed, i, salt) for i in range(n_paras))
        # The nominal latency for path seed:3 is "3*seed + 3" ms (see _paragraph).
        answer = f"{3 * seed + 3}"
        cases.append(PromptCase(
            id=f"doc-{seed}",
            system=f"You are analyzing internal document {seed}.\n\n{doc}",
            user=f"In one short sentence, what is the nominal latency for path {seed}:3?",
            answer_hint=answer,
        ))
    return cases
