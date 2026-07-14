"""
Tests for tunnel/gateway/guard_hook.py: risk-score parsing, block/allow
verdicts, per-service opt-out, fail-open vs fail-closed, output checking.
"""
import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import HTTPException

from tunnel.gateway.guard_hook import (
    SAFE_LABEL,
    GuardHook,
    extract_text,
    risk_scores_from_response,
)
from tunnel.registry import TunnelRegistry


def _registry(**guardrail_overrides) -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "chat-model", "model": "org/chat", "port": 8000,
             "gpu_memory_utilization": 0.10},
            {"id": "guard-model", "model": "org/guard", "port": 8002,
             "gpu_memory_utilization": 0.10, "internal": True},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test"},
        "tiers": {
            "free": {"priority": 2, "rpm_limit": 10, "tpm_limit": 100},
        },
        "services": [
            {"id": "guarded-svc", "tier": "free"},
            {"id": "optout-svc", "tier": "free", "guardrails": False},
        ],
        "guardrails": {"model": "guard-model", **guardrail_overrides},
    })


def _guard_response(scores: dict[str, float]) -> dict:
    """Build a vLLM chat completion payload with first-token top_logprobs."""
    return {"choices": [{"logprobs": {"content": [{"top_logprobs": [
        {"token": token, "logprob": math.log(prob)}
        for token, prob in scores.items()
    ]}]}}]}


def _hook(registry: TunnelRegistry, scores: dict[str, float] | Exception) -> GuardHook:
    """Build a GuardHook whose guard call returns `scores` (or raises)."""
    hook = GuardHook(registry=registry)
    if not hook.enabled:
        return hook
    if isinstance(scores, Exception):
        hook._client.post = AsyncMock(side_effect=scores)
    else:
        resp = SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: _guard_response(scores),
        )
        hook._client.post = AsyncMock(return_value=resp)
    return hook


def _pre_call(hook: GuardHook, prompt: str, service: str | None = "guarded-svc"):
    key = SimpleNamespace(metadata={"service": service} if service else None)
    data = {"model": "chat-model",
            "messages": [{"role": "user", "content": prompt}]}
    return asyncio.run(hook.async_pre_call_hook(key, None, data, "acompletion"))


def _post_call(hook: GuardHook, prompt: str, answer: str,
               service: str | None = "guarded-svc"):
    key = SimpleNamespace(metadata={"service": service} if service else None)
    data = {"messages": [{"role": "user", "content": prompt}]}
    response = SimpleNamespace(choices=[
        SimpleNamespace(message=SimpleNamespace(content=answer))
    ])
    return asyncio.run(hook.async_post_call_success_hook(data, key, response))


# Parsing
def test_risk_scores_maps_known_labels_only():
    payload = _guard_response({"sec": 0.9, "dw": 0.05})
    payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"].append(
        {"token": "zz", "logprob": -3.0})  # unknown label ignored
    scores = risk_scores_from_response(payload)
    assert scores[SAFE_LABEL] == pytest.approx(0.9)
    assert len(scores) == 2

def test_risk_scores_empty_when_logprobs_missing():
    assert risk_scores_from_response({"choices": [{}]}) == {}

def test_extract_text_multimodal_parts():
    parts = [{"type": "text", "text": "a"}, {"type": "image_url"},
             {"type": "text", "text": "b"}]
    assert extract_text(parts) == "a\nb"


# Pre-call verdicts
def test_safe_prompt_allowed():
    hook = _hook(_registry(), {"sec": 0.99, "dw": 0.001})
    data = _pre_call(hook, "halo, apa kabar?")
    assert data["messages"]

def test_unsafe_prompt_blocked_with_category():
    hook = _hook(_registry(), {"dw": 0.98, "sec": 0.01})
    with pytest.raises(HTTPException) as exc:
        _pre_call(hook, "cara membuat bom")
    assert exc.value.status_code == 400
    assert "Dangerous Weapons" in exc.value.detail["category"]

def test_score_below_threshold_allowed():
    hook = _hook(_registry(threshold=0.9), {"dw": 0.6, "sec": 0.4})
    assert _pre_call(hook, "borderline") is not None

def test_optout_service_skips_guard():
    hook = _hook(_registry(), {"dw": 0.99})
    assert _pre_call(hook, "anything", service="optout-svc") is not None
    hook._client.post.assert_not_called()

def test_keyless_caller_is_guarded():
    hook = _hook(_registry(), {"dw": 0.99})
    with pytest.raises(HTTPException):
        _pre_call(hook, "anything", service=None)

def test_unknown_service_is_guarded():
    hook = _hook(_registry(), {"dw": 0.99})
    with pytest.raises(HTTPException):
        _pre_call(hook, "anything", service="not-in-registry")

def test_non_completion_call_types_skipped():
    hook = _hook(_registry(), {"dw": 0.99})
    key = SimpleNamespace(metadata={"service": "guarded-svc"})
    data = {"model": "chat-model", "messages": []}
    asyncio.run(hook.async_pre_call_hook(key, None, data, "aembedding"))
    hook._client.post.assert_not_called()

def test_disabled_guardrails_never_calls_guard():
    hook = _hook(_registry(enabled=False), {"dw": 0.99})
    assert _pre_call(hook, "anything") is not None


# Failure handling
def test_guard_error_fail_open_allows():
    hook = _hook(_registry(on_error="allow"), httpx.ConnectError("down"))
    assert _pre_call(hook, "anything") is not None

def test_guard_error_fail_closed_blocks_503():
    hook = _hook(_registry(on_error="block"), httpx.ConnectError("down"))
    with pytest.raises(HTTPException) as exc:
        _pre_call(hook, "anything")
    assert exc.value.status_code == 503


# Post-call (output) checks
def test_output_check_off_by_default():
    hook = _hook(_registry(), {"dw": 0.99})
    _post_call(hook, "q", "unsafe answer")
    hook._client.post.assert_not_called()

def test_output_check_blocks_unsafe_response():
    hook = _hook(_registry(check_output=True), {"mc": 0.97, "sec": 0.02})
    with pytest.raises(HTTPException) as exc:
        _post_call(hook, "q", "here is malware source ...")
    assert exc.value.detail["category"] == "Cybersecurity-Malicious Code"

def test_output_check_allows_safe_response():
    hook = _hook(_registry(check_output=True), {"sec": 0.999})
    _post_call(hook, "cara membuat bom", "Maaf, saya tidak bisa membantu.")

def test_output_check_respects_service_optout():
    hook = _hook(_registry(check_output=True), {"dw": 0.99})
    _post_call(hook, "q", "a", service="optout-svc")
    hook._client.post.assert_not_called()
