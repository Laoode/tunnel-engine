"""Integration tests for the LiteLLM proxy.

Requires live vLLM instances + the proxy running.
Auto-skipped when proxy is unreachable (handled in conftest.py).

Run: make test-integration
"""
import pytest
import httpx

pytestmark = pytest.mark.integration


def _chat(
    proxy_url: str,
    auth_headers: dict[str, str],
    model_id: str,
    content: str = "Reply with one word: ready",
    max_tokens: int = 8,
) -> httpx.Response:
    """Send a minimal chat completion request to the proxy.

    Args:
        proxy_url: Base URL of the LiteLLM proxy.
        auth_headers: Authorization header dict.
        model_id: Registered model ID to route to.
        content: User message content.
        max_tokens: Cap on generated tokens — keep small to stay fast.

    Returns:
        The raw httpx Response.
    """
    return httpx.post(
        f"{proxy_url}/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
        },
        timeout=60.0,
    )


def test_proxy_health(proxy_url: str, auth_headers: dict[str, str]) -> None:
    resp = httpx.get(f"{proxy_url}/health", headers=auth_headers, timeout=5.0)
    assert resp.status_code == 200


def test_model_list_contains_all_registered_instances(
    proxy_url: str,
    auth_headers: dict[str, str],
    registry,
) -> None:
    resp = httpx.get(f"{proxy_url}/v1/models", headers=auth_headers, timeout=5.0)
    assert resp.status_code == 200
    proxy_model_ids = {m["id"] for m in resp.json()["data"]}
    registered_ids = {inst.id for inst in registry.instances}
    missing = registered_ids - proxy_model_ids
    assert not missing, f"These registered models are absent from proxy: {missing}"


def test_all_instances_route_and_respond(
    proxy_url: str,
    auth_headers: dict[str, str],
    registry,
) -> None:
    """Each registered instance should accept a request and return a non-empty reply."""
    for inst in registry.instances:
        resp = _chat(proxy_url, auth_headers, inst.id)
        assert resp.status_code == 200, (
            f"Instance '{inst.id}' returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
        body = resp.json()
        assert "choices" in body, f"No 'choices' in response for '{inst.id}'"
        assert body["choices"][0]["message"]["content"].strip(), (
            f"Empty response content for '{inst.id}'"
        )


def test_response_includes_usage_stats(
    proxy_url: str,
    auth_headers: dict[str, str],
    registry,
) -> None:
    first_id = registry.instances[0].id
    resp = _chat(proxy_url, auth_headers, first_id)
    assert resp.status_code == 200
    usage = resp.json().get("usage", {})
    assert usage.get("completion_tokens", 0) > 0, "Expected non-zero completion_tokens"
    assert usage.get("prompt_tokens", 0) > 0, "Expected non-zero prompt_tokens"


def test_unknown_model_returns_error(
    proxy_url: str,
    auth_headers: dict[str, str],
) -> None:
    resp = _chat(proxy_url, auth_headers, "this-model-does-not-exist")
    assert resp.status_code in (400, 404, 500), (
        f"Expected 4xx/5xx for unknown model, got {resp.status_code}"
    )
