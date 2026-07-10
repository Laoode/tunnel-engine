"""
tests/services/tools/main.py
=============================
Standalone tool-calling smoke test against the Tunnel Engine gateway.

Not a pytest test module -- pytest's default discovery only collects
`test_*.py` / `*_test.py`, so `main.py` is never picked up by `make
test-unit` or `make test`. Run it directly once the proxy and instances
are up:

  python tests/services/tools/main.py

Loads the registry (configs/models.yaml) to get the LiteLLM port, master
key, and the list of registered instance IDs -- no hardcoded model list --
then sends one tool-calling chat completion per instance through the
gateway. A model passes only if it actually returns a structured
`tool_calls` entry for the `get_weather` function with valid JSON
arguments containing "city"; see docs/PLAN.md item H for why this check
matters (a missing `tool_parser` makes vLLM silently return an empty
response instead of a tool call).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

# Standalone script, not run via `python -m`: put the repo root on sys.path
# so `tunnel.registry` is importable regardless of the caller's cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tunnel.registry import load_registry  # noqa: E402

REGISTRY_PATH = str(Path(__file__).resolve().parents[3] / "configs" / "models.yaml")
REQUEST_TIMEOUT_S = 60.0

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get the weather for.",
                },
            },
            "required": ["city"],
        },
    },
}

_PROMPT = "What is the weather in Jakarta right now?"


def _check_instance(base_url: str, master_key: str | None, instance_id: str) -> tuple[bool, str]:
    """Send one tool-calling chat completion and verify the model used the tool.

    Args:
        base_url: Base URL of the LiteLLM proxy, e.g. "http://localhost:4000".
        master_key: LiteLLM master key for the Authorization header, or None.
        instance_id: Registered instance ID to route the request to.

    Returns:
        A (passed, reason) tuple. `reason` is empty on pass, a human-readable
        failure description otherwise.
    """
    body = {
        "model": instance_id,
        "messages": [{"role": "user", "content": _PROMPT}],
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "max_tokens": 256,
    }
    headers = {"Authorization": f"Bearer {master_key}"}

    try:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=REQUEST_TIMEOUT_S,
        )
    except httpx.HTTPError as exc:
        return False, f"request error: {exc}"

    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"

    try:
        message = resp.json()["choices"][0]["message"]
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"unexpected response shape: {exc}"

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        return False, "no tool_calls in response"

    function = tool_calls[0].get("function", {})
    if function.get("name") != "get_weather":
        return False, f"unexpected function name: {function.get('name')!r}"

    try:
        arguments = json.loads(function.get("arguments", ""))
    except json.JSONDecodeError as exc:
        return False, f"arguments not valid JSON: {exc}"

    if "city" not in arguments:
        return False, f"arguments missing 'city': {arguments}"

    return True, ""


def main() -> int:
    """Run the tool-calling smoke test against every registered instance.

    Returns:
        0 if every instance passed, 1 otherwise.
    """
    # Load .env so the os.environ/LITELLM_MASTER_KEY reference resolves to the
    # same key the proxy authenticates with. No-op if python-dotenv is absent.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    registry = load_registry(REGISTRY_PATH)
    base_url = f"http://localhost:{registry.litellm.port}"

    all_passed = True
    for inst in registry.instances:
        passed, reason = _check_instance(base_url, registry.litellm.resolved_master_key, inst.id)
        if passed:
            print(f"PASS {inst.id}")
        else:
            print(f"FAIL {inst.id} {reason}")
            all_passed = False

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
