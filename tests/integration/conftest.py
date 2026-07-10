"""Integration test fixtures for the LiteLLM proxy.

All integration tests auto-skip when the proxy is not reachable.
Start it with: make proxy
"""
import pytest
import httpx

from tunnel.registry import TunnelRegistry, load_registry

# Load .env so LITELLM_MASTER_KEY (referenced as os.environ/... in the registry)
# is available to resolve the proxy auth key. No-op if python-dotenv is absent.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


@pytest.fixture(scope="session")
def registry() -> TunnelRegistry:
    """Loaded registry — reads configs/models.yaml relative to cwd."""
    return load_registry("configs/models.yaml")


@pytest.fixture(scope="session")
def proxy_url(registry: TunnelRegistry) -> str:
    return f"http://localhost:{registry.litellm.port}"


@pytest.fixture(scope="session")
def auth_headers(registry: TunnelRegistry) -> dict[str, str]:
    return {"Authorization": f"Bearer {registry.litellm.resolved_master_key}"}


@pytest.fixture(scope="session", autouse=True)
def require_proxy(proxy_url: str, auth_headers: dict[str, str]) -> None:
    """Skip the entire integration suite when the proxy is not running."""
    try:
        resp = httpx.get(f"{proxy_url}/health", headers=auth_headers, timeout=2.0)
        if resp.status_code != 200:
            pytest.skip(f"Proxy at {proxy_url} returned HTTP {resp.status_code}")
    except Exception as exc:
        pytest.skip(f"Proxy not reachable at {proxy_url}: {exc} — run `make proxy`")
