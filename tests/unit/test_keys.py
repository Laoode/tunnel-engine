"""
Tests for tunnel/gateway/keys.py: desired_key_spec shapes, the secret store,
and sync_keys reconciliation flows against pytest-httpx mocks.
"""
import httpx
import pytest

from tunnel.gateway.keys import (
    SyncReport,
    desired_key_spec,
    key_env_name,
    read_secrets,
    sync_keys,
    write_secrets,
)
from tunnel.registry import ServiceConfig, TierConfig, TunnelRegistry

BASE_URL = "http://localhost:4000"
MASTER = "sk-master"


def _registry(services=None) -> TunnelRegistry:
    return TunnelRegistry.model_validate({
        "instances": [
            {"id": "model-a", "model": "org/model-a", "port": 8000,
             "gpu_memory_utilization": 0.10},
            {"id": "model-b", "model": "org/model-b", "port": 8001,
             "gpu_memory_utilization": 0.10},
        ],
        "litellm": {"port": 4000, "master_key": "sk-test"},
        "tiers": {
            "free": {"priority": 2, "rpm_limit": 60, "tpm_limit": 20000,
                     "max_budget": 5.0, "budget_duration": "30d"},
            "enterprise": {"priority": 0, "rpm_limit": 1200, "tpm_limit": 500000},
        },
        "services": services if services is not None else [
            {"id": "dev", "tier": "free", "models": ["model-a"]},
        ],
    })


def _free_tier() -> TierConfig:
    return _registry().tiers["free"]


def test_key_env_name_uppercases_and_replaces_dashes():
    assert key_env_name("tools-bench") == "SVC_TOOLS_BENCH"


def test_desired_key_spec_full_shape():
    svc = ServiceConfig(id="dev", tier="free", models=["model-a"])
    spec = desired_key_spec(svc, _free_tier())
    assert spec == {
        "key_alias": "svc-dev",
        "tpm_limit": 20000,
        "rpm_limit": 60,
        "max_budget": 5.0,
        "budget_duration": "30d",
        "models": ["model-a"],
        "metadata": {"service": "dev", "tier": "free", "priority": 2,
                     "managed_by": "tunnel"},
    }


def test_desired_key_spec_omits_models_when_empty():
    svc = ServiceConfig(id="dev", tier="free")
    assert "models" not in desired_key_spec(svc, _free_tier())


def test_desired_key_spec_omits_budget_when_tier_has_none():
    svc = ServiceConfig(id="ent", tier="enterprise")
    spec = desired_key_spec(svc, _registry().tiers["enterprise"])
    assert "max_budget" not in spec
    assert "budget_duration" not in spec


def test_secrets_roundtrip_and_permissions(tmp_path):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-1", "SVC_OTHER": "sk-2"}, path)
    assert read_secrets(path) == {"SVC_DEV": "sk-1", "SVC_OTHER": "sk-2"}
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_read_secrets_missing_file_returns_empty(tmp_path):
    assert read_secrets(tmp_path / "absent.env") == {}


def _info_response(spec: dict) -> dict:
    """Build a /key/info body whose managed fields match the desired spec."""
    return {"key": "sk-existing", "info": {
        "key_alias": spec["key_alias"],
        "models": spec.get("models", []),
        "tpm_limit": spec["tpm_limit"],
        "rpm_limit": spec["rpm_limit"],
        "max_budget": spec.get("max_budget"),
        "budget_duration": spec.get("budget_duration"),
        "metadata": {**spec["metadata"], "litellm_internal": True},
        "spend": 0.0,
    }}


def test_sync_creates_key_when_no_secret(tmp_path, httpx_mock):
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/generate", json={"key": "sk-new"},
    )
    report = sync_keys(_registry(), BASE_URL, MASTER,
                       secrets_path=tmp_path / "keys.env")
    assert report.created == ["dev"]
    assert read_secrets(tmp_path / "keys.env") == {"SVC_DEV": "sk-new"}
    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == f"Bearer {MASTER}"


def test_sync_unchanged_when_info_matches(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-existing"}, path)
    registry = _registry()
    spec = desired_key_spec(registry.services[0], registry.tiers["free"])
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing",
        json=_info_response(spec),
    )
    report = sync_keys(registry, BASE_URL, MASTER, secrets_path=path)
    assert report.unchanged == ["dev"]
    assert report.created == report.updated == []


def test_sync_updates_on_drift(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-existing"}, path)
    registry = _registry()
    spec = desired_key_spec(registry.services[0], registry.tiers["free"])
    drifted = _info_response(spec)
    drifted["info"]["rpm_limit"] = 999
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing", json=drifted,
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/update", json={"key": "sk-existing"},
    )
    report = sync_keys(registry, BASE_URL, MASTER, secrets_path=path)
    assert report.updated == ["dev"]
    update = [r for r in httpx_mock.get_requests() if r.url.path == "/key/update"]
    assert len(update) == 1


def test_sync_regenerates_when_db_wiped(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-lost"}, path)
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-lost",
        status_code=404, json={"error": "key not found"},
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/generate", json={"key": "sk-fresh"},
    )
    report = sync_keys(_registry(), BASE_URL, MASTER, secrets_path=path)
    assert report.regenerated == ["dev"]
    assert read_secrets(path) == {"SVC_DEV": "sk-fresh"}


def test_sync_raises_on_auth_error_instead_of_regenerating(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-existing"}, path)
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing",
        status_code=401, json={"error": "invalid master key"},
    )
    with pytest.raises(httpx.HTTPStatusError):
        sync_keys(_registry(), BASE_URL, MASTER, secrets_path=path)
    # The stale secret must not be overwritten on an auth failure.
    assert read_secrets(path) == {"SVC_DEV": "sk-existing"}


def test_sync_prunes_orphaned_entries(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-existing", "SVC_GONE": "sk-orphan"}, path)
    registry = _registry()
    spec = desired_key_spec(registry.services[0], registry.tiers["free"])
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing",
        json=_info_response(spec),
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/delete", json={"deleted": 1},
    )
    report = sync_keys(registry, BASE_URL, MASTER, prune=True, secrets_path=path)
    assert report.pruned == ["SVC_GONE"]
    assert read_secrets(path) == {"SVC_DEV": "sk-existing"}


def test_sync_without_prune_keeps_orphans(tmp_path, httpx_mock):
    path = tmp_path / "keys.env"
    write_secrets({"SVC_DEV": "sk-existing", "SVC_GONE": "sk-orphan"}, path)
    registry = _registry()
    spec = desired_key_spec(registry.services[0], registry.tiers["free"])
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing",
        json=_info_response(spec),
    )
    report = sync_keys(registry, BASE_URL, MASTER, secrets_path=path)
    assert report.pruned == []
    assert "SVC_GONE" in read_secrets(path)


def test_sync_report_defaults_empty():
    report = SyncReport()
    assert (report.created, report.updated, report.unchanged,
            report.regenerated, report.pruned) == ([], [], [], [], [])


def test_update_resets_relaxed_fields(tmp_path, httpx_mock):
    # Key in the DB is restricted to model-a with a budget; the registry now
    # says unrestricted. The update payload must explicitly reset those fields
    # (LiteLLM keeps omitted fields unchanged, so omission can never relax).
    path = tmp_path / "keys.env"
    write_secrets({"SVC_ENT": "sk-existing"}, path)
    registry = _registry(services=[{"id": "ent", "tier": "enterprise"}])
    spec = desired_key_spec(registry.services[0], registry.tiers["enterprise"])
    restricted = _info_response(spec)
    restricted["info"].update(models=["model-a"], max_budget=9.0,
                              budget_duration="30d")
    httpx_mock.add_response(
        method="GET", url=f"{BASE_URL}/key/info?key=sk-existing", json=restricted,
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/update", json={"key": "sk-existing"},
    )
    report = sync_keys(registry, BASE_URL, MASTER, secrets_path=path)
    assert report.updated == ["ent"]
    import json
    update = [r for r in httpx_mock.get_requests() if r.url.path == "/key/update"][0]
    body = json.loads(update.content)
    assert body["models"] == []
    assert body["max_budget"] is None
    assert body["budget_duration"] is None


def test_partial_failure_still_persists_created_secrets(tmp_path, httpx_mock):
    # First service's key is created, second explodes: the first secret must
    # land in keys.env anyway or the created key is orphaned in the DB.
    path = tmp_path / "keys.env"
    registry = _registry(services=[
        {"id": "one", "tier": "free", "models": ["model-a"]},
        {"id": "two", "tier": "free", "models": ["model-b"]},
    ])
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/generate", json={"key": "sk-one"},
    )
    httpx_mock.add_response(
        method="POST", url=f"{BASE_URL}/key/generate",
        status_code=500, json={"error": "boom"},
    )
    with pytest.raises(httpx.HTTPStatusError):
        sync_keys(registry, BASE_URL, MASTER, secrets_path=path)
    assert read_secrets(path) == {"SVC_ONE": "sk-one"}
