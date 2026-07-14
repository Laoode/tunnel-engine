"""Declarative sync of LiteLLM virtual keys from the registry.

The registry's services/tiers blocks are desired state; the LiteLLM DB is
actual state. Key secrets are returned exactly once by /key/generate, so this
module is the sole writer of the local secret store (.tunnel/keys.env).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

from tunnel.registry import ServiceConfig, TierConfig, TunnelRegistry

log = structlog.get_logger(__name__)

KEYS_ENV_PATH = Path(".tunnel/keys.env")

_KEYS_ENV_HEADER = (
    "# LiteLLM virtual key secrets, managed by `tunnel keys sync`. Do not edit.\n"
    "# This file is the only copy of these keys: if it is lost (or the DB is\n"
    "# wiped), `tunnel keys sync` regenerates keys, but consumers must be\n"
    "# re-issued the new secrets.\n"
)

# Fields compared between the desired spec and /key/info to detect drift.
_MANAGED_FIELDS = (
    "models", "tpm_limit", "rpm_limit", "max_budget", "budget_duration", "metadata",
)


@dataclass
class SyncReport:
    """Outcome of one sync_keys() run, service ids grouped by action taken."""

    created: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    regenerated: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)


def key_env_name(service_id: str) -> str:
    """Return the keys.env variable name for a service id.

    Args:
        service_id: Registry service id, e.g. "tools-bench".

    Returns:
        Env-style name, e.g. "SVC_TOOLS_BENCH".
    """
    return "SVC_" + service_id.upper().replace("-", "_")


def desired_key_spec(service: ServiceConfig, tier: TierConfig) -> dict:
    """Build the /key/generate payload for a service under its tier.

    Args:
        service: The service the key belongs to.
        tier: The resolved TierConfig for service.tier.

    Returns:
        Payload dict; `models` is omitted when empty (= all models allowed),
        budget fields are omitted when the tier has no budget.
    """
    spec: dict = {
        "key_alias": f"svc-{service.id}",
        "tpm_limit": tier.tpm_limit,
        "rpm_limit": tier.rpm_limit,
        "metadata": {
            "service": service.id,
            "tier": service.tier,
            "priority": tier.priority,
            "managed_by": "tunnel",
        },
    }
    if service.models:
        spec["models"] = service.models
    if tier.max_budget is not None:
        spec["max_budget"] = tier.max_budget
    if tier.budget_duration is not None:
        spec["budget_duration"] = tier.budget_duration
    return spec


def read_secrets(path: Path = KEYS_ENV_PATH) -> dict[str, str]:
    """Read the secret store into {env_name: key} (empty dict if missing).

    Args:
        path: Location of keys.env.

    Returns:
        Mapping of env-style names (see key_env_name) to key secrets.
    """
    if not path.exists():
        return {}
    secrets: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        secrets[name.strip()] = value.strip()
    return secrets


def write_secrets(secrets: dict[str, str], path: Path = KEYS_ENV_PATH) -> None:
    """Write the secret store atomically-ish with owner-only permissions.

    Args:
        secrets: Mapping of env-style names to key secrets.
        path: Location of keys.env.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{name}={value}" for name, value in sorted(secrets.items())]
    # O_CREAT with mode 0600 so the file is never world-readable, even briefly.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(_KEYS_ENV_HEADER + "\n".join(lines) + "\n")
    path.chmod(0o600)  # repair permissions of a pre-existing file too


def _quiet_httpx() -> None:
    """Stop httpx's request-line INFO logs: /key/info puts secrets in the URL."""
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _client(base_url: str, master_key: str) -> httpx.Client:
    """Build an authenticated httpx client for the proxy's admin key API.

    Args:
        base_url: Proxy base URL, e.g. "http://localhost:4000".
        master_key: Resolved LiteLLM master key.

    Returns:
        A configured httpx.Client (caller manages its lifetime).
    """
    return httpx.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {master_key}"},
        timeout=30.0,
    )


def _update_payload(secret: str, spec: dict) -> dict:
    """Build a /key/update payload that also RESETS relaxed fields.

    LiteLLM applies updates with exclude-unset semantics: a field missing
    from the payload keeps its old DB value. desired_key_spec() omits
    models/budget fields when unrestricted, so an update built from the spec
    alone could never relax a previously-set restriction. Explicit defaults
    for every managed field make relaxation propagate.

    Args:
        secret: The key being updated.
        spec: Output of desired_key_spec().

    Returns:
        Payload for POST /key/update.
    """
    return {
        "key": secret,
        "models": [],
        "max_budget": None,
        "budget_duration": None,
        **spec,
    }


def _key_info(client: httpx.Client, key: str) -> dict | None:
    """Fetch /key/info for a key; None when the key is unknown to the proxy.

    Args:
        client: Authenticated httpx client bound to the proxy base URL.
        key: The key secret to look up.

    Returns:
        The `info` dict from the response, or None on 400/404 (key not in
        the DB, e.g. after a wipe).

    Raises:
        httpx.HTTPStatusError: On auth or server errors, which must not be
            silently treated as "regenerate everything".
    """
    resp = client.get("/key/info", params={"key": key})
    if resp.status_code in (400, 404):
        # LiteLLM signals an unknown key as 400 or 404 depending on version.
        log.warning("key_not_in_db", status=resp.status_code, body=resp.text[:200])
        return None
    resp.raise_for_status()
    return resp.json().get("info") or {}


def _drifted(current: dict, spec: dict) -> bool:
    """True when any managed field differs between DB state and desired spec."""
    for fld in _MANAGED_FIELDS:
        if fld == "models":
            desired = spec.get("models", [])
            actual = current.get("models") or []
        elif fld == "metadata":
            # Compare only the keys we manage. Note: /key/update REPLACES the
            # stored metadata wholesale; these keys are tunnel-managed, so
            # foreign metadata on them is not preserved by design.
            desired = spec.get("metadata") or {}
            current_md = current.get("metadata") or {}
            actual = {k: current_md.get(k) for k in desired}
        else:
            desired = spec.get(fld)
            actual = current.get(fld)
        if actual != desired:
            return True
    return False


def sync_keys(
    registry: TunnelRegistry,
    base_url: str,
    master_key: str,
    *,
    prune: bool = False,
    secrets_path: Path = KEYS_ENV_PATH,
) -> SyncReport:
    """Reconcile LiteLLM virtual keys with the registry's services/tiers.

    Known secret -> /key/info; drift -> /key/update; unknown to the DB
    (wiped) -> regenerate. No local secret -> /key/generate and persist.
    With prune=True, keys.env entries absent from the registry are deleted
    via /key/delete and dropped from the file.

    Args:
        registry: Loaded registry with non-empty services.
        base_url: Proxy base URL, e.g. "http://localhost:4000".
        master_key: Resolved LiteLLM master key (admin auth).
        prune: Delete keys for services no longer in the registry.
        secrets_path: Override of the keys.env location (tests).

    Returns:
        SyncReport with service ids grouped by action.

    Raises:
        httpx.HTTPError: On connection, auth, or server errors.
    """
    _quiet_httpx()
    report = SyncReport()
    secrets = read_secrets(secrets_path)

    # finally-write so a failure partway through never orphans keys already
    # created in the proxy DB (their secrets would be lost otherwise).
    try:
        with _client(base_url, master_key) as client:
            for service in registry.services:
                tier = registry.tiers[service.tier]
                spec = desired_key_spec(service, tier)
                env_name = key_env_name(service.id)
                secret = secrets.get(env_name)

                info = _key_info(client, secret) if secret else None
                if info is not None:
                    if _drifted(info, spec):
                        resp = client.post("/key/update",
                                           json=_update_payload(secret, spec))
                        resp.raise_for_status()
                        report.updated.append(service.id)
                        log.info("key_updated", service=service.id, tier=service.tier)
                    else:
                        report.unchanged.append(service.id)
                    continue

                resp = client.post("/key/generate", json=spec)
                resp.raise_for_status()
                secrets[env_name] = resp.json()["key"]
                if secret:
                    report.regenerated.append(service.id)
                    log.warning("key_regenerated", service=service.id,
                                reason="secret unknown to proxy DB")
                else:
                    report.created.append(service.id)
                    log.info("key_created", service=service.id, tier=service.tier)

            if prune:
                registry_envs = {key_env_name(s.id) for s in registry.services}
                for env_name in sorted(set(secrets) - registry_envs):
                    resp = client.post("/key/delete",
                                       json={"keys": [secrets[env_name]]})
                    resp.raise_for_status()
                    del secrets[env_name]
                    report.pruned.append(env_name)
                    log.info("key_pruned", entry=env_name)
    finally:
        write_secrets(secrets, secrets_path)
    return report


def fetch_key_overview(
    registry: TunnelRegistry,
    base_url: str,
    master_key: str,
    secrets_path: Path = KEYS_ENV_PATH,
) -> list[dict]:
    """Collect per-service key state for `tunnel keys list`.

    Args:
        registry: Loaded registry.
        base_url: Proxy base URL.
        master_key: Resolved LiteLLM master key.
        secrets_path: Override of the keys.env location (tests).

    Returns:
        One row per service: id, tier, models, spend (None when the service
        has no synced key yet).
    """
    _quiet_httpx()
    secrets = read_secrets(secrets_path)
    rows: list[dict] = []
    with _client(base_url, master_key) as client:
        for service in registry.services:
            secret = secrets.get(key_env_name(service.id))
            info = _key_info(client, secret) if secret else None
            rows.append({
                "service": service.id,
                "tier": service.tier,
                "models": service.models or ["(all)"],
                "spend": (info or {}).get("spend"),
                "synced": info is not None,
            })
    return rows
