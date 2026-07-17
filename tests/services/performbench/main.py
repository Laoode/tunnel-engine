"""Unified performance benchmark runner for the Tunnel Engine.

One entry point for every performance scenario (AIPerf profiles, long-doc KV
verification, TTFT sweeps), so feature work always ends with the same bench:

  python tests/services/performbench/main.py --list
  python tests/services/performbench/main.py                # default_suite
  python tests/services/performbench/main.py smoke goodput  # explicit set

Scenarios live in scenarios.yaml next to this file. Targets resolve from the
registry: `gateway` hits LiteLLM (with the master key), `instance:<id>` hits
the vLLM port directly. Artifacts go to results/<timestamp>/<scenario>/ and a
summary lands in RESULTS.md. Exits nonzero when a gated scenario fails.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tunnel.registry import TunnelRegistry, load_registry  # noqa: E402

_ROOT = Path(__file__).resolve().parents[3]
_HERE = Path(__file__).resolve().parent
SCENARIOS_PATH = _HERE / "scenarios.yaml"
RESULTS_DIR = _HERE / "results"
RESULTS_MD = _HERE / "RESULTS.md"
REGISTRY_PATH = str(_ROOT / "configs" / "models.yaml")


class ScenarioError(Exception):
    """A scenario definition is invalid or its target cannot be resolved."""


def resolve_target(registry: TunnelRegistry, target: str) -> tuple[str, str, str | None]:
    """Resolve a scenario target to (base_url, model_name, api_key).

    Args:
        registry: Loaded TunnelRegistry.
        target: "gateway" or "instance:<registry-id>".

    Returns:
        Tuple of (base_url without /v1, model name to send, api key or None).

    Raises:
        ScenarioError: on an unknown target or instance id.
    """
    if target == "gateway":
        key = registry.litellm.resolved_master_key
        if not key:
            raise ScenarioError(
                "gateway target needs LITELLM_MASTER_KEY (see .env.example)"
            )
        # Gateway scenarios address models by registry id; use the first
        # routable local instance unless the scenario overrides via args.
        return f"http://localhost:{registry.litellm.port}", "", key
    if target.startswith("instance:"):
        inst = registry.get_instance(target.split(":", 1)[1])
        if inst is None:
            raise ScenarioError(f"unknown instance in target '{target}'")
        model = inst.served_model_name or inst.model
        return f"http://localhost:{inst.port}", model, None
    raise ScenarioError(f"unknown target '{target}' (gateway | instance:<id>)")


def gateway_model(registry: TunnelRegistry) -> str:
    """Default model id for gateway scenarios: the first routable instance."""
    for inst in registry.instances:
        if not inst.internal:
            return inst.id
    raise ScenarioError("no routable instances in the registry")


def build_argv(
    kind: str, args: list[str], url: str, model: str, api_key: str | None,
    outdir: Path, tokenizer: str | None = None,
) -> tuple[list[str], dict]:
    """Build the subprocess argv + env for one scenario.

    Args:
        kind: "aiperf" | "long_doc_qa" | "ttft_estimator".
        args: Scenario args from scenarios.yaml, placeholders unresolved.
        url: Target base URL (no /v1).
        model: Model name for the target.
        api_key: Bearer key, or None for direct instance access.
        outdir: Artifact directory for this scenario run.
        tokenizer: HF tokenizer id for aiperf when `model` is not a HF id
            (gateway targets address models by registry id).

    Returns:
        (argv, env) ready for subprocess.run.

    Raises:
        ScenarioError: on an unknown scenario kind.
    """
    subst = {"{url}": url, "{model}": model, "{output}": str(outdir)}
    resolved = [subst.get(a, a) for a in args]
    env = os.environ.copy()

    if kind == "aiperf":
        argv = ["aiperf", "profile", "--model", model, "--url", url,
                "--artifact-dir", str(outdir)] + resolved
        if tokenizer:
            argv += ["--tokenizer", tokenizer]
        if api_key:
            argv += ["--api-key", api_key]
        return argv, env
    if kind == "long_doc_qa":
        argv = [sys.executable, str(_HERE / "long_doc_qa" / "long_doc_qa.py"),
                "--base-url", f"{url}/v1", "--model", model, "--json-output",
                "--output", str(outdir / "responses.log"),
                "--output-dir", str(outdir)] + resolved
        env["OPENAI_API_KEY"] = api_key or "sk-dummy"
        return argv, env
    if kind == "ttft_estimator":
        argv = [sys.executable, str(_HERE / "ttft-estimator" / "ttft-estimator.py"),
                "--base-url", f"{url}/v1", "--model", model,
                "--output-dir", str(outdir)] + resolved
        env["OPENAI_API_KEY"] = api_key or "sk-dummy"
        return argv, env
    raise ScenarioError(f"unknown scenario kind '{kind}'")


def run_scenario(name: str, spec: dict, registry: TunnelRegistry, run_dir: Path) -> dict:
    """Run one scenario and return its result row.

    Args:
        name: Scenario name from the catalog.
        spec: Scenario definition (kind/target/args/gate/description).
        registry: Loaded TunnelRegistry.
        run_dir: Parent directory for this whole bench run.

    Returns:
        Dict with name, description, gate, exit code, duration, artifact dir.
    """
    outdir = run_dir / name
    outdir.mkdir(parents=True, exist_ok=True)
    url, model, api_key = resolve_target(registry, spec["target"])
    tokenizer = None
    if spec["target"] == "gateway":
        model = spec.get("model", gateway_model(registry))
        # Gateway model names are registry ids, which aiperf can't resolve to
        # a HF tokenizer; local instances know their real model id.
        inst = registry.get_instance(model)
        if inst is not None:
            tokenizer = inst.model
    argv, env = build_argv(
        spec["kind"], spec.get("args", []), url, model, api_key, outdir, tokenizer
    )

    print(f"\n=== {name}: {spec.get('description', '')}")
    print(f"    target={spec['target']} model={model}")
    started = time.perf_counter()
    log_path = outdir / "run.log"
    with log_path.open("w") as log:
        proc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
    duration = time.perf_counter() - started
    status = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"    {status} in {duration:.1f}s -> {outdir.relative_to(_ROOT)}")
    if proc.returncode != 0:
        print(f"    last output lines ({log_path.relative_to(_ROOT)}):")
        for line in log_path.read_text().splitlines()[-5:]:
            print(f"      {line}")
    return {
        "name": name,
        "description": spec.get("description", ""),
        "gate": bool(spec.get("gate", False)),
        "returncode": proc.returncode,
        "duration_s": duration,
        "artifacts": str(outdir.relative_to(_ROOT)),
    }


def render_markdown(rows: list[dict], meta: dict) -> str:
    """Render the run summary as Markdown for RESULTS.md."""
    lines = [
        "# Performance Bench Results",
        "",
        "> Auto-generated by `tests/services/performbench/main.py`. Re-run to refresh.",
        "",
        f"- Timestamp (UTC): {meta['timestamp']}",
        f"- Registry: `{meta['registry']}`",
        f"- Artifacts: `{meta['run_dir']}`",
        "",
        "| Scenario | Status | Gate | Duration | Description |",
        "|---|---|---|---:|---|",
    ]
    for r in rows:
        status = "PASS" if r["returncode"] == 0 else f"FAIL ({r['returncode']})"
        gate = "yes" if r["gate"] else "-"
        lines.append(
            f"| {r['name']} | {status} | {gate} | {r['duration_s']:.1f}s "
            f"| {r['description']} |"
        )
    lines += [
        "",
        "Per-scenario metrics (AIPerf csv/json exports, long-doc TTFT stats,",
        "TTFT sweep plots) live under the artifacts directory listed above.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(_ROOT / ".env")
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenarios", nargs="*",
                        help="scenario names (default: the default_suite list)")
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    parsed = parser.parse_args()

    catalog = yaml.safe_load(SCENARIOS_PATH.read_text())
    scenarios: dict = catalog["scenarios"]

    if parsed.list:
        for name, spec in scenarios.items():
            gate = " [gate]" if spec.get("gate") else ""
            print(f"  {name:<14} {spec['kind']:<15} {spec['target']:<22}"
                  f"{gate}  {spec.get('description', '')}")
        return 0

    selected = parsed.scenarios or catalog.get("default_suite", list(scenarios))
    unknown = [s for s in selected if s not in scenarios]
    if unknown:
        print(f"ERROR: unknown scenario(s): {unknown}. Use --list.", file=sys.stderr)
        return 1

    registry = load_registry(REGISTRY_PATH)
    run_dir = RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    rows = []
    for name in selected:
        try:
            rows.append(run_scenario(name, scenarios[name], registry, run_dir))
        except ScenarioError as exc:
            print(f"ERROR: scenario '{name}': {exc}", file=sys.stderr)
            rows.append({"name": name, "description": str(exc), "gate": True,
                         "returncode": -1, "duration_s": 0.0, "artifacts": "-"})

    meta = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
        "registry": REGISTRY_PATH,
        "run_dir": str(run_dir.relative_to(_ROOT)),
    }
    RESULTS_MD.write_text(render_markdown(rows, meta))

    print("\n" + "=" * 70)
    for r in rows:
        status = "PASS" if r["returncode"] == 0 else "FAIL"
        gate = " [gate]" if r["gate"] else ""
        print(f"  {r['name']:<16} {status}{gate}  {r['duration_s']:.1f}s")
    print(f"\nWrote {RESULTS_MD.relative_to(_ROOT)}")

    gated_failures = [r for r in rows if r["gate"] and r["returncode"] != 0]
    return 1 if gated_failures else 0


if __name__ == "__main__":
    sys.exit(main())
