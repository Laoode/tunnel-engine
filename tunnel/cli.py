"""Tunnel Engine CLI: single entry point for all dev operations.

Usage (each command also has a Makefile target of the same name):
  python -m tunnel.cli serve   <instance-id>   Launch a vLLM instance
  python -m tunnel.cli health                  Poll all instance health + GPU memory
  python -m tunnel.cli generate                Rebuild all derived configs
  python -m tunnel.cli list                    List registered instances
  python -m tunnel.cli proxy                   Start the LiteLLM proxy
  python -m tunnel.cli start                   Wait for instances, then start proxy
  python -m tunnel.cli up                      Launch all instances, health-gate, start proxy
  python -m tunnel.cli stop    <instance-id>   Stop ONE instance, leave the rest + proxy running
  python -m tunnel.cli down                    Stop every instance and the proxy
  python -m tunnel.cli keys    sync [--prune]  Reconcile LiteLLM virtual keys with the registry
  python -m tunnel.cli keys    list            Per-service key status + spend
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import json
from pathlib import Path

import httpx

from tunnel.cache.lmcache_config import write_lmcache_configs
from tunnel.gateway.config_builder import write_litellm_config
from tunnel.gateway.keys import KEYS_ENV_PATH, fetch_key_overview, sync_keys
from tunnel.health.checker import check_all, collect_gpu_stats, format_report
from tunnel.logging import configure_logging
from tunnel.orchestrator import (
    PID_DIR,
    adopt_instance,
    find_listening_pid,
    find_listening_pids,
    is_alive,
    launch_instance,
    read_pid,
    stop_instance,
)
from tunnel.registry import InstanceConfig, TunnelRegistry, load_registry
from tunnel.startup import DEFAULT_TIMEOUT_S, wait_for_all, wait_for_one

LITELLM_CONFIG = "configs/litellm/config.yaml"


def registry_path() -> str:
    """Return the active registry file path from TUNNEL_REGISTRY.

    Read at call time (not import time) so tests can monkeypatch per-case.

    Returns:
        Path to the registry YAML; "configs/models.yaml" when the env is unset.
    """
    return os.environ.get("TUNNEL_REGISTRY", "configs/models.yaml")


def _require_instance(registry: TunnelRegistry, instance_id: str) -> InstanceConfig:
    """Return the instance with the given id, or exit(1) listing available ids.

    Args:
        registry: Loaded TunnelRegistry.
        instance_id: The registry id to look up.

    Returns:
        The matching InstanceConfig. Does not return on a miss.
    """
    inst = registry.get_instance(instance_id)
    if inst is None:
        available = [i.id for i in registry.instances]
        print(
            f"ERROR: Instance '{instance_id}' not found in {registry_path()}.\n"
            f"  Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    return inst


def _stop_untracked(inst_id: str, port: int, pid: int) -> None:
    """Adopt an untracked listener into a pidfile, then stop it.

    Args:
        inst_id: Instance id (or "litellm-proxy") to record the pid under.
        port: The port the process is listening on, for the report line.
        pid: The live untracked pid.
    """
    adopt_instance(inst_id, pid)
    outcome = stop_instance(inst_id)
    print(
        f".  {inst_id}  {outcome} (untracked on :{port}, pid {pid})",
        file=sys.stderr,
    )


def build_serve_command(inst: InstanceConfig) -> list[str]:
    """Build the `vllm serve` argv for an instance.

    Pure function: no printing, no env handling. The only I/O is a
    `Path.exists()` check on `chat_template` so the flag is omitted (rather
    than passed as a broken path) when the file is missing — callers that
    want a WARN for that case should check existence themselves before
    calling this.

    Args:
        inst: Validated InstanceConfig.

    Returns:
        Full argv list for `os.execvpe`.
    """
    cmd = [
        "vllm", "serve", inst.model,
        "--port",                    str(inst.port),
        "--tensor-parallel-size",    str(inst.tensor_parallel_size),
        "--gpu-memory-utilization",  str(inst.gpu_memory_utilization),
        "--max-model-len",           str(inst.max_model_len),
        "--dtype",                   inst.dtype,
        "--default-chat-template-kwargs", json.dumps({"enable_thinking": inst.enable_thinking})
    ]

    if inst.lora.enabled:
        cmd.append("--enable-lora")
        for module in inst.lora.modules:
            cmd += ["--lora-modules", f"{module.name}={module.path}"]

    if inst.attention_backend:
        cmd += ["--attention-backend", inst.attention_backend]

    if inst.chat_template and Path(inst.chat_template).exists():
        cmd += ["--chat-template", str(Path(inst.chat_template))]

    if inst.quantization:
        cmd += ["--quantization", inst.quantization]

    if inst.served_model_name:
        cmd += ["--served-model-name", inst.served_model_name]

    if inst.tool_parser:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", inst.tool_parser]

    if inst.reasoning_parser:
        cmd += ["--reasoning-parser", inst.reasoning_parser]

    if inst.lmcache.enabled:
        # Register LMCache as vLLM's KV connector so KV blocks are offloaded to /
        # loaded from the LMCache tiers (CPU/disk/redis). Without this flag vLLM
        # ignores LMCACHE_CONFIG_FILE entirely and only its in-GPU prefix cache runs.
        # kv_both = this instance both stores and retrieves (single-node offload).
        cmd += ["--kv-transfer-config",
                json.dumps({"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"})]

    cmd.extend(inst.extra_args)

    return cmd


def _inject_redis_env(inst: InstanceConfig, env: dict) -> None:
    """Assemble LMCACHE_REMOTE_URL/SERDE into `env` from LMCACHE_REDIS_HOST/PORT.

    Redis host/port are environment-specific (never committed); LMCache merges
    these env vars over the config file. Warns and degrades to the local CPU
    tier when the host is unset.

    Args:
        inst: The InstanceConfig being launched.
        env: Environment dict passed to os.execvpe, mutated in place.
    """
    if inst.lmcache.backend != "redis":
        return
    host = env.get("LMCACHE_REDIS_HOST")
    if not host:
        print(
            f"WARN: instance '{inst.id}' uses lmcache.backend=redis but "
            "LMCACHE_REDIS_HOST is unset. Falling back to the local CPU tier only.",
            file=sys.stderr,
        )
        return
    port = env.get("LMCACHE_REDIS_PORT", "6379")
    env["LMCACHE_REMOTE_URL"] = f"redis://{host}:{port}"
    env["LMCACHE_REMOTE_SERDE"] = inst.lmcache.remote_serde


def cmd_serve(instance_id: str) -> None:
    """Launch a vLLM instance by registry ID.

    Handles LoRA, custom Jinja2 templates, LMCache env var,
    tensor-parallel-size, and extra_args passthrough.
    Replaces the current process via os.execvpe — no return.
    """
    registry = load_registry(registry_path())
    inst = _require_instance(registry, instance_id)

    if inst.chat_template and not Path(inst.chat_template).exists():
        print(
            f"WARN: chat_template '{inst.chat_template}' not found — "
            "model will use its default template.",
            file=sys.stderr,
        )

    cmd = build_serve_command(inst)

    env = os.environ.copy()
    if inst.lmcache.enabled:
        lmcache_cfg_path = Path(f"configs/lmcache/{inst.id}.yaml")
        if lmcache_cfg_path.exists():
            env["LMCACHE_CONFIG_FILE"] = str(lmcache_cfg_path)
            _inject_redis_env(inst, env)
        else:
            print(
                f"WARN: LMCache config not found at '{lmcache_cfg_path}'. "
                "Run `make generate` first. Launching without LMCache.",
                file=sys.stderr,
            )

    print(f">>  {' '.join(cmd)}\n", file=sys.stderr)
    os.execvpe(cmd[0], cmd, env)


def cmd_health() -> None:
    """Poll all vLLM instances and print a health + GPU memory report."""
    registry = load_registry(registry_path())
    results = asyncio.run(check_all(registry))
    gpu_stats = collect_gpu_stats()
    print(format_report(results, gpu_stats=gpu_stats))
    if not all(h.is_healthy for h in results):
        sys.exit(1)


def cmd_generate() -> None:
    """Rebuild all derived configs from configs/models.yaml."""
    registry = load_registry(registry_path())

    master_key = registry.litellm.master_key
    if master_key and not master_key.startswith("os.environ/"):
        print(
            f"WARN: litellm.master_key is a literal value in {registry_path()}. "
            f"It will be written into {LITELLM_CONFIG} and committed to git in "
            "plaintext.\n  Use 'os.environ/LITELLM_MASTER_KEY' and set the secret "
            "in .env (dev) or the container env (prod) instead.",
            file=sys.stderr,
        )

    litellm_path = write_litellm_config(registry, LITELLM_CONFIG)
    print(f"  LiteLLM config    -> {litellm_path}")

    lmcache_paths = write_lmcache_configs(registry)
    for p in lmcache_paths:
        print(f"  LMCache config    -> {p}")

    print(f"\n  {len(registry.instances)} instance(s) registered:")
    for inst in registry.instances:
        flags = []
        if inst.lora.enabled:
            flags.append(f"LoRA x{len(inst.lora.modules)}")
        if inst.chat_template:
            flags.append(f"template:{Path(inst.chat_template).name}")
        if inst.tensor_parallel_size > 1:
            flags.append(f"TP={inst.tensor_parallel_size}")
        if inst.fallbacks:
            flags.append(f"fallback->{inst.fallbacks}")
        if inst.quantization:
            flags.append(f"quant:{inst.quantization}")
        if inst.tool_parser:
            flags.append(f"tools:{inst.tool_parser}")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  . {inst.id:<24}  :{inst.port}  {inst.model}{flag_str}")

    prom = " [prometheus: on]" if registry.litellm.prometheus else ""
    print(f"\n  LiteLLM proxy -> :{registry.litellm.port}{prom}")


def cmd_list() -> None:
    """List all registered instances."""
    registry = load_registry(registry_path())
    print(f"\n{'ID':<24}  {'PORT':<6}  {'GPU':<6}  {'TP':<4}  MODEL")
    print("-" * 72)
    for inst in registry.instances:
        fb = f"  -> {inst.fallbacks}" if inst.fallbacks else ""
        print(
            f"{inst.id:<24}  {inst.port:<6}  "
            f"{inst.gpu_memory_utilization:<6}  {inst.tensor_parallel_size:<4}  "
            f"{inst.model}{fb}"
        )
    if registry.remote_models:
        print(f"\n{'REMOTE ID':<24}  {'PROVIDER':<10}  UPSTREAM  ->  API BASE")
        print("-" * 72)
        for rm in registry.remote_models:
            print(
                f"{rm.id:<24}  {rm.provider:<10}  "
                f"{rm.upstream_model}  ->  {rm.api_base}"
            )

    print(f"\n  Proxy -> :{registry.litellm.port}  "
          f"({registry.litellm.routing_strategy})\n")


def cmd_proxy() -> None:
    """Start the LiteLLM proxy. Requires `make generate` to have been run first."""
    config_path = Path(LITELLM_CONFIG)
    if not config_path.exists():
        print(
            f"ERROR: LiteLLM config not found at '{config_path}'.\n"
            "  Run `make generate` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    registry = load_registry(registry_path())
    if registry.services and not os.environ.get("DATABASE_URL"):
        print(
            "WARN: services are defined in the registry but DATABASE_URL is "
            "unset; the proxy's virtual-key DB layer will fail to start. "
            "Run `make db-up` and set DATABASE_URL in .env.",
            file=sys.stderr,
        )
    cmd = [
        "litellm",
        "--config", str(config_path),
        "--port",   str(registry.litellm.port),
    ]
    print(f">>  {' '.join(cmd)}\n", file=sys.stderr)
    os.execvpe(cmd[0], cmd, os.environ.copy())


def cmd_start(args: list[str]) -> None:
    """Wait for all vLLM instances to become healthy, then exec the LiteLLM proxy.

    Prevents the LiteLLM startup-cooldown failure: if LiteLLM starts before vLLM
    finishes loading (~30-120s), it marks models as failed and enters 60s cooldown.
    This blocks until all instances are healthy, then hands off to cmd_proxy.

    Args:
        args: Optional ["--timeout", "<seconds>"].
    """
    parser = argparse.ArgumentParser(prog="tunnel start")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-instance health wait timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parsed = parser.parse_args(args)

    registry = load_registry(registry_path())
    result = asyncio.run(wait_for_all(registry, timeout_s=parsed.timeout))

    if not result.ready:
        print(
            f"ERROR: {len(result.failed_instances)} instance(s) did not become healthy "
            f"within {parsed.timeout}s: {result.failed_instances}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"All {len(registry.instances)} instance(s) healthy "
        f"after {result.elapsed_s}s. Starting proxy...",
        file=sys.stderr,
    )
    cmd_proxy()  # exec()s into LiteLLM — no return


def cmd_up(args: list[str]) -> None:
    """Launch every registered instance, health-gate, then exec the proxy.

    Sequential by default: concurrent vLLM startups corrupt each other's GPU
    memory profiling and crash with "No available memory for the cache blocks"
    (--parallel restores launch-all-then-gate). Live tracked pids are skipped;
    untracked port listeners are adopted instead of double-launched. A failed
    health wait doesn't block the rest of the fleet; the final pid-aware
    wait_for_all() gate still catches it and exits nonzero.

    Args:
        args: Optional ["--timeout", "<seconds>", "--parallel"].
    """
    parser = argparse.ArgumentParser(prog="tunnel up")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=(
            f"Per-instance health wait timeout in seconds (default: {DEFAULT_TIMEOUT_S}). "
            "Applies per instance, not to the whole fleet, in both sequential and "
            "--parallel mode."
        ),
    )
    parser.add_argument(
        "--parallel", action="store_true",
        help=(
            "Launch all instances immediately, then health-gate them together, "
            "instead of the sequential default. Only safe for fleets whose models "
            "profile GPU memory fast enough to avoid concurrent-startup corruption."
        ),
    )
    parsed = parser.parse_args(args)

    registry = load_registry(registry_path())

    launched: dict[str, int] = {}  # instance id -> pid, for dead-process fail-fast below

    for inst in registry.instances:
        pid = read_pid(inst.id)
        if pid is not None and is_alive(pid):
            print(f".  {inst.id}  already running (pid {pid})", file=sys.stderr)
            launched[inst.id] = pid
            continue
        # Probed per iteration, not snapshotted upfront: sequential health-gating
        # can take minutes, and a listener appearing mid-run must still be
        # adopted instead of double-launched onto its port.
        listening_pid = find_listening_pid(inst.port)
        if listening_pid is not None:
            adopt_instance(inst.id, listening_pid)
            print(
                f".  {inst.id}  adopted untracked process on :{inst.port} "
                f"(pid {listening_pid})",
                file=sys.stderr,
            )
            launched[inst.id] = listening_pid
            continue

        pid = launch_instance(inst)
        print(f".  {inst.id}  launched (pid {pid}) -> logs/{inst.id}.log", file=sys.stderr)
        launched[inst.id] = pid

        if not parsed.parallel:
            ok = asyncio.run(wait_for_one(inst, timeout_s=parsed.timeout, pid=pid))
            if not ok:
                print(
                    f"ERROR: {inst.id} did not become healthy within {parsed.timeout}s "
                    f"-- check logs/{inst.id}.log",
                    file=sys.stderr,
                )

    result = asyncio.run(wait_for_all(registry, timeout_s=parsed.timeout, pids=launched))

    if not result.ready:
        print(
            f"ERROR: {len(result.failed_instances)} instance(s) did not become healthy "
            f"within {parsed.timeout}s: {result.failed_instances}\n"
            f"  Check logs/<id>.log for the failing instance(s).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        f"All {len(registry.instances)} instance(s) healthy "
        f"after {result.elapsed_s}s. Starting proxy...",
        file=sys.stderr,
    )
    cmd_proxy()  # exec()s into LiteLLM — no return


def cmd_down(_: list[str]) -> None:
    """Stop everything this engine started, tracked or not, and free the GPU.

    Three passes so nothing leaks a GPU allocation or a port: tracked pidfiles
    (ground truth for what `up` launched, even if the registry changed since),
    untracked listeners on registry instance ports (`make serve` never writes
    a pidfile), and the LiteLLM proxy. Untracked pids are adopted first so
    stop_instance's SIGTERM-then-SIGKILL escalation applies to everything.
    """
    registry = load_registry(registry_path())
    listeners = find_listening_pids(
        {inst.port for inst in registry.instances} | {registry.litellm.port}
    )
    stopped = 0

    for pidfile in sorted(PID_DIR.glob("*.pid")):
        inst_id = pidfile.stem
        outcome = stop_instance(inst_id)
        print(f".  {inst_id}  {outcome}", file=sys.stderr)
        stopped += 1

    for inst in registry.instances:
        listening_pid = listeners.get(inst.port)
        if listening_pid is not None and is_alive(listening_pid):
            _stop_untracked(inst.id, inst.port, listening_pid)
            stopped += 1

    proxy_pid = listeners.get(registry.litellm.port)
    if proxy_pid is not None and is_alive(proxy_pid):
        _stop_untracked("litellm-proxy", registry.litellm.port, proxy_pid)
        stopped += 1

    if stopped == 0:
        print(".  nothing running", file=sys.stderr)


def cmd_stop(instance_id: str) -> None:
    """Stop ONE instance, leaving the rest of the fleet and the proxy running.

    Resolves the target like `cmd_down`: tracked pidfile plus any untracked
    listener on its port. The proxy keeps advertising the stopped model in
    /v1/models until the registry is edited and regenerated.

    Args:
        instance_id: The registry id of the instance to stop.
    """
    registry = load_registry(registry_path())
    inst = _require_instance(registry, instance_id)

    stopped = False
    if read_pid(inst.id) is not None:
        outcome = stop_instance(inst.id)
        print(f".  {inst.id}  {outcome}", file=sys.stderr)
        stopped = True

    listening_pid = find_listening_pid(inst.port)
    if listening_pid is not None and is_alive(listening_pid):
        _stop_untracked(inst.id, inst.port, listening_pid)
        stopped = True

    if not stopped:
        print(f".  {inst.id}  not running", file=sys.stderr)


def cmd_keys(args: list[str]) -> None:
    """Manage LiteLLM virtual keys for registry services (sync / list).

    Requires the proxy to be running: the keys API is served on litellm.port
    and is authenticated with the master key.

    Args:
        args: ["sync", "--prune"] or ["list"].
    """
    parser = argparse.ArgumentParser(prog="tunnel keys")
    parser.add_argument("action", choices=["sync", "list"])
    parser.add_argument(
        "--prune", action="store_true",
        help="delete keys whose service is no longer in the registry",
    )
    parsed = parser.parse_args(args)

    registry = load_registry(registry_path())
    if not registry.services:
        print("No services defined in the registry; nothing to do.", file=sys.stderr)
        return
    master_key = registry.litellm.resolved_master_key
    if not master_key:
        _die("master key unavailable: set LITELLM_MASTER_KEY (see .env.example)")
    base_url = f"http://localhost:{registry.litellm.port}"

    try:
        if parsed.action == "sync":
            report = sync_keys(registry, base_url, master_key, prune=parsed.prune)
            for action in ("created", "updated", "regenerated", "unchanged", "pruned"):
                ids = getattr(report, action)
                if ids:
                    print(f"  {action:<12} {', '.join(ids)}", file=sys.stderr)
            print(f"  secrets -> {KEYS_ENV_PATH}", file=sys.stderr)
        else:
            rows = fetch_key_overview(registry, base_url, master_key)
            print(f"\n{'SERVICE':<20}  {'TIER':<12}  {'SPEND $':<12}  MODELS")
            print("-" * 72)
            for row in rows:
                spend = f"{row['spend']:.6f}" if row["spend"] is not None else "-"
                synced = "" if row["synced"] else "  (not synced)"
                print(
                    f"{row['service']:<20}  {row['tier']:<12}  {spend:<10}  "
                    f"{', '.join(row['models'])}{synced}"
                )
            print()
    except httpx.HTTPError as exc:
        _die(
            f"keys {parsed.action} failed: {exc}\n"
            f"  Is the proxy running on :{registry.litellm.port} (make up)?"
        )


_COMMANDS = {
    "serve":    lambda args: cmd_serve(args[0] if args else _die("serve requires <instance-id>")),
    "health":   lambda _: cmd_health(),
    "generate": lambda _: cmd_generate(),
    "list":     lambda _: cmd_list(),
    "proxy":    lambda _: cmd_proxy(),
    "start":    lambda args: cmd_start(args),
    "up":       lambda args: cmd_up(args),
    "stop":     lambda args: cmd_stop(args[0] if args else _die("stop requires <instance-id>")),
    "down":     lambda args: cmd_down(args),
    "keys":     lambda args: cmd_keys(args),
}


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def _load_env() -> None:
    """Load .env into the process environment before anything reads os.environ.

    Values already present in the real environment win (override=False), so a
    container or secret manager that exports LITELLM_MASTER_KEY / HF_TOKEN
    takes precedence over the dev .env file. No-op if python-dotenv is absent.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def main() -> None:
    _load_env()
    configure_logging(level=os.getenv("TUNNEL_LOG_LEVEL", "INFO"))
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__, file=sys.stderr)
        print(f"Available: {list(_COMMANDS)}", file=sys.stderr)
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
