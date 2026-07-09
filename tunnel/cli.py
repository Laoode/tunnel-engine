"""
Tunnel Engine CLI
=================
Single entry point for all dev operations.

Usage:
  python -m tunnel.cli serve   <instance-id>   Launch a vLLM instance
  python -m tunnel.cli health                  Poll all instance health + GPU memory
  python -m tunnel.cli generate                Rebuild all derived configs
  python -m tunnel.cli list                    List registered instances
  python -m tunnel.cli proxy                   Start the LiteLLM proxy
  python -m tunnel.cli start                   Wait for instances, then start proxy
  python -m tunnel.cli up                      Launch all instances, health-gate, start proxy
  python -m tunnel.cli down                    Stop all tracked vLLM instances

Or via Makefile:
  make serve ID=qwen-0.8b
  make health
  make generate
  make proxy
  make start
  make up
  make down
"""
from __future__ import annotations

import argparse
import asyncio
from math import e
import os
import sys
import json
from pathlib import Path

from tunnel.cache.lmcache_config import write_lmcache_configs
from tunnel.gateway.config_builder import write_litellm_config
from tunnel.health.checker import check_all, collect_gpu_stats, format_report
from tunnel.logging import configure_logging
from tunnel.orchestrator import is_alive, launch_instance, read_pid, stop_instance
from tunnel.registry import InstanceConfig, load_registry
from tunnel.startup import DEFAULT_TIMEOUT_S, wait_for_all

REGISTRY_PATH = "configs/models.yaml"
LITELLM_CONFIG = "configs/litellm/config.yaml"


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

    cmd.extend(inst.extra_args)

    return cmd


def cmd_serve(instance_id: str) -> None:
    """Launch a vLLM instance by registry ID.

    Handles LoRA, custom Jinja2 templates, LMCache env var,
    tensor-parallel-size, and extra_args passthrough.
    Replaces the current process via os.execvpe — no return.
    """
    registry = load_registry(REGISTRY_PATH)
    inst = registry.get_instance(instance_id)

    if inst is None:
        available = [i.id for i in registry.instances]
        print(
            f"ERROR: Instance '{instance_id}' not found in {REGISTRY_PATH}.\n"
            f"  Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)

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
    registry = load_registry(REGISTRY_PATH)
    results = asyncio.run(check_all(registry))
    gpu_stats = collect_gpu_stats()
    print(format_report(results, gpu_stats=gpu_stats))
    if not all(h.is_healthy for h in results):
        sys.exit(1)


def cmd_generate() -> None:
    """Rebuild all derived configs from configs/models.yaml."""
    registry = load_registry(REGISTRY_PATH)

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
    registry = load_registry(REGISTRY_PATH)
    print(f"\n{'ID':<24}  {'PORT':<6}  {'GPU':<6}  {'TP':<4}  MODEL")
    print("-" * 72)
    for inst in registry.instances:
        fb = f"  -> {inst.fallbacks}" if inst.fallbacks else ""
        print(
            f"{inst.id:<24}  {inst.port:<6}  "
            f"{inst.gpu_memory_utilization:<6}  {inst.tensor_parallel_size:<4}  "
            f"{inst.model}{fb}"
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

    registry = load_registry(REGISTRY_PATH)
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

    registry = load_registry(REGISTRY_PATH)
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
    """Launch every registered instance in the background, health-gate, then exec the proxy.

    Skips instances that already have a live tracked pid. Instances that fail to
    become healthy within the timeout are left running (they may still be
    loading) — the user decides whether to wait longer or investigate the log.

    Args:
        args: Optional ["--timeout", "<seconds>"].
    """
    parser = argparse.ArgumentParser(prog="tunnel up")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_S,
        help=f"Per-instance health wait timeout in seconds (default: {DEFAULT_TIMEOUT_S})",
    )
    parsed = parser.parse_args(args)

    registry = load_registry(REGISTRY_PATH)

    for inst in registry.instances:
        pid = read_pid(inst.id)
        if pid is not None and is_alive(pid):
            print(f".  {inst.id}  already running (pid {pid})", file=sys.stderr)
            continue
        pid = launch_instance(inst)
        print(f".  {inst.id}  launched (pid {pid}) -> logs/{inst.id}.log", file=sys.stderr)

    result = asyncio.run(wait_for_all(registry, timeout_s=parsed.timeout))

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
    """Stop every tracked vLLM instance cleanly (SIGTERM, then SIGKILL if needed)."""
    registry = load_registry(REGISTRY_PATH)

    for inst in registry.instances:
        outcome = stop_instance(inst.id)
        print(f".  {inst.id}  {outcome}", file=sys.stderr)

    print(
        "\nNote: the LiteLLM proxy is not tracked by pidfiles — if it's running "
        "in another terminal, Ctrl-C it there.",
        file=sys.stderr,
    )


_COMMANDS = {
    "serve":    lambda args: cmd_serve(args[0] if args else _die("serve requires <instance-id>")),
    "health":   lambda _: cmd_health(),
    "generate": lambda _: cmd_generate(),
    "list":     lambda _: cmd_list(),
    "proxy":    lambda _: cmd_proxy(),
    "start":    lambda args: cmd_start(args),
    "up":       lambda args: cmd_up(args),
    "down":     lambda args: cmd_down(args),
}


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    configure_logging(level=os.getenv("TUNNEL_LOG_LEVEL", "INFO"))
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__, file=sys.stderr)
        print(f"Available: {list(_COMMANDS)}", file=sys.stderr)
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
