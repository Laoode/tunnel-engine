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

Or via Makefile:
  make serve ID=qwen-0.8b
  make health
  make generate
  make proxy
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from tunnel.cache.lmcache_config import write_lmcache_configs
from tunnel.gateway.config_builder import write_litellm_config
from tunnel.health.checker import check_all, collect_gpu_stats, format_report
from tunnel.registry import load_registry

REGISTRY_PATH = "configs/models.yaml"
LITELLM_CONFIG = "configs/litellm/config.yaml"


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

    cmd = [
        "vllm", "serve", inst.model,
        "--port",                    str(inst.port),
        "--tensor-parallel-size",    str(inst.tensor_parallel_size),
        "--gpu-memory-utilization",  str(inst.gpu_memory_utilization),
        "--max-model-len",           str(inst.max_model_len),
        "--dtype",                   inst.dtype,
    ]

    if inst.lora.enabled:
        cmd.append("--enable-lora")
        for module in inst.lora.modules:
            cmd += ["--lora-modules", f"{module.name}={module.path}"]

    if inst.chat_template:
        template_path = Path(inst.chat_template)
        if not template_path.exists():
            print(
                f"WARN: chat_template '{inst.chat_template}' not found — "
                "model will use its default template.",
                file=sys.stderr,
            )
        else:
            cmd += ["--chat-template", str(template_path)]

    cmd.extend(inst.extra_args)

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
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  . {inst.id:<24}  :{inst.port}  {inst.model}{flag_str}")

    print(f"\n  LiteLLM proxy will listen on :{registry.litellm.port}")


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


_COMMANDS = {
    "serve":    lambda args: cmd_serve(args[0] if args else _die("serve requires <instance-id>")),
    "health":   lambda _: cmd_health(),
    "generate": lambda _: cmd_generate(),
    "list":     lambda _: cmd_list(),
    "proxy":    lambda _: cmd_proxy(),
}


def _die(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(__doc__, file=sys.stderr)
        print(f"Available: {list(_COMMANDS)}", file=sys.stderr)
        sys.exit(1)
    _COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
