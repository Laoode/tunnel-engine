"""Loads and validates the model registry from configs/models.yaml.

Foundation layer: nothing else reads raw YAML; go through load_registry(),
which merges defaults into instance blocks and fail-fast validates via Pydantic.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


def _find_duplicates(items: list) -> list:
    """Return the sorted distinct values that appear more than once in items."""
    return sorted(v for v, n in Counter(items).items() if n > 1)


class LoRAModule(BaseModel):
    name: str
    path: str


class LoRAConfig(BaseModel):
    enabled: bool = False
    modules: list[LoRAModule] = Field(default_factory=list)

    @model_validator(mode="after")
    def modules_required_when_enabled(self) -> "LoRAConfig":
        if self.enabled and not self.modules:
            raise ValueError(
                "lora.enabled=true but lora.modules is empty. "
                "Define at least one module or set enabled: false."
            )
        return self


class LMCacheInstanceConfig(BaseModel):
    enabled: bool = True
    backend: str = "cpu"
    max_cache_size_gb: int = 20
    chunk_size: int = 256
    remote_serde: str = "naive"  # only used when backend == "redis"

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"cpu", "disk", "redis"}
        if v not in allowed:
            raise ValueError(f"lmcache.backend must be one of {allowed}, got '{v}'")
        return v

    @field_validator("remote_serde")
    @classmethod
    def validate_remote_serde(cls, v: str) -> str:
        allowed = {"naive", "cachegen"}
        if v not in allowed:
            raise ValueError(
                f"lmcache.remote_serde must be one of {allowed}, got '{v}'"
            )
        return v


class ModelCost(BaseModel):
    """Per-token pricing used for spend tracking through the gateway.

    Local GPU models use a synthetic amortized rate; remote models use the
    provider's list price. Emitted into LiteLLM as cost-per-token.
    """
    input_per_mtok: float   # USD per 1M input tokens
    output_per_mtok: float  # USD per 1M output tokens

    @field_validator("input_per_mtok", "output_per_mtok")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"cost must be >= 0, got {v}")
        return v


class InstanceConfig(BaseModel):
    id: str
    model: str
    port: int
    gpu_memory_utilization: float
    description: str = ""
    # Internal instances (e.g. the guardrail classifier) are launched and
    # health-gated like any other, but excluded from the LiteLLM model_list:
    # clients cannot route to them, and services/fallbacks may not reference them.
    internal: bool = False
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    attention_backend: Optional[str] = None
    max_model_len: Optional[int] = None
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    lmcache: LMCacheInstanceConfig = Field(default_factory=LMCacheInstanceConfig)
    chat_template: Optional[str] = None
    extra_args: list[str] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)  # ordered fallback instance IDs
    tool_parser: Optional[str] = None
    reasoning_parser: Optional[str] = None
    quantization: Optional[str] = None
    served_model_name: Optional[str] = None
    enable_thinking: bool = False
    cost: Optional[ModelCost] = None
    scheduling_policy: Optional[str] = None  # None = vLLM default (fcfs)

    @field_validator("scheduling_policy")
    @classmethod
    def validate_scheduling_policy(cls, v: Optional[str]) -> Optional[str]:
        allowed = {"fcfs", "priority"}
        if v is not None and v not in allowed:
            raise ValueError(
                f"scheduling_policy must be one of {allowed}, got '{v}'"
            )
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not (1024 <= v <= 65535):
            raise ValueError(f"port {v} is out of valid range [1024, 65535]")
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def validate_gpu_mem(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"gpu_memory_utilization must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("tensor_parallel_size")
    @classmethod
    def validate_tp_size(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tensor_parallel_size must be >= 1, got {v}")
        return v

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/health"


class GPUConfig(BaseModel):
    budget: float = 0.90  # max allowed sum of instance gpu_memory_utilization

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"gpu.budget must be in (0.0, 1.0], got {v}")
        return v


class LiteLLMGatewayConfig(BaseModel):
    port: int = 4000
    master_key: Optional[str] = None
    routing_strategy: str = "least-busy"
    prometheus: bool = False  # enable /metrics endpoint (requires prometheus-client)

    @field_validator("routing_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {"least-busy", "simple-shuffle", "latency-based-routing"}
        if v not in allowed:
            raise ValueError(
                f"litellm.routing_strategy must be one of {allowed}, got '{v}'"
            )
        return v

    @property
    def resolved_master_key(self) -> Optional[str]:
        """Master key with an ``os.environ/`` reference resolved from the environment.

        Mirrors LiteLLM's own resolution so proxy clients (health checks, smoke
        tests) can authenticate with the same value. Returns the literal key
        unchanged, or None when unset / the referenced env var is absent.
        """
        if self.master_key and self.master_key.startswith("os.environ/"):
            return os.environ.get(self.master_key.split("/", 1)[1])
        return self.master_key


class RemoteModelConfig(BaseModel):
    """An OpenAI-compatible upstream API exposed through the LiteLLM gateway.

    Unlike InstanceConfig, a remote model runs no local process: it has no port,
    no GPU footprint, and is never launched or health-gated. It only contributes
    an entry to the generated LiteLLM model_list, routed to a hosted API.
    """
    id: str                       # gateway model_name clients call, e.g. "deepseek-v4-pro"
    upstream_model: str           # provider-side model id, e.g. "deepseek-v4-pro"
    api_base: str                 # e.g. "https://api.deepseek.com"
    api_key_env: str              # env var NAME holding the key, e.g. "DEEPSEEK_API_KEY"
    provider: str = "openai"      # LiteLLM prefix; DeepSeek is OpenAI-compatible
    description: str = ""
    cost: Optional[ModelCost] = None


class GuardrailsConfig(BaseModel):
    """Gateway-level content safety via an XGuard-style classifier instance.

    The referenced instance must be marked internal: it is called directly by
    the guard hook (bypassing LiteLLM routing) and must not be client-routable.
    Input (pre-call) checking is always on when enabled; response checking is
    opt-in via check_output and applies to non-streaming responses only.
    """
    enabled: bool = True
    model: str                   # registry instance id serving the guard model
    threshold: float = 0.5       # block when any non-safe risk score >= threshold
    check_output: bool = False   # also classify responses (post-call, non-streaming)
    on_error: str = "allow"      # guard call failure: "allow" (fail-open) | "block"
    timeout_s: float = 2.0       # guard call timeout; on_error applies past this

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"guardrails.threshold must be in (0.0, 1.0], got {v}")
        return v

    @field_validator("on_error")
    @classmethod
    def validate_on_error(cls, v: str) -> str:
        allowed = {"allow", "block"}
        if v not in allowed:
            raise ValueError(f"guardrails.on_error must be one of {allowed}, got '{v}'")
        return v

    @field_validator("timeout_s")
    @classmethod
    def validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"guardrails.timeout_s must be > 0, got {v}")
        return v


class TierConfig(BaseModel):
    """Rate/budget limits and scheduling priority for one service tier."""
    priority: int           # vLLM scheduling priority: lower = served earlier
    rpm_limit: int
    tpm_limit: int
    max_budget: Optional[float] = None       # USD; None = unlimited
    budget_duration: Optional[str] = None    # LiteLLM reset window, e.g. "30d"

    @field_validator("priority")
    @classmethod
    def validate_priority(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"tier priority must be >= 0, got {v}")
        return v

    @field_validator("rpm_limit", "tpm_limit")
    @classmethod
    def validate_limits(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"tier rpm/tpm limits must be >= 1, got {v}")
        return v


class ServiceConfig(BaseModel):
    """A consumer of the gateway, issued its own LiteLLM virtual key.

    `models` restricts which registry ids the key may call; empty = all.
    """
    id: str
    tier: str
    models: list[str] = Field(default_factory=list)
    description: str = ""
    # None = follow the global guardrails toggle; False = opt this service out.
    guardrails: Optional[bool] = None


class TunnelRegistry(BaseModel):
    instances: list[InstanceConfig]
    remote_models: list[RemoteModelConfig] = Field(default_factory=list)
    litellm: LiteLLMGatewayConfig = Field(default_factory=LiteLLMGatewayConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    tiers: dict[str, TierConfig] = Field(default_factory=dict)
    services: list[ServiceConfig] = Field(default_factory=list)
    guardrails: Optional[GuardrailsConfig] = None

    @property
    def routable_ids(self) -> set[str]:
        """Model ids clients may call through the gateway (internal excluded)."""
        return {inst.id for inst in self.instances if not inst.internal} | {
            rm.id for rm in self.remote_models
        }

    @model_validator(mode="after")
    def validate_no_port_collisions(self) -> "TunnelRegistry":
        dupes = _find_duplicates([inst.port for inst in self.instances])
        if dupes:
            raise ValueError(f"Duplicate instance ports found: {dupes}")
        return self

    @model_validator(mode="after")
    def validate_no_id_collisions(self) -> "TunnelRegistry":
        # Local and remote ids share the LiteLLM model_name namespace, so both
        # must be unique together — a client picks a model by this single id.
        ids = [inst.id for inst in self.instances] + [
            rm.id for rm in self.remote_models
        ]
        dupes = _find_duplicates(ids)
        if dupes:
            raise ValueError(f"Duplicate model IDs found (instances + remote): {dupes}")
        return self

    @model_validator(mode="after")
    def validate_no_litellm_port_collision(self) -> "TunnelRegistry":
        for inst in self.instances:
            if inst.port == self.litellm.port:
                raise ValueError(
                    f"Instance '{inst.id}' port {inst.port} collides with "
                    f"LiteLLM proxy port {self.litellm.port}."
                )
        return self

    @model_validator(mode="after")
    def validate_fallback_ids(self) -> "TunnelRegistry":
        """Ensure every fallback ID references a real registered model, not itself.

        Fallback targets may be local instances or remote models, so a local
        model can escalate to a hosted API (e.g. overflow -> DeepSeek).
        Internal instances are not valid targets: they are absent from the
        LiteLLM model_list, so the router could never reach them.
        """
        valid_ids = self.routable_ids
        for inst in self.instances:
            unknown = [fb for fb in inst.fallbacks if fb not in valid_ids]
            if unknown:
                raise ValueError(
                    f"Instance '{inst.id}' fallbacks reference unknown IDs: {unknown}"
                )
            if inst.id in inst.fallbacks:
                raise ValueError(
                    f"Instance '{inst.id}' cannot fall back to itself."
                )
        return self

    @model_validator(mode="after")
    def validate_gpu_budget(self) -> "TunnelRegistry":
        """Naive sum of gpu_memory_utilization across instances; assumes a single GPU, TP=1."""
        total = sum(inst.gpu_memory_utilization for inst in self.instances)
        if total > self.gpu.budget:
            terms = " + ".join(
                f"{inst.id}={inst.gpu_memory_utilization}" for inst in self.instances
            )
            raise ValueError(
                f"GPU memory over budget: {terms} = {total:.2f} > budget {self.gpu.budget:.2f}. "
                "Reduce gpu_memory_utilization or raise gpu.budget in models.yaml."
            )
        return self

    @model_validator(mode="after")
    def validate_services(self) -> "TunnelRegistry":
        """Every service references a defined tier and known model ids, once."""
        dupes = _find_duplicates([s.id for s in self.services])
        if dupes:
            raise ValueError(f"Duplicate service ids found: {dupes}")
        # Ids like "a-b" and "a_b" collide once normalized to the keys.env
        # variable name (SVC_A_B) and would silently share one virtual key.
        env_dupes = _find_duplicates(
            [s.id.upper().replace("-", "_") for s in self.services]
        )
        if env_dupes:
            raise ValueError(
                f"Service ids collide after '-'/'_' normalization: {env_dupes}"
            )
        valid_ids = self.routable_ids
        for svc in self.services:
            if svc.tier not in self.tiers:
                raise ValueError(
                    f"Service '{svc.id}' references unknown tier '{svc.tier}'. "
                    f"Defined tiers: {sorted(self.tiers)}"
                )
            unknown = [m for m in svc.models if m not in valid_ids]
            if unknown:
                raise ValueError(
                    f"Service '{svc.id}' models reference unknown IDs: {unknown}"
                )
        return self

    @model_validator(mode="after")
    def validate_guardrails(self) -> "TunnelRegistry":
        """The guard model must be a registered instance marked internal.

        Internal is required so the classifier is never client-routable: the
        guard hook calls its vLLM port directly, outside LiteLLM routing,
        keys, and spend tracking.
        """
        if self.guardrails is None:
            return self
        inst = self.get_instance(self.guardrails.model)
        if inst is None:
            raise ValueError(
                f"guardrails.model '{self.guardrails.model}' is not a "
                f"registered instance. Instances: {[i.id for i in self.instances]}"
            )
        if not inst.internal:
            raise ValueError(
                f"guardrails.model '{inst.id}' must set internal: true so it "
                "is excluded from client routing and key allowlists."
            )
        return self

    def get_instance(self, instance_id: str) -> Optional[InstanceConfig]:
        """Return the instance with the given ID, or None."""
        return next((i for i in self.instances if i.id == instance_id), None)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Override wins on leaf conflicts."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_registry(path: Path | str | None = None) -> TunnelRegistry:
    """Load, merge defaults, validate, and return the TunnelRegistry.

    Args:
        path: Path to the YAML registry file. If omitted, resolves from the
            TUNNEL_REGISTRY env var (falling back to "configs/models.yaml")
            — the same lookup tunnel.cli.registry_path() performs, so a bare
            load_registry() call is still env-aware. CLI call sites pass
            registry_path() explicitly rather than relying on this fallback.

    Returns:
        Validated TunnelRegistry instance.

    Raises:
        FileNotFoundError: if the YAML file doesn't exist.
        pydantic.ValidationError: if the registry is invalid.
    """
    if path is None:
        path = os.environ.get("TUNNEL_REGISTRY", "configs/models.yaml")
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    defaults: dict[str, Any] = raw.pop("defaults", {})
    raw["instances"] = [
        _deep_merge(defaults, inst_raw)
        for inst_raw in raw.get("instances", [])
    ]
    return TunnelRegistry.model_validate(raw)
