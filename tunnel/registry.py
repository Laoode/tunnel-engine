"""
tunnel/registry.py
==================
Loads and validates the model registry from configs/models.yaml.

This is the foundation layer. Every other module imports from here.
Nothing else should read raw YAML — go through load_registry().

Design principles:
  - Pydantic validates at load time: fail fast, fail loudly.
  - _deep_merge applies defaults so instance blocks stay concise.
  - Pure data classes only — no side effects, no I/O except load_registry().
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


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

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        allowed = {"cpu", "disk", "redis"}
        if v not in allowed:
            raise ValueError(f"lmcache.backend must be one of {allowed}, got '{v}'")
        return v


class InstanceConfig(BaseModel):
    id: str
    model: str
    port: int
    gpu_memory_utilization: float
    description: str = ""
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: int = 65536
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    lmcache: LMCacheInstanceConfig = Field(default_factory=LMCacheInstanceConfig)
    chat_template: Optional[str] = None
    extra_args: list[str] = Field(default_factory=list)
    fallbacks: list[str] = Field(default_factory=list)  # ordered fallback instance IDs

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


class LiteLLMGatewayConfig(BaseModel):
    port: int = 4000
    master_key: str = "sk-tunnel-dev-change-in-prod"
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


class GlobalLMCacheConfig(BaseModel):
    backend: str = "cpu"
    max_cache_size_gb: int = 20
    chunk_size: int = 256


class TunnelRegistry(BaseModel):
    instances: list[InstanceConfig]
    litellm: LiteLLMGatewayConfig = Field(default_factory=LiteLLMGatewayConfig)
    lmcache: GlobalLMCacheConfig = Field(default_factory=GlobalLMCacheConfig)

    @model_validator(mode="after")
    def validate_no_port_collisions(self) -> "TunnelRegistry":
        ports = [inst.port for inst in self.instances]
        dupes = sorted({p for p in ports if ports.count(p) > 1})
        if dupes:
            raise ValueError(f"Duplicate instance ports found: {dupes}")
        return self

    @model_validator(mode="after")
    def validate_no_id_collisions(self) -> "TunnelRegistry":
        ids = [inst.id for inst in self.instances]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"Duplicate instance IDs found: {dupes}")
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
        """Ensure every fallback ID references a real registered instance, not itself."""
        valid_ids = {inst.id for inst in self.instances}
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


def load_registry(path: Path | str = "configs/models.yaml") -> TunnelRegistry:
    """Load, merge defaults, validate, and return the TunnelRegistry.

    Args:
        path: Path to the YAML registry file.

    Returns:
        Validated TunnelRegistry instance.

    Raises:
        FileNotFoundError: if the YAML file doesn't exist.
        pydantic.ValidationError: if the registry is invalid.
    """
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    defaults: dict[str, Any] = raw.pop("defaults", {})
    raw["instances"] = [
        _deep_merge(defaults, inst_raw)
        for inst_raw in raw.get("instances", [])
    ]
    return TunnelRegistry.model_validate(raw)
