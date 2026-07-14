"""LiteLLM pre-call hook: inject vLLM scheduling priority from key metadata.

Registered via litellm_settings.callbacks when the registry defines tiers.
`tunnel keys sync` writes each key's tier priority into its metadata; this
hook copies it onto the request as vLLM's `priority` param (lower = served
earlier, preempts by (priority, arrival_time)). Only local instances running
with scheduling_policy=priority receive the param; remote upstreams and fcfs
instances are left untouched.

The registry is snapshotted once at proxy import: after editing tiers or
scheduling_policy and running `make generate`, restart the proxy.
"""
from __future__ import annotations

from litellm.integrations.custom_logger import CustomLogger

from tunnel.registry import TunnelRegistry, load_registry


class TierPriorityHook(CustomLogger):
    """Maps a virtual key's tier priority onto vLLM's per-request priority."""

    def __init__(self, registry: TunnelRegistry | None = None):
        """Snapshot the registry's priority-enabled instances and tier bounds.

        Args:
            registry: Injected for tests; defaults to load_registry(), which
                honors TUNNEL_REGISTRY the same way the CLI does.
        """
        super().__init__()
        registry = registry or load_registry()
        self._priority_model_ids = {
            inst.id for inst in registry.instances
            if inst.scheduling_policy == "priority"
        }
        # Keyless / unmanaged callers get the worst defined priority.
        self._default_priority = max(
            (t.priority for t in registry.tiers.values()), default=0
        )

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Set data["priority"] for priority-scheduled local models.

        Args:
            user_api_key_dict: LiteLLM's resolved key auth (metadata carries
                the tier priority written by `tunnel keys sync`).
            cache: LiteLLM DualCache (unused).
            data: The outgoing request payload (mutated).
            call_type: LiteLLM call type, e.g. "completion".

        Returns:
            The (possibly mutated) request payload.
        """
        if data.get("model") in self._priority_model_ids:
            metadata = getattr(user_api_key_dict, "metadata", None) or {}
            priority = metadata.get("priority", self._default_priority)
            try:
                data["priority"] = int(priority)
            except (TypeError, ValueError):
                # Metadata is editable outside `tunnel keys sync`; a garbage
                # value must degrade to the worst tier, not fail the request.
                data["priority"] = self._default_priority
        return data


tier_priority_handler = TierPriorityHook()
