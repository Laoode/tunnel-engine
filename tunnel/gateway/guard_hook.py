"""LiteLLM hooks: gateway content safety via a local XGuard classifier.

Registered via litellm_settings.callbacks when the registry defines an
enabled `guardrails:` block. The hook calls the internal guard instance's
vLLM port directly (no LiteLLM routing, keys, or spend) with max_tokens=1
and reads the first-token top logprobs: XGuard's two-stage output paradigm
puts the risk label in the first decoded token, so one prefill pass yields
per-category risk scores.

Input (pre-call) checks classify the last user message and are always on.
Response (post-call) checks classify the (user, assistant) pair and run only
when guardrails.check_output is true; streaming responses are not checked.
Services may opt out per-service (`guardrails: false` in the registry);
callers without a service key are always guarded.

Like tier_hook, the registry is snapshotted at proxy import: after editing
the guardrails block, run `make generate` and restart the proxy.
"""
from __future__ import annotations

import math

import httpx
import structlog
from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger

from tunnel.registry import TunnelRegistry, load_registry

log = structlog.get_logger(__name__)

# First-token label -> risk category, vendored from the XGuard tokenizer's
# id2risk (Alibaba-AAIG/YuFeng-XGuard-Reason-0.6B tokenizer_config.json).
SAFE_LABEL = "Safe-Safe"
ID2RISK = {
    "sec": SAFE_LABEL,
    "pc": "Crimes and Illegal Activities-Pornographic Contraband",
    "dc": "Crimes and Illegal Activities-Drug Crimes",
    "dw": "Crimes and Illegal Activities-Dangerous Weapons",
    "pi": "Crimes and Illegal Activities-Property Infringement",
    "ec": "Crimes and Illegal Activities-Economic Crimes",
    "ac": "Hate Speech-Abusive Curses",
    "def": "Hate Speech-Defamation",
    "ti": "Hate Speech-Threats and Intimidation",
    "cy": "Hate Speech-Cyberbullying",
    "ph": "Physical and Mental Health-Physical Health",
    "mh": "Physical and Mental Health-Mental Health",
    "se": "Ethics and Morality-Social Ethics",
    "sci": "Ethics and Morality-Science Ethics",
    "pp": "Data Privacy-Personal Privacy",
    "cs": "Data Privacy-Commercial Secret",
    "acc": "Cybersecurity-Access Control",
    "mc": "Cybersecurity-Malicious Code",
    "ha": "Cybersecurity-Hacker Attack",
    "ps": "Cybersecurity-Physical Security",
    "ter": "Extremism-Violent Terrorist Activities",
    "sd": "Extremism-Social Disruption",
    "ext": "Extremism-Extremist Ideological Trends",
    "fin": "Inappropriate Suggestions-Finance",
    "med": "Inappropriate Suggestions-Medicine",
    "law": "Inappropriate Suggestions-Law",
    "cm": "Risks Involving Minors-Corruption of Minors",
    "ma": "Risks Involving Minors-Minor Abuse and Exploitation",
    "md": "Risks Involving Minors-Minor Delinquency",
}


def extract_text(content) -> str:
    """Flatten an OpenAI message content field to plain text.

    Args:
        content: A string, or a list of content parts (multimodal format).

    Returns:
        The concatenated text; non-text parts are ignored.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def risk_scores_from_response(payload: dict) -> dict[str, float]:
    """Map a vLLM chat completion's first-token logprobs to risk scores.

    Args:
        payload: Parsed /v1/chat/completions response with logprobs requested.

    Returns:
        {risk_category: probability} for every top-logprob token that is a
        known XGuard label. Empty when logprobs are absent.
    """
    try:
        top = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {
        ID2RISK[entry["token"]]: math.exp(entry["logprob"])
        for entry in top
        if entry.get("token") in ID2RISK
    }


class GuardHook(CustomLogger):
    """Blocks unsafe requests (and optionally responses) at the gateway."""

    def __init__(self, registry: TunnelRegistry | None = None):
        """Snapshot the guardrails config and per-service opt-outs.

        Args:
            registry: Injected for tests; defaults to load_registry().
        """
        super().__init__()
        registry = registry or load_registry()
        cfg = registry.guardrails
        self.enabled = bool(cfg and cfg.enabled)
        if not self.enabled:
            return
        guard_inst = registry.get_instance(cfg.model)
        self._url = f"{guard_inst.api_base}/chat/completions"
        self._guard_model = guard_inst.served_model_name or guard_inst.model
        self._threshold = cfg.threshold
        self.check_output = cfg.check_output
        self._fail_open = cfg.on_error == "allow"
        self._service_guarded = {
            s.id: s.guardrails if s.guardrails is not None else True
            for s in registry.services
        }
        self._client = httpx.AsyncClient(timeout=cfg.timeout_s)

    def _should_guard(self, user_api_key_dict) -> bool:
        """True unless the calling service opted out; keyless callers are guarded."""
        metadata = getattr(user_api_key_dict, "metadata", None) or {}
        return self._service_guarded.get(metadata.get("service"), True)

    async def _classify(self, messages: list[dict]) -> dict[str, float]:
        """Run one guard inference and return per-category risk scores.

        Args:
            messages: XGuard-format messages: [user] for an input check,
                [user, assistant] for a response check.

        Returns:
            {risk_category: probability}.

        Raises:
            httpx.HTTPError: On connection errors, timeouts, or HTTP failures.
        """
        resp = await self._client.post(self._url, json={
            "model": self._guard_model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0,
            "logprobs": True,
            "top_logprobs": 10,
        })
        resp.raise_for_status()
        return risk_scores_from_response(resp.json())

    async def _verdict(self, messages: list[dict], stage: str) -> None:
        """Classify and raise HTTPException when the risk threshold is crossed.

        Args:
            messages: Guard-format messages to classify.
            stage: "input" or "output", for the block payload and logs.

        Raises:
            HTTPException: 400 on a block; 503 when the guard is unreachable
                and on_error is "block".
        """
        try:
            scores = await self._classify(messages)
        except httpx.HTTPError as exc:
            log.warning("guard_unavailable", stage=stage, error=str(exc),
                        fail_open=self._fail_open)
            if self._fail_open:
                return
            raise HTTPException(
                status_code=503,
                detail={"error": "Guardrail unavailable and on_error=block."},
            ) from exc
        risky = {cat: p for cat, p in scores.items() if cat != SAFE_LABEL}
        if not risky:
            return
        top_category, top_score = max(risky.items(), key=lambda kv: kv[1])
        if top_score >= self._threshold:
            log.info("guard_blocked", stage=stage, category=top_category,
                     score=round(top_score, 4))
            raise HTTPException(status_code=400, detail={
                "error": f"Blocked by content guardrail ({stage} check).",
                "guardrail": "xguard",
                "category": top_category,
                "score": round(top_score, 4),
            })

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        """Classify the last user message before the request reaches a model."""
        if not self.enabled or call_type not in ("completion", "acompletion"):
            return data
        if not self._should_guard(user_api_key_dict):
            return data
        user_text = next(
            (extract_text(m.get("content")) for m in reversed(data.get("messages", []))
             if m.get("role") == "user"), "",
        )
        if user_text:
            await self._verdict([{"role": "user", "content": user_text}], "input")
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        """Classify the model response when check_output is enabled (non-streaming)."""
        if not (self.enabled and self.check_output):
            return
        if not self._should_guard(user_api_key_dict):
            return
        try:
            answer = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            return
        user_text = next(
            (extract_text(m.get("content")) for m in reversed(data.get("messages", []))
             if m.get("role") == "user"), "",
        )
        if answer:
            await self._verdict(
                [{"role": "user", "content": user_text},
                 {"role": "assistant", "content": answer}],
                "output",
            )


guard_handler = GuardHook()
