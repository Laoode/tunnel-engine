# DeepSeek API Integration

Maintenance reference for the DeepSeek provider wired into the tunnel-engine gateway.
DeepSeek is an OpenAI-compatible hosted API, so it rides the same LiteLLM proxy on `:4000`
as the local vLLM models — no local process, no GPU, never health-gated.

## Models

| Gateway id (`model_name`) | Upstream model | Notes |
|---|---|---|
| `deepseek-v4-pro` | `deepseek-v4-pro` | Thinking-capable (reasoning). Tool calls + JSON mode. |
| `deepseek-v4-flash` | `deepseek-v4-flash` | Fast / cheap. |

Deprecated (removed 2026-07-24): `deepseek-chat`, `deepseek-reasoner`. Do not add these.

- Base URL: `https://api.deepseek.com` (OpenAI format). Anthropic format is available at
  `https://api.deepseek.com/anthropic`; beta strict-mode tool calls at `https://api.deepseek.com/beta`.
- Auth: `Authorization: Bearer $DEEPSEEK_API_KEY`.

## How it is registered

Remote upstreams live in a top-level `remote_models:` block in `configs/models.yaml` (and
`configs/models-prod.yaml`), separate from local GPU `instances:`:

```yaml
remote_models:
  - id: deepseek-v4-pro
    upstream_model: deepseek-v4-pro
    api_base: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY   # env var NAME, not the secret
    thinking: true
    description: "DeepSeek V4 Pro – hosted, thinking-capable"
```

`make generate` compiles each entry into a LiteLLM `model_list` item. The key is emitted as an
`os.environ/` reference, so no secret is written into the generated config:

```yaml
- model_name: deepseek-v4-pro
  litellm_params:
    model: openai/deepseek-v4-pro         # openai/ prefix = OpenAI-compatible upstream
    api_base: https://api.deepseek.com
    api_key: os.environ/DEEPSEEK_API_KEY
```

Set the real key in `.env` (dev) or the container env / secret manager (prod):

```
DEEPSEEK_API_KEY=sk-...
```

### Schema fields (`RemoteModelConfig`, tunnel/registry.py)

| Field | Required | Default | Meaning |
|---|---|---|---|
| `id` | yes | – | Gateway `model_name` clients call. Unique across instances + remote_models. |
| `upstream_model` | yes | – | Provider-side model id. |
| `api_base` | yes | – | Upstream base URL. |
| `api_key_env` | yes | – | Name of the env var holding the key. |
| `provider` | no | `openai` | LiteLLM prefix. DeepSeek is OpenAI-compatible. |
| `thinking` | no | `false` | Documents intent; passthrough is per-request (below). |
| `description` | no | `""` | Free text. |

Remote models are exempt from port-collision, GPU-budget, and health-gating logic (they are not
in `registry.instances`). A **local instance may fall back to a remote model** — e.g.
`fallbacks: [deepseek-v4-flash]` on a local instance escalates overflow / hard queries to DeepSeek.

## Calling it

Through the gateway (recommended), authenticating with the LiteLLM master key:

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $LITELLM_MASTER_KEY" -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Capital of France?"}],"max_tokens":64}'
```

### Thinking mode (`deepseek-v4-pro`)

Enabled per request via `extra_body`, defaults to enabled on `pro`. Effort is `high` (default)
or `max`:

```python
resp = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[...],
    max_tokens=800,                       # leave room: reasoning consumes completion tokens
    extra_body={"thinking": {"type": "enabled"}, "reasoning_effort": "high"},
)
reasoning = resp.choices[0].message.reasoning_content   # sibling of .content
```

- The response carries `reasoning_content` at the same level as `content`.
- Thinking mode ignores `temperature`, `top_p`, `presence_penalty`, `frequency_penalty`. The
  gateway sets `litellm_settings.drop_params: true`, so these are dropped rather than erroring.
- In multi-turn chat WITHOUT tool calls, `reasoning_content` is not concatenated into the next
  turn's context. WITH tool calls, `reasoning_content` **must** be passed back into context in
  every subsequent turn.
- Keep `max_tokens` generous: with a tiny budget the reasoning tokens are consumed first and
  `content` can come back empty (verified behavior).

### JSON mode

`response_format={"type":"json_object"}`. The prompt (system or user) must contain the word
"json" and ideally an example shape. The API may occasionally return empty content — retry.

### Tool calls

OpenAI-compatible: send `tools` / `tool_choice`, read `message.tool_calls`. Supported on
`deepseek-v4-pro` in both thinking and non-thinking modes. Optional strict-schema mode: set
`strict: true` on a function and point `api_base` at `https://api.deepseek.com/beta` (requires
`additionalProperties: false` and all properties marked required).

## Verified (2026-07-11)

- `deepseek-v4-flash` and `deepseek-v4-pro` reachable with the configured key.
- `pro` thinking mode returns a populated `reasoning_content`.
- End-to-end through the `:4000` gateway: `/v1/models` lists both DeepSeek ids; a
  `deepseek-v4-flash` completion returns real content.
