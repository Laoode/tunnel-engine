# Tunnel Engine Playbook

> **This is a living document.** The engine is under active development, so parts of
> this playbook WILL go stale. Any developer or AI session (Claude Code) that changes
> behavior described here must update, add, or delete the matching section in the same
> change. When this file and the code disagree, the code wins; fix the playbook.
>
> Last verified against the code: 2026-07-17 (LMCache MP + performbench, uncommitted).

## What this engine is, in one paragraph

Tunnel Engine turns one GPU box into a multi-tenant LLM gateway. You declare models
and consumers in a single YAML file, and the engine generates every downstream config,
launches vLLM instances, and fronts them with a LiteLLM proxy on `:4000` that speaks
the OpenAI API. Each consuming service gets its own virtual key with a model
allowlist, tier-based rate limits, a budget, and a scheduling priority, so multiple
microservices can safely share the same GPUs while their costs stay separately
attributable.

## The one mental model to internalize

```
configs/models.yaml  ──make generate──▶  derived configs  ──make up──▶  running stack
(single source of truth)                 (never hand-edit)
```

- `configs/models.yaml` is the **only** file you edit to change models, tiers,
  services, costs, or scheduling. (`configs/models-prod.yaml` is the prod variant,
  selected with `REGISTRY=configs/models-prod.yaml make <target>`.)
- `make generate` rebuilds `configs/litellm/` and
  `configs/prometheus/`. These are build artifacts. Editing them by hand is the #1
  way to lose work, because the next `generate` overwrites them.
- `make check` validates the registry without writing anything. Run it after every
  YAML edit; it catches duplicate ids/ports, GPU over-budget, unknown tiers, and
  services referencing models that do not exist.

## Runtime map (what runs where)

| Thing | Port | Started by | Notes |
|---|---|---|---|
| LiteLLM proxy (the gateway) | 4000 | `make up` or `make start` | OpenAI-compatible; all clients call this |
| vLLM instances | 8000, 8001, ... | `make up` or `make serve ID=` | one process per registry instance |
| LMCache servers | 9000, 9001, ... (= inst. port + 1000) | auto by `serve`/`up` per `lmcache.enabled` instance | KV cache tiers (MP mode); survives `make stop ID=`, cleaned by `make down` |
| XGuard classifier | 8002 | `make up` or `make serve ID=xguard-0.6b` | `internal:` instance; content safety, not client-routable |
| Postgres (keys, spend logs) | 5433 | `make db-up` (auto via `up`/`start` when the container is down) | Docker, data in `/teamspace/.../.tunnel-pg` |
| Prometheus | 9092 | `make obs-up` | 9090/9091 are taken by Lightning infra |
| Grafana | 3000 | `make obs-up` | admin/admin, dashboard auto-provisioned |

## Repo map

| Path | What lives there |
|---|---|
| `configs/models.yaml` | THE source of truth: instances, remote models, tiers, services, costs |
| `tunnel/registry.py` | Pydantic schema + validation for the registry |
| `tunnel/cli.py` | `python -m tunnel.cli` entry point; every make target wraps this |
| `tunnel/orchestrator.py` | process launch/stop/adopt, pidfiles, port scanning |
| `tunnel/gateway/config_builder.py` | registry -> LiteLLM config generation |
| `tunnel/gateway/keys.py` | declarative virtual-key sync against the LiteLLM DB |
| `tunnel/gateway/tier_hook.py` | LiteLLM pre-call hook: key tier -> vLLM request priority |
| `tunnel/gateway/guard_hook.py` | LiteLLM pre/post-call hook: XGuard content safety (calls :8002 directly) |
| `tunnel/cache/lmcache_config.py` | registry -> `lmcache server` argv (MP mode) |
| `tunnel/observability/prometheus_config.py` | registry -> Prometheus scrape config |
| `tunnel/health/checker.py` | instance health polling used by the `up`/`start` gate |
| `tests/unit/` | 193 fast tests, no GPU (`make test-unit`) |
| `tests/services/loadgen/` | load generator + analysis plots (`make loadtest`) |
| `tests/services/guardbench/` | guardrail eval: DeepSeek gen -> Sonnet judge -> latency/accuracy bench |
| `tests/services/performbench/` | unified perf bench: `make perf` gated suite + scenario catalog |
| `.tunnel/keys.env` | the ONLY copy of virtual-key secrets (0600, gitignored) |
| `MEMORY.md` | AI session brain: current state, decisions, lessons |
| `docs/PLAN.md` | roadmap and phase history (gitignored working doc) |

## Day 1: from clone to first response

```bash
make install                  # uv pip install requirements/dev.txt
cp .env.example .env          # then set HF_TOKEN, LITELLM_MASTER_KEY,
                              # PG_PASSWORD, DATABASE_URL (see .env.example comments)
make check                    # registry parses, GPU budget ok
make generate                 # rebuild derived configs
make up                       # starts Postgres if needed, then all vLLM instances
                              # (+ their lmcache servers) + proxy; first cold model
                              # load can take 10+ min on slow filesystems
make keys-sync                # create/reconcile one virtual key per service
make keys-list                # verify: alias, tier, models, spend per service
```

Smoke test with a service key (never ship the master key to a service):

```bash
KEY=$(grep SVC_DEVELOPMENT .tunnel/keys.env | cut -d= -f2)
curl http://localhost:4000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "qwen-0.8b", "messages": [{"role": "user", "content": "hi"}]}'
```

Tear down with `make down` (stops all instances + proxy, frees the GPU) and
`make db-down` / `make obs-down` for the containers.

## How multi-tenancy works (the Phase 1-2 features)

**Virtual keys.** Each `services:` entry in the registry gets one LiteLLM virtual key.
`make keys-sync` is declarative: registry = desired state, LiteLLM's Postgres = actual
state, and it creates, updates (on drift), or regenerates (if the DB was wiped) keys
to match. Secrets land in `.tunnel/keys.env` as `SVC_<ID>=sk-...`; that file is the
only copy, so guard it. `tunnel keys sync --prune` deletes keys for services removed
from the registry.

**Tiers = rate limits + budget + priority.** A tier (`free`/`pro`/`enterprise` today,
but any names work) sets `rpm_limit`, `tpm_limit`, optional `max_budget` +
`budget_duration`, and a scheduling `priority` (lower number = served earlier).
Limits are enforced by LiteLLM per key: exceeding rpm/tpm returns HTTP 429, calling a
model outside the key's allowlist returns 403, blowing the budget blocks the key.

**Priority scheduling.** vLLM instances run with `--scheduling-policy priority`
(registry `defaults.scheduling_policy`). The `tier_hook` reads the calling key's tier
priority from its metadata and stamps it onto the request, so under saturation an
enterprise request preempts free-tier decodes (verified live: TTFT 5.5s vs 83.6s
under a full queue). Two caveats: the hook snapshots the registry at proxy start, so
tier/service changes need a proxy restart; and vLLM crashes on `priority` + `n>1`
(vllm#29747), so no service may send `n > 1`.

**Cost attribution.** Each model's `cost:` block (USD per 1M tokens; synthetic
amortized rate for local models, list price for remote) feeds LiteLLM spend logs.
`make keys-list` shows cumulative spend per service. DeepSeek prices in the registry
are PLACEHOLDERS until real list prices are entered.

**Guardrails (content safety).** The `guardrails:` block enables an in-proxy XGuard
classifier: `guard_hook` calls the internal `xguard-0.6b` instance (:8002) with
`max_tokens=1` and reads the first-token logprobs as per-category risk scores, blocking
when any non-safe score crosses `threshold`. **Input (pre-call) checking is always on
when enabled; output (post-call, non-streaming) checking is opt-in via `check_output`.**
The guard model must be an instance marked `internal: true`, which keeps it out of the
LiteLLM model_list, key allowlists, and fallback targets (clients cannot route to it).
Services opt out with `guardrails: false` (e.g. the `tools-bench` harness); keyless
callers are always guarded. `on_error: allow` fails open (a guard outage must not take
the gateway down); `block` fails closed with a 503. Like `tier_hook`, the hook snapshots
the registry at proxy start, so guardrails edits need `make generate` + a proxy restart.

## Common workflows

**Add a local model:** copy an instance block in `models.yaml` (unique `id` + `port`,
GPU utilizations must sum under `gpu.budget`), then `make check && make generate`,
then `make down && make up` (or `make serve ID=<new-id>` while others keep running,
followed by a proxy restart to advertise it).

**Serve a model from S3 (no local weights):** instance block with `model: s3://bucket/path`,
`load_format: runai_streamer`, and a `served_model_name`; AWS_* / RUNAI_STREAMER_* vars in
`.env` pick the provider (Wasabi today — see `docs/s3-llm.md`). Weights stream to VRAM at
startup (9.3GB ≈ 15s) and re-stream on every cold start. The `qwen3.5-4b-s3` block in
`models.yaml` is the commented reference; at 0.90 gpu it only fits the L4 solo.

**Add a consuming service:** add a `services:` entry (pick a tier, optionally restrict
`models:`), `make generate`, restart the proxy (the tier hook snapshot), then
`make keys-sync` and hand the new `SVC_*` secret to that service.

**Change tier limits:** edit the `tiers:` block, `make generate`, restart the proxy,
`make keys-sync` (sync pushes the new limits onto existing keys, including resetting
previously relaxed fields).

**Load test + analyze:** `make loadtest` (env vars `RATE`, `DURATION`, `MIX`
one-of/all of LISO/LILO/SILO/SISO, `TIER_MIX="tools-bench:4,development:2"` to drive
real service keys), then `make loadtest-plots` renders five PNGs into
`tests/services/loadgen/results/`. Watch live behavior in Grafana (`make obs-up`).

**Tune / benchmark guardrails:** toggle globally (`guardrails.enabled`), per-service
(`guardrails: false`), or turn on response checks (`check_output: true`) in `models.yaml`,
then `make generate` + proxy restart. Benchmark the classifier end-to-end with
`make guard-dataset` (DeepSeek v4 Pro generates an Indonesian eval set; needs
`DEEPSEEK_API_KEY`), `make guard-judge` (Sonnet 5 audits the labels; needs the `claude`
CLI), then `make guard-bench` (live latency + accuracy vs the judged set; needs the fleet
up). Sweep the block cutoff with `THRESHOLD=0.3 make guard-bench`; results land in
`tests/services/guardbench/RESULTS.md`.

**Retire one model with zero downtime for the rest:** `make stop ID=<id>`. The proxy
still advertises it until you remove its registry block, `make generate`, and restart
the proxy in a maintenance window.

**Run the performance gate (required for serving-path changes):** `make perf` with
the fleet up runs the gated suite (smoke aiperf profile + kv-longdoc TTFT-gain
gate); nonzero exit = do not ship. `make perf-list` shows the catalog;
`make perf SCENARIOS="goodput mixed"` picks scenarios. Artifacts under
`tests/services/performbench/results/<ts>/`, summary in its `RESULTS.md`.
Results recap + method: `docs/performance-kv-cache.md`.

**Prod rollout:** same commands with `REGISTRY=configs/models-prod.yaml`. Fix the
placeholder DeepSeek prices first.

## Gotchas that will actually bite you

These are environment facts, not opinions. More detail in `MEMORY.md` Lessons.

1. **Lightning `/teamspace` drops empty directories on studio restart.** Postgres
   then refuses to boot (missing `pg_notify` etc.). `make db-up` self-heals this,
   and `make up`/`start` now auto-run it when the tunnel-pg container is down.
   Docker images/containers are wiped on restart; bind-mounted data survives.
2. **Cold model loads are slow** (5-15 min on the Lightning filesystem). If `make up`
   times out at the health gate, the instances usually keep loading; run `make start`
   once `make health` goes green instead of relaunching.
3. **Never hand-edit `configs/litellm|prometheus`** - regenerated on every
   `make generate`. (LMCache no longer uses config files: each instance's
   `lmcache server` is launched with argv built from the registry.)
4. **A leftover vLLM process holds GPU memory and its port.** `make down` before
   swapping models; verify with `nvidia-smi` (0 MiB used). `make kill` is the nuclear
   option (kills every GPU compute process on the box). Related: after
   `make stop ID=`, wait for `nvidia-smi` to show the memory freed before
   relaunching - a killed EngineCore takes seconds to release VRAM and an
   instant relaunch fails at GPU profiling.
5. **Secrets:** everything lives in `.env` (gitignored) and `.tunnel/keys.env`
   (gitignored, 0600). Configs reference secrets as `os.environ/NAME` strings. Never
   paste a real key into YAML, code, logs, or this doc.
6. **First proxy boot with a fresh DB** needs `prisma generate` + network access for
   LiteLLM's migrations. If the proxy dies complaining about Prisma binaries, run
   `prisma generate` inside the litellm/proxy site-packages directory once.
7. **LiteLLM loads custom callbacks relative to the config file's directory**, not
   `sys.path`. That is why `configs/litellm/tier_hook.py` and `guard_hook.py` (generated
   one-line shims) exist; do not delete them or "simplify" the callback path.
8. **Hybrid-attention models (qwen-0.8b) use LMCache via the MP connector.**
   Each enabled instance runs a companion `lmcache server` on instance port
   + 1000 (`logs/lmcache-<id>.log`); Mamba hybrids need `mamba_align: true`
   and `chunk_size` = the unified attention block size N from the startup
   log. See docs/lmcache.md.
9. **A vLLM instance can hang during `make up`'s sequential handoff** - the API server
   comes up but its EngineCore never spawns (stuck in `futex_wait`, no OOM, no traceback),
   seen adding the 3rd instance. The vLLM child stdout is block-buffered to `logs/<id>.log`,
   so the log looks frozen even when it is loading; force live output with
   `PYTHONUNBUFFERED=1 stdbuf -oL -eL make serve ID=<id>`. If it is truly hung (GPU util 0%,
   no EngineCore in `ps aux | grep EngineCore`), `make stop ID=<id>` and relaunch it
   standalone - it comes up in ~90s on its own.

## Where to look when something is wrong

| Symptom | First place to look |
|---|---|
| 401/403 from the gateway | wrong key, or model not in the service's allowlist (`make keys-list`) |
| 429 from the gateway | tier rpm/tpm limit hit; by design, not a bug |
| 400 "Blocked by content guardrail" | XGuard flagged the prompt/response; category+score in the error body. Tune `threshold` or opt the service out |
| Instance never healthy | its log at `logs/<id>.log`, then `nvidia-smi` for OOM/orphans (see gotcha 9 for the EngineCore-never-spawned hang) |
| Instance with lmcache stuck at launch | `logs/lmcache-<id>.log`; `serve` fail-fasts if the lmcache server never opens its port |
| Warm TTFT not improving (cache "dead") | `grep -E "Stored\|Prefetch" logs/lmcache-<id>.log`; for Mamba hybrids check `chunk_size` matches the block size N in the vLLM startup log |
| S3 instance refuses to launch | AWS creds unset in `.env` (serve fails fast), or `runai-model-streamer` not installed; preflight bucket+creds with a boto3 list call (free) |
| Proxy up but model errors | stale derived config: `make generate` + proxy restart |
| Spend stuck at 0 | Postgres down (`docker ps`), or `DATABASE_URL` unset in `.env` |
| No metrics in Grafana | `make obs-up` running? Prometheus targets at `localhost:9092/targets` |

## Maintaining this playbook (for the next AI session)

- Update the "Last verified" date whenever you confirm or fix a section.
- When you add a feature, add its workflow here; when you remove one, delete its
  section - do not leave "deprecated" tombstones.
- Keep it under ~250 lines. This doc is for moving fast, not for completeness:
  deep rationale belongs in `MEMORY.md` (decisions/lessons) and `docs/PLAN.md`
  (roadmap), reference detail belongs in the code's docstrings.
- If you change `Makefile` targets, registry schema, ports, or the key-sync
  contract, grep this file for the old name and fix every mention.
