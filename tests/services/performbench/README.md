# performbench — unified performance testing

One scheme for all performance work on the Tunnel Engine. The rule:
**every feature or research branch ends with a performbench run** before it
ships. Gated scenarios are the ship/no-ship signal; the rest are diagnostics.

## Usage

```bash
make perf                       # default suite (smoke + kv-longdoc), gated
make perf SCENARIOS="goodput"   # explicit scenario set
python tests/services/performbench/main.py --list
```

Requires a running engine (`make up`) for gateway scenarios; instance
scenarios only need that instance (`make serve ID=...`).

## Layout

| Path | Role |
|---|---|
| `main.py` | Runner: resolves targets from the registry, runs scenarios, writes `RESULTS.md` + `results/<ts>/` |
| `scenarios.yaml` | Scenario catalog (the only file you edit to add one) |
| `long_doc_qa/` | Long-document repeat QA workload (KV-cache verification, from vLLM benchmarks) |
| `ttft-estimator/` | TTFT vs context length sweep (prefill scaling curve) |
| `aiperf/` | Reference docs for the AIPerf profiles used in scenarios |

## Scenario kinds

- **aiperf** — NVIDIA AIPerf profiles: static ISL/OSL, mixed
  sequence-length distributions, multi-turn conversations, goodput vs SLOs.
  Full metrics (TTFT/ITL/throughput percentiles) land as csv/json artifacts.
- **long_doc_qa** — repeated long-prefix prompts; `--expected-ttft-gain`
  makes it a hard KV-cache gate (a broken cache yields ~1.0x and fails).
- **ttft_estimator** — single-request TTFT sweep over context lengths.

## Adding a scenario

Add a block to `scenarios.yaml`:

```yaml
  my-scenario:
    kind: aiperf                # aiperf | long_doc_qa | ttft_estimator
    target: gateway             # or instance:<registry-id>
    gate: true                  # nonzero exit fails `make perf`
    description: "What this measures and why"
    args: [--endpoint-type, chat, --streaming, ...]
```

Targets resolve from `configs/models.yaml`: `gateway` goes through LiteLLM
with the master key (what services experience), `instance:<id>` hits the
vLLM port directly (engine-only, no proxy overhead).

## Interpreting results

- `RESULTS.md` — latest run summary (committed, like the other services).
- `results/<timestamp>/<scenario>/` — full artifacts per run (gitignored):
  AIPerf exports, response logs, plots. Use `aiperf plot` on an artifact
  dir for visualizations.
- Compare runs by keeping the same scenario args; change one variable at a
  time (model, cache backend, concurrency).
