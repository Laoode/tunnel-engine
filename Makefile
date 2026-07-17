PYTHON      := python3
REGISTRY    ?= configs/models.yaml
LINT_PATHS  := tunnel/ tests/
HF_CACHE    := $(HOME)/.cache/huggingface/hub/
PG_DATA_DIR ?= /teamspace/studios/this_studio/.tunnel-pg

export TUNNEL_REGISTRY := $(REGISTRY)

.DEFAULT_GOAL := help

.PHONY: help generate list health proxy serve test lint fmt check up stop down \
        db-up db-down db-ensure keys-sync keys-list obs-up obs-down loadtest \
        loadtest-plots guard-dataset guard-judge guard-bench perf perf-list

help:
	@echo ""
	@echo "  Tunnel Engine"
	@echo "  ─────────────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

generate: ## Rebuild all derived configs from configs/models.yaml
	$(PYTHON) -m tunnel.cli generate

check: ## Validate the registry (configs/models.yaml, or REGISTRY=<path>) without writing anything
	@$(PYTHON) -c \
	  "from tunnel.cli import registry_path; \
	   from tunnel.registry import load_registry; \
	   r = load_registry(registry_path()); \
	   print(f'✓ Registry valid — {len(r.instances)} instance(s), proxy on :{r.litellm.port}')"

list: ## List all registered model instances
	$(PYTHON) -m tunnel.cli list

health: ## Poll health of all vLLM instances
	$(PYTHON) -m tunnel.cli health

# Lightning studio restarts wipe Docker containers; the proxy then dies with
# prisma "Not connected to the query engine". Auto-heal by starting tunnel-pg
# before anything that launches the proxy.
db-ensure: ## Start Postgres only if the tunnel-pg container is not running
	@if docker inspect -f '{{.State.Running}}' tunnel-pg 2>/dev/null | grep -q true; then \
		echo ".  tunnel-pg already running"; \
	else \
		echo ".  tunnel-pg not running -> make db-up"; \
		$(MAKE) db-up; \
	fi

start: db-ensure ## Health-gate + proxy: wait for all vLLM instances, then launch proxy
	$(PYTHON) -m tunnel.cli start

start-timeout: db-ensure ## Same as start with custom timeout. Usage: make start-timeout TIMEOUT=120
	@if [ -z "$(TIMEOUT)" ]; then echo "Usage: make start-timeout TIMEOUT=<seconds>"; exit 1; fi
	$(PYTHON) -m tunnel.cli start --timeout $(TIMEOUT)

up: db-ensure ## Launch ALL instances in background, health-gate, then start proxy
	$(PYTHON) -m tunnel.cli up

up-timeout: db-ensure ## Same as up with custom timeout. Usage: make up-timeout TIMEOUT=120
	@if [ -z "$(TIMEOUT)" ]; then echo "Usage: make up-timeout TIMEOUT=<seconds>"; exit 1; fi
	$(PYTHON) -m tunnel.cli up --timeout $(TIMEOUT)

stop: ## Stop ONE instance, leaving the others and the proxy running. Usage: make stop ID=<instance-id>
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make stop ID=<instance-id>"; \
		$(PYTHON) -m tunnel.cli list; \
		exit 1; \
	fi
	$(PYTHON) -m tunnel.cli stop $(ID)

down: ## Stop every instance (serve or up) and the proxy, then free the GPU
	$(PYTHON) -m tunnel.cli down

# Lightning's /teamspace drops EMPTY directories on studio restart, but Postgres
# refuses to boot without them (pg_notify etc.), so recreate them before starting.
PG_EMPTY_DIRS := pg_commit_ts pg_dynshmem pg_notify pg_replslot pg_serial \
	pg_snapshots pg_stat pg_stat_tmp pg_tblspc pg_twophase \
	pg_wal/archive_status pg_logical/mappings pg_logical/snapshots

db-up: ## Start Postgres (Docker) for LiteLLM virtual keys; data persists in $(PG_DATA_DIR)
	@PG_PASSWORD=$$(grep -E '^PG_PASSWORD=' .env 2>/dev/null | cut -d= -f2-); \
	if [ -z "$$PG_PASSWORD" ]; then echo "ERROR: set PG_PASSWORD in .env (see .env.example)"; exit 1; fi; \
	if [ -f $(PG_DATA_DIR)/PG_VERSION ]; then \
		docker run --rm -v $(PG_DATA_DIR):/var/lib/postgresql/data postgres:16-alpine \
			sh -c 'cd /var/lib/postgresql/data && for d in $(PG_EMPTY_DIRS); do \
				mkdir -p "$$d" && chown postgres:postgres "$$d" && chmod 700 "$$d"; done'; \
	fi; \
	docker rm -f tunnel-pg >/dev/null 2>&1 || true; \
	docker run -d --name tunnel-pg --restart unless-stopped -p 5433:5432 \
		-e POSTGRES_USER=litellm -e POSTGRES_PASSWORD=$$PG_PASSWORD -e POSTGRES_DB=litellm \
		-v $(PG_DATA_DIR):/var/lib/postgresql/data \
		postgres:16-alpine

db-down: ## Stop and remove the Postgres container (data persists in $(PG_DATA_DIR))
	docker stop tunnel-pg && docker rm tunnel-pg

keys-sync: ## Reconcile LiteLLM virtual keys with registry services (requires running proxy)
	$(PYTHON) -m tunnel.cli keys sync

keys-list: ## Per-service key status + spend (requires running proxy)
	$(PYTHON) -m tunnel.cli keys list

obs-up: ## Start Prometheus (:9092; 9090-9091 taken on Lightning) + Grafana (:3000, admin/admin)
	docker rm -f tunnel-prom tunnel-grafana >/dev/null 2>&1 || true
	docker run -d --name tunnel-prom --restart unless-stopped --network host \
		-v $(CURDIR)/configs/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro \
		prom/prometheus --config.file=/etc/prometheus/prometheus.yml \
		--web.listen-address=0.0.0.0:9092
	docker run -d --name tunnel-grafana --restart unless-stopped --network host \
		-e GF_SERVER_HTTP_PORT=3000 \
		-v $(CURDIR)/configs/grafana/provisioning:/etc/grafana/provisioning:ro \
		-v $(CURDIR)/configs/grafana/dashboards:/etc/grafana/dashboards:ro \
		grafana/grafana

obs-down: ## Stop Prometheus + Grafana containers
	docker rm -f tunnel-prom tunnel-grafana

serve: ## Launch one vLLM instance in the foreground. Usage: make serve ID=<instance-id>
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make serve ID=<instance-id>"; \
		$(PYTHON) -m tunnel.cli list; \
		exit 1; \
	fi
	$(PYTHON) -m tunnel.cli serve $(ID)

test: ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only (fast, no live services required)
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests (requires a running engine: make up, or serve + start)
	$(PYTHON) -m pytest tests/integration/ -v -m integration

test-tools: ## Tool-calling smoke test through the gateway (requires a running engine)
	$(PYTHON) tests/services/tools/main.py

bench-cache: ## KV-cache benchmark -> tests/services/kv_cache/RESULTS.md (requires a running engine)
	$(PYTHON) tests/services/kv_cache/main.py

loadtest: ## Open-loop load generator (RATE/DURATION/MIX/TIER_MIX env vars; running engine)
	$(PYTHON) tests/services/loadgen/main.py

loadtest-plots: ## Render analysis PNGs from loadgen results
	$(PYTHON) tests/services/loadgen/plots.py

guard-bench: ## Benchmark XGuard latency + accuracy on the judged dataset (needs a running fleet)
	$(PYTHON) tests/services/guardbench/main.py

perf: ## Unified performance bench (SCENARIOS="smoke goodput"; default: gated suite; running engine)
	$(PYTHON) tests/services/performbench/main.py $(SCENARIOS)

perf-list: ## List available performance scenarios
	$(PYTHON) tests/services/performbench/main.py --list

lint: ## Lint with ruff
	ruff check $(LINT_PATHS)

fmt: ## Format with ruff
	ruff format $(LINT_PATHS)

fmt-check: ## Check formatting without writing (CI-safe)
	ruff format --check $(LINT_PATHS)

tree: ## Show project tree, excluding common noise
	tree -I '__pycache__|*.pyc|.pytest_cache|.git|.venv|venv|dist|build'

install: ## Install dependency
	uv pip install -r requirements/dev.txt --torch-backend=auto

uninstall: ## Uninstall dependency
	uv pip uninstall -r tunnel-engine/requirements/dev.txt -y

kill:
	@nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
	@pkill -9 -f -i "[v]llm" 2>/dev/null || true

view-models: ## List all cached models
	@echo "Cached models:"
	@ls -d $(HF_CACHE)/models--* 2>/dev/null | xargs -n1 basename || echo "None found."

delete-model: ## Delete a specific model. Usage: make delete-model NAME=models--Qwen--Qwen3.5-0.8B
ifndef NAME
	$(error NAME is required. Usage: make delete-model NAME=models--...)
endif
	@if [ ! -d "$(HF_CACHE)/$(NAME)" ]; then \
		echo "Error: Model '$(NAME)' not found."; \
		exit 1; \
	fi
	@echo "Removing: $(HF_CACHE)/$(NAME)"
	@rm -rf "$(HF_CACHE)/$(NAME)"
	@echo "Done."

delete-all-models: ## Delete all cached models. Usage: make delete-all-models
	@echo "Warning: Deleting all models in $(HF_CACHE)..."
	@rm -rf $(HF_CACHE)/models--*
	@echo "Done."