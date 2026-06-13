PYTHON      := python3
REGISTRY    := configs/models.yaml
LINT_PATHS  := tunnel/ tests/

.DEFAULT_GOAL := help

.PHONY: help generate list health proxy serve test lint fmt check

help:
	@echo ""
	@echo "  Tunnel Engine"
	@echo "  ─────────────────────────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""

generate: ## Rebuild all derived configs from configs/models.yaml
	$(PYTHON) -m tunnel.cli generate

check: ## Validate configs/models.yaml without writing anything
	@$(PYTHON) -c \
	  "from tunnel.registry import load_registry; \
	   r = load_registry(); \
	   print(f'✓ Registry valid — {len(r.instances)} instance(s), proxy on :{r.litellm.port}')"

list: ## List all registered model instances
	$(PYTHON) -m tunnel.cli list

health: ## Poll health of all vLLM instances
	$(PYTHON) -m tunnel.cli health

proxy: ## Start the LiteLLM proxy (run `make generate` first)
	$(PYTHON) -m tunnel.cli proxy

serve: ## Launch a vLLM instance. Usage: make serve ID=qwen-0.8b
	@if [ -z "$(ID)" ]; then \
		echo "Usage: make serve ID=<instance-id>"; \
		$(PYTHON) -m tunnel.cli list; \
		exit 1; \
	fi
	$(PYTHON) -m tunnel.cli serve $(ID)

test: ## Run the full test suite
	python3 -m pytest tests/ -v

test-unit: ## Run unit tests only
	python3 -m pytest tests/unit/ -v

lint: ## Lint with ruff
	ruff check $(LINT_PATHS)

fmt: ## Format with ruff
	ruff format $(LINT_PATHS)

fmt-check: ## Check formatting without writing (CI-safe)
	ruff format --check $(LINT_PATHS)

tree: ## Show project tree, excluding common noise
	tree -I '__pycache__|*.pyc|.pytest_cache|.git|.venv|venv|dist|build'
