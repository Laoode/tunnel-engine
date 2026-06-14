PYTHON      := python3
REGISTRY    := configs/models.yaml
LINT_PATHS  := tunnel/ tests/
HF_CACHE    := ~/.cache/huggingface/hub/

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

start: ## Health-gate + proxy: wait for all vLLM instances, then launch proxy
	$(PYTHON) -m tunnel.cli start

start-timeout: ## Same as start with custom timeout. Usage: make start-timeout TIMEOUT=120
	@if [ -z "$(TIMEOUT)" ]; then echo "Usage: make start-timeout TIMEOUT=<seconds>"; exit 1; fi
	$(PYTHON) -m tunnel.cli start --timeout $(TIMEOUT)

serve: ## Launch a vLLM instance. Usage: make serve ID=qwen-0.8b
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

test-integration: ## Run integration tests (requires: make proxy + vLLM instances)
	$(PYTHON) -m pytest tests/integration/ -v -m integration

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

kill:
	pkill -f vllm

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