#!/usr/bin/env bash
# Bonsai GGUF bench servers (PrismML llama.cpp fork). Internal benchmarking
# only: routed through the :4000 gateway as remote_models, never managed by
# the vLLM orchestrator. See docs/models/ for model details.
set -euo pipefail

STUDIO_DIR="${STUDIO_DIR:-/teamspace/studios/this_studio}"
LLAMA_SERVER="${LLAMA_SERVER:-$STUDIO_DIR/llama.cpp-prism/build/bin/llama-server}"
BONSAI_1BIT_GGUF="${BONSAI_1BIT_GGUF:-$STUDIO_DIR/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf}"
TERNARY_GGUF="${TERNARY_GGUF:-$STUDIO_DIR/Ternary-Bonsai-27B-gguf/Ternary-Bonsai-27B-Q2_0.gguf}"
BONSAI_1BIT_PORT="${BONSAI_1BIT_PORT:-8005}"
TERNARY_PORT="${TERNARY_PORT:-8006}"
CTX_SIZE="${CTX_SIZE:-16384}"          # match registry defaults.max_model_len
RUN_DIR="$STUDIO_DIR/tunnel-engine/.bonsai"
HEALTH_TIMEOUT_S=120

start_server() {
    local name="$1" gguf="$2" port="$3"
    local pid_file="$RUN_DIR/$name.pid" log_file="$RUN_DIR/$name.log"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "$name already running (pid $(cat "$pid_file"))"
        return 0
    fi
    [[ -x "$LLAMA_SERVER" ]] || { echo "ERROR: $LLAMA_SERVER not found; run: $0 build" >&2; exit 1; }
    [[ -f "$gguf" ]] || { echo "ERROR: model file not found: $gguf" >&2; exit 1; }
    nohup "$LLAMA_SERVER" \
        --model "$gguf" \
        --alias "$name" \
        --port "$port" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers 999 \
        --jinja \
        --reasoning-budget 0 \
        >"$log_file" 2>&1 &
    echo $! >"$pid_file"
    echo "$name starting on :$port (pid $!, log $log_file)"
}

wait_healthy() {
    local name="$1" port="$2" waited=0
    until curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; do
        (( waited >= HEALTH_TIMEOUT_S )) && { echo "ERROR: $name not healthy after ${HEALTH_TIMEOUT_S}s; see $RUN_DIR/$name.log" >&2; return 1; }
        sleep 2; (( waited += 2 ))
    done
    echo "$name healthy on :$port"
}

stop_server() {
    local name="$1"
    local pid_file="$RUN_DIR/$name.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        kill "$(cat "$pid_file")" && echo "$name stopped"
    else
        echo "$name not running"
    fi
    rm -f "$pid_file"
}

status_server() {
    local name="$1" port="$2"
    local pid_file="$RUN_DIR/$name.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            echo "$name: running + healthy (:$port, pid $(cat "$pid_file"))"
        else
            echo "$name: process up but NOT healthy (:$port) — check $RUN_DIR/$name.log"
        fi
    else
        echo "$name: stopped"
    fi
}

mkdir -p "$RUN_DIR"
case "${1:-}" in
    build)
        git clone --depth 1 https://github.com/PrismML-Eng/llama.cpp "$STUDIO_DIR/llama.cpp-prism" 2>/dev/null || true
        cmake -S "$STUDIO_DIR/llama.cpp-prism" -B "$STUDIO_DIR/llama.cpp-prism/build" \
            -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DLLAMA_CURL=OFF
        cmake --build "$STUDIO_DIR/llama.cpp-prism/build" --target llama-server -j "$(nproc)"
        ;;
    up)
        start_server bonsai-27b-1bit "$BONSAI_1BIT_GGUF" "$BONSAI_1BIT_PORT"
        start_server ternary-bonsai-27b "$TERNARY_GGUF" "$TERNARY_PORT"
        wait_healthy bonsai-27b-1bit "$BONSAI_1BIT_PORT"
        wait_healthy ternary-bonsai-27b "$TERNARY_PORT"
        ;;
    down)
        stop_server bonsai-27b-1bit
        stop_server ternary-bonsai-27b
        ;;
    status)
        status_server bonsai-27b-1bit "$BONSAI_1BIT_PORT"
        status_server ternary-bonsai-27b "$TERNARY_PORT"
        ;;
    *)
        echo "usage: $0 {build|up|down|status}" >&2
        exit 1
        ;;
esac
