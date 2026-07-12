#!/bin/sh
# Launcher. The llama.cpp sidecar server starts ONLY when the local lane is
# enabled (LOCAL_CATEGORIES non-empty). With the lane off the runtime is pure
# remote: no sidecar process, no RAM/CPU contention — byte-identical to the
# proven remote-lane behavior that graded 89.47%.
# FAIL-OPEN either way: a sidecar that can't start never kills the agent;
# the LocalLane detects its absence and escalates every task to Fireworks.
if [ -n "$LOCAL_CATEGORIES" ]; then
    ( python3 -m llama_cpp.server \
        --model "${LOCAL_MODEL_PATH:-/models/model.gguf}" \
        --host 127.0.0.1 --port 8081 \
        --n_ctx 2048 --n_threads 2 --n_gpu_layers 0 \
        >/tmp/llama.log 2>&1 & ) || true
fi
exec python3 /app/entrypoint.py
