#!/bin/sh
# Launcher: Ollama sidecar in the background, agent in the foreground.
# FAIL-OPEN: if ollama can't start, the agent still runs (pure remote path);
# the LocalLane in src/local_models/local.py detects the dead sidecar and
# escalates every task to Fireworks. The agent must never die because of
# the sidecar, so the launch is fire-and-forget.
( /bin/ollama serve >/tmp/ollama.log 2>&1 & ) || true
exec python3 /app/entrypoint.py
