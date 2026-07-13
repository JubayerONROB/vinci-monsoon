#!/bin/sh
# Launcher. The Ollama sidecar starts ONLY when the local lane is enabled
# (LOCAL_CATEGORIES non-empty). With the lane off the runtime is pure remote:
# no sidecar process, no RAM/CPU contention, exactly the proven remote-lane
# behavior that graded 89.47%.
# FAIL-OPEN either way: a sidecar that can't start never kills the agent;
# the LocalLane detects its absence and escalates every task to Fireworks.
if [ -n "$LOCAL_CATEGORIES" ]; then
    ( /opt/ollama/bin/ollama serve >/tmp/ollama.log 2>&1 & ) || true
fi
exec python3 /app/entrypoint.py
