# =============================================================================
# SUBMISSION IMAGE — CPU-ONLY, linux/amd64, HYBRID MODE.
#
# Ollama sidecar + qwen2.5:3b baked at BUILD time (nothing downloads at
# runtime -> ready <60s). The local lane answers sentiment/ner/summarization
# for ZERO scored tokens, verifier-gated and fail-open: if the sidecar can't
# start, the agent silently runs the proven pure-remote path — it must NEVER
# die because of the sidecar. Everything else goes to Fireworks
# (FIREWORKS_API_KEY / FIREWORKS_BASE_URL / ALLOWED_MODELS injected at
# runtime — never baked in).
#
# Grading VM: 4 GB RAM / 2 vCPU / no GPU. Do NOT add CUDA/ROCm layers.
# Rollback anchor: ghcr.io/jubayeronrob/vinci-monsoon:a9a1d02 (pure remote).
#
# Build:  docker build --platform linux/amd64 -t hybrid-router:local .
# =============================================================================

FROM ollama/ollama:latest

# Python runtime for the agent (the base image is Ubuntu).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
    || pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Bake the local model INTO the image: start a temporary server, pull, stop.
# The weights layer ships with the image so runtime never downloads anything.
ARG LOCAL_MODEL=qwen2.5:3b
RUN /bin/ollama serve >/tmp/ollama-build.log 2>&1 & \
    i=0; \
    until /bin/ollama list >/dev/null 2>&1; do \
        i=$((i+1)); \
        if [ "$i" -gt 60 ]; then cat /tmp/ollama-build.log; exit 1; fi; \
        sleep 2; \
    done; \
    /bin/ollama pull "$LOCAL_MODEL" && /bin/ollama list

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py start.sh ./

# Windows checkouts can introduce CRLF — /bin/sh chokes on it. Strip always.
RUN sed -i 's/\r$//' /app/start.sh /app/entrypoint.py

# LOCAL LANE KILL-SWITCH: empty = lane OFF (pure proven remote lanes; no
# sidecar even starts). The grading harness injects no custom env, so this
# image default IS the submission behavior. Set to
# "sentiment,ner,summarization" to ship the full hybrid.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OLLAMA_MODEL=qwen2.5:3b \
    OLLAMA_KEEP_ALIVE=30m \
    LOCAL_CATEGORIES=""

# The base image's ENTRYPOINT is /bin/ollama — override with our launcher.
ENTRYPOINT ["/bin/sh", "/app/start.sh"]
