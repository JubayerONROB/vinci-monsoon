# =============================================================================
# SUBMISSION IMAGE — SLIM CPU-ONLY HYBRID (llama.cpp runtime), linux/amd64.
#
# DIAGNOSTIC ITERATION: the local lane scored a deterministic 0/19 on TWO
# runtimes (4.9 GB Ollama, 1.9 GB slim Ollama) while lane-OFF images grade
# 84-89% and never zero. This build swaps the model host to llama.cpp
# (llama-cpp-python server, in full control of n_ctx/threads/memory) and
# instruments every answer so a third zero yields a SIGNATURE, not a mystery.
#
# Lane defaults ON. Kill-switch: LOCAL_CATEGORIES="" => the llama.cpp server
# NEVER starts and behavior is byte-identical to the proven pure-remote lanes
# (:anchor-a9a1d02 = 89.47% remains the one-step rollback).
# Target <= ~2 GB compressed; CI hard-fails > 2.5 GB.
#
# Build:  docker build --platform linux/amd64 -t hybrid-router:local .
# =============================================================================

# ---------- Stage 1: build the CPU-only llama-cpp-python wheel ---------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Portable amd64 build: no -march=native (the grading CPU is unknown), no GPU.
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1
RUN pip install --no-cache-dir --prefix=/install "llama-cpp-python[server]"

# Bake the GGUF at BUILD time — nothing downloads at runtime.
ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && curl -fL --retry 3 -o /models/model.gguf "$MODEL_URL" \
    && du -sh /models

# ---------- Stage 2: slim runtime --------------------------------------------
FROM python:3.12-slim

# libgomp is the only extra runtime lib llama.cpp needs on CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /install /usr/local
COPY --from=builder /models /models

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py start.sh ./

# Windows checkouts can introduce CRLF — /bin/sh chokes on it. Strip always.
RUN sed -i 's/\r$//' /app/start.sh /app/entrypoint.py

# LOCAL LANE default ON. LOCAL_CATEGORIES="" = kill-switch: no server starts,
# pure proven remote lanes.
ENV LOCAL_MODEL_PATH=/models/model.gguf \
    LOCAL_SERVER_URL=http://127.0.0.1:8081 \
    LOCAL_CATEGORIES="sentiment,ner,summarization" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are injected by the
# grading harness at runtime — never baked into the image.
ENTRYPOINT ["/bin/sh", "/app/start.sh"]
