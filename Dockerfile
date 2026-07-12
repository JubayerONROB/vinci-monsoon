# =============================================================================
# SUBMISSION IMAGE — SLIM CPU-ONLY HYBRID, linux/amd64. TARGET <= ~2 GB.
#
# The previous hybrid (ollama/ollama base) was 4.9 GB compressed and scored a
# deterministic 0/19 twice; the 50 MB pure-remote image never scored 0. This
# image separates "image size" from everything else: slim python base + ONLY
# the CPU pieces of ollama (GPU/CUDA/ROCm libraries deleted at build) +
# qwen2.5:3b weights baked at BUILD time (nothing downloads at runtime).
#
# Local lane defaults ON (sentiment,ner,summarization; verifier-gated,
# fail-open). LOCAL_CATEGORIES="" reverts to the proven pure-remote lanes and
# the sidecar never even starts. Grading VM: 4 GB RAM / 2 vCPU / no GPU.
# Rollback anchor: ghcr.io/jubayeronrob/vinci-monsoon:anchor-a9a1d02.
#
# Build:  docker build --platform linux/amd64 -t hybrid-router:local .
# =============================================================================

# ---------- Stage 1: fetch CPU-only ollama + bake the model ------------------
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Official linux-amd64 bundle, then strip EVERY GPU artifact: the grading box
# is CPU-only, so CUDA/ROCm runners are pure pull-time dead weight.
RUN curl -fL --retry 3 https://ollama.com/download/ollama-linux-amd64.tgz \
        -o /tmp/ollama.tgz \
    && mkdir -p /opt/ollama \
    && tar -xzf /tmp/ollama.tgz -C /opt/ollama \
    && rm /tmp/ollama.tgz \
    && find /opt/ollama -depth -type d \( -iname 'cuda*' -o -iname 'rocm*' \) \
         -exec rm -rf {} + \
    && find /opt/ollama -type f \( -iname '*cublas*' -o -iname '*cudart*' \
         -o -iname '*nvml*' -o -iname '*hipblas*' -o -iname '*rocblas*' \
         -o -iname '*amdhip*' \) -delete \
    && echo "--- ollama payload after GPU strip:" && du -sh /opt/ollama \
    && /opt/ollama/bin/ollama --version

# Bake qwen2.5:3b INTO the image under a fixed models dir.
ARG LOCAL_MODEL=qwen2.5:3b
ENV OLLAMA_MODELS=/opt/models
RUN /opt/ollama/bin/ollama serve >/tmp/ollama-build.log 2>&1 & \
    i=0; \
    until /opt/ollama/bin/ollama list >/dev/null 2>&1; do \
        i=$((i+1)); \
        if [ "$i" -gt 60 ]; then cat /tmp/ollama-build.log; exit 1; fi; \
        sleep 2; \
    done; \
    /opt/ollama/bin/ollama pull "$LOCAL_MODEL" && /opt/ollama/bin/ollama list \
    && du -sh /opt/models

# ---------- Stage 2: slim runtime --------------------------------------------
FROM python:3.12-slim

# libstdc++/libgomp: the bundled llama runners need them; slim may lack them.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libstdc++6 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /opt/ollama /opt/ollama
COPY --from=builder /opt/models /opt/models

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py start.sh ./

# Windows checkouts can introduce CRLF — /bin/sh chokes on it. Strip always.
RUN sed -i 's/\r$//' /app/start.sh /app/entrypoint.py

# LOCAL LANE default ON (the kill-switch: set LOCAL_CATEGORIES="" to get the
# proven pure-remote lanes with no sidecar at all).
ENV PATH="/opt/ollama/bin:${PATH}" \
    OLLAMA_MODELS=/opt/models \
    OLLAMA_MODEL=qwen2.5:3b \
    OLLAMA_KEEP_ALIVE=30m \
    LOCAL_CATEGORIES="sentiment,ner,summarization" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are injected by the
# grading harness at runtime — never baked into the image.
ENTRYPOINT ["/bin/sh", "/app/start.sh"]
