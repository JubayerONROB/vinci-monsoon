# Hybrid Router — AMD Hackathon ACT II, Track 1

A token-efficient general-purpose agent. It reads a batch of tasks, answers
each of 8 categories correctly, and tries to spend as few Fireworks tokens as
possible doing it — because after an accuracy gate, **fewer tokens wins.**

## Harness contract

```
/input/tasks.json   →  [{"task_id": "...", "prompt": "..."}, ...]
                              │
                       (this container)
                              │
/output/results.json  →  [{"task_id": "...", "answer": "..."}, ...]
exit 0
```

Scoring is two stages: an LLM judge first gates on accuracy (this track
requires ≥16/19 to pass); submissions that clear the gate are then ranked by
total Fireworks tokens spent, fewest wins. Getting an answer right cheaply
only counts if it's still right — the router never trades accuracy for
tokens on a category proven to need the extra spend.

## Architecture as shipped

```
/input/tasks.json
      │
      ▼
Stage 1 — classify LOCALLY (deterministic keyword heuristic, 0 tokens, instant)
      {factual | math | sentiment | summarization | ner | code_debugging |
       logical_reasoning | code_generation}
      │
      ▼
Stage 2 — per-category lane (config/routing_map.yaml)
      ├─ sentiment / ner / summarization
      │     → LOCAL Ollama sidecar (qwen2.5:3b, baked into the image)
      │       verifier-gated, fail-open, ZERO scored tokens
      │       (bad shape / timeout / error → escalate to remote, never silently wrong)
      │
      └─ everything else, plus any local escalation
            → category → role → Fireworks model (ALLOWED_MODELS, resolved at runtime)
                 factual, math            → minimax-m3   (math: reasoning_effort=low)
                 logic, debug, codegen    → kimi-k2p7-code
      │
      ▼
/output/results.json  →  exit 0
```

Every task is dispatched from a `ThreadPoolExecutor` in parallel. Model IDs
are **never hardcoded** — each role resolves at runtime by case-insensitive
substring match against the `ALLOWED_MODELS` env var, with graceful fallback
to the first non-gemma entry (gemma is an on-demand, expensive-to-idle model
and is deliberately never a routing target). Edit the hints in
[config/routing_map.yaml](config/routing_map.yaml) if the platform's model
list changes.

### Local lane (zero-token answers)

An Ollama sidecar with `qwen2.5:3b` baked into the image at build time (no
runtime download) answers sentiment, NER, and summarization tasks directly.
Every local answer is checked by a deterministic verifier for that category
(sentence/bullet-count discipline, entity-list shape and completeness,
sentiment-label validity) before it ships. Any failure — timeout, malformed
output, or a verifier reject — escalates the task to the paid remote lane
instead of shipping a bad free answer: the lane can only save tokens, never
cost accuracy. Setting `LOCAL_CATEGORIES=""` disables the sidecar entirely
(it never even starts) and falls back to pure-remote lanes.

### Reliability

- **Parallel dispatch**: all tasks submitted to a thread pool at once
  (`MAX_WORKERS`, default 5), not processed one at a time.
- **Bounded remote calls**: every Fireworks request has its own timeout with
  a single retry on transient failures; an empty completion gets one retry
  on the sibling allowed model.
- **Global deadline**: one hard wall-clock deadline anchored at process
  launch bounds the whole run; any task still outstanding when it hits gets
  a deterministic non-blank fallback answer so the output is always
  complete and schema-valid.
- **Crash-safe output**: results are written exactly once, from a single
  flush path that also runs on the crash/exception path — a run can never
  exit without a `results.json`, and an answer is never blank.
- **Background model warm-up**: the local model is warmed asynchronously at
  startup so it's ready without blocking task dispatch.
- Compressed image ~1.9 GB, CPU-only, `linux/amd64`; container is ready in
  well under a minute, comfortably inside the harness's time budget.

## Repo layout

| Path | Purpose |
| --- | --- |
| `entrypoint.py` | Harness contract: read tasks, dispatch in parallel, enforce the global deadline, write outputs |
| `src/router/classifier.py` | Stage-1 keyword-heuristic category classifier |
| `src/router/dispatch.py` | Stage-2 category → role → model resolution and remote/local dispatch |
| `src/router/formatting.py` | Answer post-processing/formatting safety net |
| `src/local_models/local.py` | Local Ollama lane: verifiers, prompt building, fail-open escalation logic |
| `src/local_models/loader.py` | Deterministic fallback answer generator |
| `src/api_clients/fireworks.py` | Fireworks client: retries, timeouts, thread-safe token accounting |
| `config/routing_map.yaml` | Category→role→model hints, token caps, thresholds — all policy knobs |
| `config/prompts.py` | System prompt + per-category output-format instructions |
| `tests/` | Mock tasks, offline eval harness, pytest suite (schema, verifiers, model resolution, output integrity, lane isolation) |
| `docs/HISTORY.md` | Full engineering log: every commit, why, and what it taught us |

## Local development

```bash
pip install -r requirements-dev.txt
make test                       # offline eval + full pytest suite (no key, no network)
# or individually:
python tests/run_eval.py
python -m pytest tests/ -q
```

## Build & run

```bash
docker build --platform linux/amd64 -t hybrid-router:local .
```

Smoke test exactly like the harness:

```bash
docker run --rm \
  -v "$PWD/tests/mock_tasks.json:/input/tasks.json:ro" \
  -v "$PWD/out:/output" \
  -e FIREWORKS_API_KEY -e FIREWORKS_BASE_URL -e ALLOWED_MODELS \
  hybrid-router:local
cat out/results.json
```

`FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL`, and `ALLOWED_MODELS` are injected
by the grading harness at runtime and must never be hardcoded or baked into
the image. `LOCAL_CATEGORIES` (default `sentiment,ner,summarization`) is the
only other environment knob that changes runtime behavior; set it to an
empty string to disable the local lane.

## Pre-submission checklist

- [x] Reads `/input/tasks.json`, writes valid `/output/results.json`, exit 0
- [x] Env-driven key/base-url/models — nothing hardcoded, all remote calls via the proxy
- [x] Per-request timeout well under the harness's per-call cap; single global deadline under the 10-minute run cap, with headroom
- [x] CPU-only `linux/amd64` image, ~1.9 GB compressed, well under the size cap
- [x] No hardcoded/cached answers; every task try/except-guarded; deterministic non-blank fallback if a step fails
- [x] Local-lane answers pass a category-specific verifier before shipping; any doubt escalates to remote rather than risking accuracy
- [x] Never routes to gemma
- [x] Offline test suite (schema, verifiers, model resolution, output integrity, lane isolation) green before every submission
