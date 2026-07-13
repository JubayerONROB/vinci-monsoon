# Engineering History — AMD Hackathon ACT II, Track 1

A commit-by-commit log of how this router got built, what broke, and what we
learned. Oldest to newest, grouped by phase. "GRADED" marks a real submission
to the AMD platform; everything else is CI/mock-only.

## At-a-glance: every graded run

| Commit | Config | Tokens | Accuracy | Note |
| --- | --- | --- | --- | --- |
| `75bd6b1` | All-remote, parallel dispatch, 50 MB image | 6,245 | **94.7% (18/19)**, rank 42 | First submission to clear the gate — proved the earlier TIMEOUTs were internal, not image-pull |
| `a9a1d02` | Token-reduction pass incl. re-laning factual/sentiment/summarization/NER onto kimi | 5,247 | **84.2% (16/19)**, rank 31 | Regression: re-laning accuracy-sensitive categories to save prompt overhead cost 2 gate points |
| rollback anchor (`:anchor-a9a1d02`) | Re-laning reverted (general role back to minimax) + local lane added | ~5,500 | **89.47% (17/19)** | Established as the rollback floor; still one task short of the 18/19 config |
| 4.9 GB Ollama hybrid (`71b2b7e`-era) | Local lane ON | — | **0/19** | First deterministic zero with the lane enabled |
| 1.9 GB slim hybrid (`436a91f`/`d6fcd4d`-era) | Local lane ON, same lane logic, different image | — | **0/19** | Second deterministic zero — image size ruled out as the cause, since two very different images both zeroed under the same lane |
| `032c1a5` | llama.cpp diagnostic swap (same lane contract) | — | **0/19**, then **94.7%** on a regrade of identical code ~1 hour later | **Platform flakiness, not a code defect** — see "The 0/19 saga" below |
| `d6fcd4d` | Slim hybrid, local lane ON, proven remote lanes | 6,276 | **94.7% (18/19)** | Current best graded baseline |
| `5566a54` | `d6fcd4d` + remote-only token squeeze | ~6,150–6,250 (CI-measured, not yet graded) | not yet submitted | Awaiting approval to spend the last resubmit |

---

## Phase 0 — Early infrastructure

**`1580b22` — Initial architecture for Track 1 Router.** Local-first design:
a llama.cpp classifier + local GGUF answered as much as possible, escalating
only hard tasks to Fireworks.

**`8fb83ab` — Add GHCR build/test/publish workflow.**
WHY: needed CI that builds, smoke-tests, and pushes the image on every push.

**`857b735` — Fix image size report for OCI index manifests.**
WHY: the size-reporting step in CI mis-measured multi-arch manifests (an
index, not a flat manifest), producing wrong numbers and masking early
manifest/pull-class errors on the registry side.
WHAT CHANGED: size report walks the manifest list and sums the correct
platform's layers.

**`7414cb4` / `865ab13` — Add manual live Fireworks integration-test job / EVAL_LIVE mode.**
WHY: needed a way to exercise real Fireworks calls (not the offline mock) from
CI, on demand, without spending tokens on every push.

**`192299d` — Set real Track 1 model IDs (minimax-m3, kimi-k2p7-code).**
Switched from placeholder model IDs to the published Track 1 list.

## Phase 1 — The TIMEOUT saga

The container kept coming back TIMEOUT from the grading harness. This phase
is the trial-and-error hunt for why, one hypothesis at a time.

**`736345f` — Force-escalate hard categories; add hard mock tasks.**
HYPOTHESIS: the local model was silently failing on hard math/logic/code and
returning garbage fast (looking like success, not a timeout). WHAT CHANGED:
math/logic/code categories force-escalate to Fireworks regardless of the
local classifier's confidence; added harder mock tasks to catch this class
of failure offline.

**`1cd8b44` — Raise remote timeout + cap reasoning tokens to fix logic fallback.**
FINDING: logic tasks were falling back locally because the remote call
timed out before a reasoning-heavy completion finished.

**`309622d` — Fix TIMEOUT: logic→kimi, no-retry on empty content, global time budget.**
WHY: empty completions were triggering a retry loop that ate the run's time
budget. WHAT CHANGED: logic role moved to kimi; empty content no longer
retries; added a first global time budget.

**`50b6d9d` — Accept non-empty truncated completions; raise math cap to 600.**
FINDING: a `finish_reason=length` response was being discarded as a failure
even when it contained a usable (if truncated) answer — wasting the tokens
already spent for nothing. WHAT CHANGED: accept truncated-but-non-empty
answers; raised the math token cap so fewer answers truncate in the first
place.

**`542fc0e` — Aggressive escalation + per-task diagnostics + raise code cap (gate fix).**
Added per-task diagnostic logging (first version of `diag.json`) so timeout
hunting stopped being blind guesswork.

**`caf1a7f` — reasoning_effort=none; factual→remote; timeout guard 480s; heuristic→all-remote.**
WHY: still chasing the timeout; tried the cheapest possible reasoning
setting and moved factual fully remote. First explicit global timeout guard
(480s) as a backstop.

**`bd0e9c7` — Hard per-task time cap + dynamic global budget + tighter escalation (fix TIMEOUT).**
Per-task cap plus a budget that shrinks as the run consumes wall-clock,
so later tasks escalate faster instead of each independently risking the
full per-task allowance.

**`4ab7bb8` — Minimax general lane + low-effort reasoning + output-format discipline + diag.json.**
Introduced the general role → minimax-m3 mapping and `reasoning_effort=low`
for math (the setting that later proves load-bearing — see Phase 3). First
named `diag.json` file for structured run diagnostics.

**`1ff6d36` — Persist results_full.json for offline answer inspection + judging.**
WHY: `results.json` alone (task_id + answer) wasn't enough to debug
*why* an answer was wrong offline — added a full dump (prompt + answer +
routing metadata) for manual/offline judging.

**`ab690d1` — TIMEOUT fix: lazy-load local GGUF + startup instrumentation + budget guard counts from process launch.**
FINDING (important): the global budget guard had been counting from some
in-process marker, not from when the container actually started — under
container/model load latency, that gap alone could eat a meaningful chunk of
the 10-minute budget unaccounted for. WHAT CHANGED: budget now anchors at
process launch; the local GGUF loads lazily instead of blocking startup;
added startup timing instrumentation.
LESSON: this is the direct ancestor of `CONTAINER_START_TS` in the current
`entrypoint.py` — anchoring every deadline at true process launch, not a
later checkpoint, is a rule that survived every later rewrite.

**`05447a9` — Accuracy: ner→remote, sentiment justify-preserve, logic→minimax-low.**
Accuracy tuning pass alongside the ongoing timeout work: NER moved fully
remote (local was unreliable on it), sentiment prompt tweaked to preserve
justification text, logic tried minimax at low effort.

**`ddf139b` — TIMEOUT fix v2: revert logic→kimi, 12s timeout, disable alternate, worst-case<520s.**
FINDING: the "try an alternate model on failure" retry path was itself a
worst-case-time risk — two sequential remote calls per task, uncapped, could
blow the deadline. WHAT CHANGED: reverted logic to kimi, tightened the
per-request timeout to 12s (the value still in use today), and disabled the
alternate-model retry except for the one case proven safe later (empty
completions only). Computed and asserted a worst-case runtime bound
(<520s) for the first time.

**`787cfb3` — Shrink image (smaller GGUF) to cut pull time + record start/end wall-clock in diag.**
HYPOTHESIS at the time: image pull time itself might be eating into the
run budget on the grading box. Swapped to a smaller GGUF and started
recording container start/end UTC timestamps in `diag.json` as evidence for
the organizers, in case a TIMEOUT verdict was a platform-side artifact
rather than ours.

## Phase 2 — The all-remote breakthrough (first passing grade)

**`20b1e50` — All-remote + parallel dispatch: drop GGUF, ThreadPool, global deadline, wall-clock diag.**
WHY: after a full phase of incremental timeout patches, the decision was to
remove the suspect entirely rather than keep patching it. WHAT CHANGED:
the bundled GGUF and llama-cpp-python were deleted outright — image dropped
from ~2.0 GB to ~150 MB. Classification became a deterministic keyword
heuristic (0 tokens, instant, no model-load risk). Every task now dispatches
remotely, in parallel, from a `ThreadPoolExecutor`. A single global hard
deadline (500s from process launch) replaced the patchwork of per-task/
dynamic budgets. `fireworks.py` became thread-safe (locked counters,
thread-local finish-reason tracking) to support concurrent dispatch safely.
RESULT: **GRADED 94.7% (18/19) @ 6,245 tokens, rank 42** on `75bd6b1`
(the immediate follow-up commit, which only added the organizer escalation
draft on top).
LESSON: the small local GGUF and its load/inference latency — not image
pull time — had been the real timeout risk all along. Removing the variable
entirely beat every attempt to tune around it.

**`75bd6b1` — Add organizer escalation draft with wall-clock diag evidence.**
Documentation commit capturing the timeline of TIMEOUT verdicts alongside
the `container_start_utc`/`container_end_utc` evidence, in case an appeal to
the organizers was ever needed. This is the commit that was actually graded
94.7%.

## Phase 3 — Token reduction, and the regression it caused

**`8c4238f` — Diversify mocks with FAQ validation set + per-task token instrumentation.**
WHY: the existing mock set was thin and didn't match the judge's own public
examples. WHAT CHANGED: folded in the AMD Hackathon Judging FAQ's public
validation tasks (T01–T05 and variants) verbatim into `tests/mock_tasks.json`,
fixed a mock logic puzzle that had more than one valid solution, and added
per-call token capture (`usage.prompt_tokens`/`completion_tokens`) to the
Fireworks client so every subsequent tuning decision could be measured
instead of guessed.

**`803449b` — Token reduction: output caps + trimmed prompts + reasoning tune + per-task token diag.**
Calibrated from a 29-mock measurement run at 7,518 tokens:
- math (33% of total spend): `reasoning_effort` tried at `"none"` (cap
  600→450) instead of `"low"`.
- logic: style changed from step-by-step to conclusion-plus-one-line
  justification; code role cap 700→500.
- debug/codegen: code-only output by default, prose only if the prompt
  explicitly asked for it.
- general lane cap 512→300 (observed worst-case completion was 98 tokens).

**`a9a1d02` — Revert failed A/B cuts (math effort, logic style); trial kimi on general role.**
FINDING from judged mock review: `reasoning_effort="none"` got a multi-phase
math problem (a tank-fill-and-drain puzzle) wrong — it skipped an entire
phase of the calculation; reverted to `"low"` + cap 600. Conclusion-only
logic broke a constraint puzzle — for this model, the visible step-by-step
reasoning IS the thinking, not just narration of it; reverted to
step-by-step-brief. Kept the clean wins (code-only debug/codegen, general
cap 300). New experiment in the same commit: general role (factual,
sentiment, summarization, NER) moved from minimax to kimi, on the
measurement that minimax carries a ~110-token fixed prompt overhead per
call versus kimi's ~10.
RESULT: **GRADED 84.2% (16/19) @ 5,247 tokens, rank 31.**
LESSON (the important one): the 29-mock self-judged smoke test showed no
quality loss from the kimi re-laning. The graded run dropped two gate
points anyway. **A clean mock self-judge does not predict graded accuracy
on categories the judge weighs differently than our own rubric-reading
does — token cuts on accuracy-sensitive categories are a bet against gate
margin, not a free win, no matter how clean the local smoke test looks.**

**`a6ad322` — Local-model lane (verifier-gated fail-open) + revert factual/sentiment/ner/summ off kimi + zero-token telemetry.**
Two things in one commit. PART B directly answers the regression above:
general role reverted back to minimax (kimi re-laning was the prime
suspect for the 84.2% drop); math/logic/debug/codegen unchanged. PART A
introduces the Ollama local lane for the first time: a `qwen2.5:3b` sidecar
baked into the image at build time, tried first for sentiment/ner/
summarization, serialized behind one lock, escalating to remote on error,
timeout, `done_reason=length`, or a deterministic verifier reject (sentence/
bullet-count checks, entity-list shape, sentiment-label validity — a
contrastive review labelled purely "Negative" always escalates per the
FAQ's own T03 rubric). A deadline guard skips the local attempt entirely
once too little run budget remains. `diag.json` gains local/remote task
counts; local answers report 0 prompt/completion tokens.
This commit's build became the rollback anchor
(`ghcr.io/jubayeronrob/vinci-monsoon:a9a1d02`, later retagged
`:anchor-a9a1d02`).
RESULT: **GRADED 89.47% (17/19) @ ~5,500 tokens** on the anchor build —
back near, but not fully at, the 18/19 config.

**`e637a02` — CI: gate on verifier tests; add manual GHCR retag workflow for anchors.**
Infra: CI now fails the build if the new verifier tests fail; added a
manual `workflow_dispatch` job to retag a known-good image as a rollback
anchor (the local PAT lacked package-write scope; `GITHUB_TOKEN` in Actions
has it).

**`43477ba` — NER completeness gate + summarization timeout headroom + escalation-reason logs.**
FINDING: a judged run showed the local model dropping an entity ("Baltic
Dynamics") from a dense NER task — the existing shape-only verifier
couldn't see a *missing* item, only a malformed one, and the FAQ's own T05
rubric fails on any missing entity. WHAT CHANGED: a completeness gate
extracts capitalized runs and date patterns from the source text and
requires all of them to appear in the local answer, else escalate.
Separately, summarization was escalating 3/3 on a timeout that was really
just CPU-bound local generation being slower than expected — timeout
widened with length-based headroom (local time is deadline-guarded and
effectively free, so this costs nothing but latency). Every escalation now
logs its reason (timeout/empty/length/verifier), turning lane tuning from
blind guessing into something measurable.

**`71b2b7e` — Local lane: retry transient 5xx/429 during warm-up, log HTTP status/body, relax dead-lane rule.**
FINDING: a run showed the lane going completely dead after 2 early
sentiment `HTTPError`s that turned out to be the sidecar still loading its
model, not a real failure — the run got zero local hits it otherwise would
have earned. WHAT CHANGED: transient server-busy responses inside a 150s
warm window now retry within the call's own timeout; the lane is only
declared dead on connection-level absence, or repeated failures after the
warm window has passed.

## The 0/19 saga

Local-lane images then hit **two separate deterministic 0/19 grades** — once
on the 4.9 GB `ollama/ollama`-based hybrid (`71b2b7e`-era), and again, after a
full image rebuild, on the 1.9 GB slim hybrid (`436a91f`/`d6fcd4d`-era, next
section). Both used the same lane logic; the only thing they had in common
across two very different images was the local lane being enabled.

**`7737143` — 0/19 postmortem: structural defects ruled out by test; ship lane-off default + grading sandbox.**
Rather than guess again, this commit is a structural audit. New tests
proved: `task_id` is preserved and aligned under judge-style ids (numeric,
UUID, whitespace-padded), the output is a bare, correctly-shaped array,
exactly one write happens (with a crash-safe flush path), and model
resolution is gemma-proof under every possible `ALLOWED_MODELS` permutation.
**No structural output bug was found.** The working theory shifted to the
runtime environment (the sidecar under 4 GB RAM / 2 vCPU) or a uniform
remote failure mode (well-formed-but-wrong fallback answers, which the new
sandbox checker can now detect directly against a raw `results.json`).
Hardening shipped regardless: `format_answer` wrapped per-task so a
formatter exception degrades to the raw answer instead of crashing the run;
`resolve_role`'s fallback path now explicitly skips gemma; and — the
decision that mattered most at the time — **`LOCAL_CATEGORIES=""` became
the image default**, so the very next submission would ship pure proven
remote lanes with the sidecar not even started, verified by a unit test that
asserts zero lane HTTP traffic when disabled. A new CI sandbox step ran the
full hybrid under `--cpus=2 -m 4g` with unseen task ids and checked the raw
output with objective, non-judging probes (known arithmetic answers, exact
entity sets) — deliberately never using an LLM to self-judge.

**`436a91f` — Slim CPU-only ollama base (~2GB) + verifier-gated local lane on the 89.47% Part-B lanes.**
WHY: the 4.9 GB `ollama/ollama` base (which bundles GPU/CUDA/ROCm libraries
never used on the CPU-only grading box) was the prime remaining suspect for
the 0/19s. WHAT CHANGED: switched to `python:3.12-slim` plus the official
CPU-only ollama release binary with every GPU artifact deleted at build
time, `qwen2.5:3b` baked into `/opt/models` (nothing downloads at runtime).
CI now hard-fails any build over 2.5 GB. Local lane defaults back ON
(sentiment/ner/summarization) with all the `71b2b7e` behavior (warm-window
retry, NER completeness gate, deadline guard, kill-switch) layered on top of
the proven 89.47% remote lanes.

**`d6fcd4d` — Fix ollama fetch: pinned GitHub release tar.zst (ollama.com tarball URL 404s).**
Build fix: `ollama.com/download`'s tarball URL 404s in CI; switched to the
pinned `github.com/ollama/ollama/releases` asset. No behavioral change.

**`032c1a5` — Diagnostic: llama.cpp lane runtime + full answer forensics + lane-isolation proof.**
By this point the lane had zeroed grading on two structurally-unrelated
images while every local sandbox reproduction stayed clean — the lane
itself was the only constant, but nothing in the code could explain it.
This commit swapped the local runtime from Ollama to a `llama-cpp-python`
server (same GGUF, pinned context/thread settings — Ollama's own runtime
defaults were uncontrolled) purely to see if a completely different local
inference stack changed the outcome, and added the evidence needed to stop
guessing: a new isolation test proving byte-identical remote answers
whether the lane is off, dead, or actively serving (lane presence literally
cannot alter a remote call's model, prompt, caps, or effort at the code
level); `diag.json` rows now carry the exact raw answer string as shipped;
`DIAG_ECHO=1` prints every shipped row to stderr at write time; and the
sandbox was hardened to disable swap (`--memory-swap` equal to `-m`),
matching a real difference between CI runners (which have swap) and the
likely grading box (which may not).
RESULT: **GRADED 0/19**, then, on a regrade of the *same, unchanged*
submission roughly an hour later, **94.7%.**
**FINDING — the single most important lesson from this project: the 0/19s
were platform-side grading flakiness, not a defect in this code.** Every
structural test passed, the lane-isolation proof showed the lane cannot
touch remote answers, sandboxes under matched resource constraints never
reproduced a zero, and identical code graded both 0 and 94.7% within the
same hour. No commit "fixed" the 0/19 — the code never had a bug to fix.
LESSON FOR NEXT TIME: don't chase a deterministic-looking failure with more
code changes once structural tests, isolation proofs, and matched-resource
sandboxes all come back clean — at that point the failure is more likely
outside the container than inside it, and the only real test is a regrade.

## Phase 4 — Settling on d6fcd4d, then squeezing tokens without risk

With the platform-flakiness explanation established, `d6fcd4d` — slim image,
local lane on, the proven remote lane map — is the settled best-known
config.
RESULT: **GRADED 94.7% (18/19) @ 6,276 tokens** — matching the earlier
all-remote high-water mark while adding a handful of zero-token local hits.

**`f926df7` — Remote token squeeze: output caps + trimmed prompts on non-gated lanes only.**
Goal: reduce tokens further with zero additional accuracy risk on top of a
config that already cleared the gate at 18/19. Image, `start.sh`, and the
entire local lane were reverted byte-exact to `d6fcd4d` — nothing about the
lane or the proven remote model map changed. Remote-only changes,
calibrated against a fresh per-category token measurement of this exact
image:
- Per-category token caps as blowup insurance (2.5–4x above observed peaks
  per category) on debug/codegen/logic. Math was deliberately left
  uncapped at its existing role-level ceiling (accuracy-critical — see the
  `803449b`/`a9a1d02` math-effort lesson above). Factual was left
  untouched (a prior under-60-word cut had regressed a competitor's grade
  to 73.7%, so factual prompt length is not touched without strong
  evidence).
- Prompt trims: the shared remote system prompt and several per-category
  style hints were shortened. "Answer in English" was deliberately kept —
  language drift is a guaranteed judge fail and the token cost of keeping
  it is negligible. Factual, summarization, NER, and logic style strings
  were left verbatim (format- or accuracy-critical).
- `diag.json`'s category breakdown gained mean prompt/completion tokens
  per category, not just totals.

**`5566a54` — Restore proven math style ('Brief working only'); fix contrastive-sentiment probe to check label token.**
Two fixes surfaced during verification of the squeeze above, before any
resubmission:
1. Measurement showed dropping the word "only" from the math style string
   (a wording change that looked purely cosmetic) coincided with +234
   completion tokens across 5 math tasks in one run — reverted to the
   exact `d6fcd4d` wording. LESSON: on this model, "brief working" and
   "brief working *only*" are not the same instruction — a single word can
   change verbosity meaningfully, and cheap-looking wording trims still
   need before/after token measurement, not just a read-through.
2. The CI sandbox's own correctness probe for contrastive sentiment (a
   review with both negative and positive elements) was checking the first
   40 characters of the answer for the substring "negative" — which false-
   failed on a fully correct answer like *"**Mixed** — ...negative elements
   ... outweighed by strong positive sentiment"*. Fixed to extract and check
   only the first alphabetic token (the actual label), not any substring
   of the justification.
STATUS at time of writing: verified via a full 29-task judged review (29/29
pass, no regressions), a CI integration run (6,364 tokens vs the 6,410
same-image mock baseline, all answers correct, no fallbacks/truncation/
blanks/gemma), and a 19-task grading-condition sandbox (`SANDBOX CHECK OK`).
**Not yet submitted for grading** — one resubmit slot remains and is being
held for explicit sign-off on the numbers above before it's spent.

## Key lessons

- **Local tokens are free, but the lane only offloads a handful of cheap
  categories.** The real graded saving from the local lane is on the order
  of a few percent, not the larger number a naive token-count projection
  might suggest — size expectations accordingly.
- **A clean self-judged mock run does not guarantee a clean graded run,**
  especially for token cuts on categories the judge may weight differently
  than an internal rubric reading suggests (`a9a1d02`'s kimi re-laning).
- **Structural bugs and platform flakiness look identical from the outside
  as "our score is wrong."** Only a structural audit (schema/task_id/output-
  path tests, isolation proofs, matched-resource sandboxes) can tell them
  apart — and once that audit comes back clean, more code changes won't fix
  a problem that was never in the code (`032c1a5`).
- **Never leave a bad `latest` sitting as the submittable image.** Every
  regression in this log was followed immediately by either a revert or an
  explicit, tested rollback anchor — never a "we'll fix it next round."
- **Don't resubmit just to double-check.** A result is either deterministic
  (a resubmit teaches nothing new) or flaky (a resubmit is a coin flip) —
  in both cases, spend the scarce resubmit budget on a change you have
  reason to believe moves the needle, not on reassurance.
- **Keep worst-case runtime bounded and anchored at true process launch**
  (`CONTAINER_START_TS`), not a later in-process checkpoint — this one
  change (`ab690d1`) closed a whole class of budget-accounting bugs.
- **Never route to gemma.** It is the one model on the allowed list that is
  expensive to leave idle; every fallback path explicitly skips it.
