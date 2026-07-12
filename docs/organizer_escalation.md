# Escalation: repeated TIMEOUT verdicts despite in-container runtime of ~150s

**Team:** vinci-monsoon (Track 1)
**Image:** `ghcr.io/jubayeronrob/vinci-monsoon:latest` (public, linux/amd64)

## Summary

Our last three graded submissions were reported as TIMEOUT, while our container's
own wall-clock instrumentation shows the process finishing in ~150 seconds —
roughly a quarter of the 10-minute budget. We believe the overrun is happening
**outside our process** (image pull, scheduling, or harness overhead) and would
like to confirm how that time is accounted.

## Evidence

Our container writes `/output/diag.json` with wall-clock stamps taken inside the
process:

- `container_start_utc` — timestamp at Python process launch (first line of the
  entrypoint, before any imports of ours)
- `container_end_utc` — timestamp when results were flushed
- `total_elapsed_secs`, `startup_secs`, and per-task timings

Observed behaviour across submissions:

| Submission (commit) | Our measured runtime | Platform verdict |
| --- | --- | --- |
| `05447a9` | ~149 s in CI, identical code path | TIMEOUT |
| `ddf139b` (worst-case redesigned to <352 s) | 145.2 s live run, all `finish=stop` | TIMEOUT |
| `787cfb3` (image shrunk 2.0 GB → 527 MB) | 75.6 s live run | — |

Design worst case after `ddf139b` is provably under 520 s (per-request timeout
12 s, one attempt per task, crash-safe early cutover), yet verdicts did not
change, which is why we suspect non-process time.

The current image is ~150 MB compressed (all-remote architecture, no bundled
model), so pull time should now be negligible — but the diag stamps will settle
the question either way on the next graded run.

## Questions

1. Does image pull / container scheduling time count against the 10-minute
   budget? If so, is there a way to pre-pull or warm the image?
2. If a run is marked TIMEOUT, can we obtain the harness-side timestamps
   (pull start, container start, kill time) or the `/output/diag.json` our
   container wrote, so we can pinpoint where the time went?
3. Were partial outputs (`results.json` is written incrementally-safe and
   flushed in a `finally`) present for the timed-out runs?

Thank you — happy to provide the full diag files from our CI runs if useful.
