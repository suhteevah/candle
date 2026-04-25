# TDPI loop — Test-Driven Performance Iteration

The discipline this sub-tree enforces:

> **No claimed speedup ships without a measured bench diff in the commit message.**

The runner + hypothesis driver here verify each `qwen-lora-train` /
`qwen-lora-serve` change against locked baselines. Losing experiments
get logged to `PERF_LOG.md` with the reason; winning experiments
update `BENCH_BASELINE.json` in place (with the previous value pushed
to a `_history` array so we can audit drift).

## Files

- `BENCH_BASELINE.json` — locked numbers per host, per preset. Edited
  ONLY by the loop on accept. Includes a tail section listing claims
  from previous commits that have NOT been verified yet.
- `HYPOTHESES.json` — ranked queue of optimizations to test. Top-of-
  list = highest expected ROI. New ideas append at the bottom.
- `run-bench.ps1` — one-shot runner. Wraps `qwen-lora-train --benchmark`,
  enforces preflight checks (idle GPU, cooldown), parses the
  single-line `BENCH {...}` JSON, writes a per-run JSON to `results/`.
- `tdpi-loop.py` — autonomous driver. Pulls a hypothesis, runs the
  bench, compares, writes a verdict to `PERF_LOG.md`. Capped at
  `--max-hours` wall-clock.
- `PERF_LOG.md` — auto-generated chronology. Append-only.
- `results/` — per-run JSON outputs from `run-bench.ps1`.

## Acceptance criterion

A hypothesis is **accepted** iff:

1. The primary metric (e.g. `tok_per_sec`, or `peak_vram_mb` for memory
   wins) moves in the desired direction by at least `threshold_pct`.
2. No other metric regresses by more than `regression_tolerance_pct`
   (default 5%).

Otherwise rejected, with the measured deltas logged to `PERF_LOG.md`.

## Baseline policy

`BENCH_BASELINE.json` is per-host because util / step time depend on
which physical card we ran on. Adding a new host = add a top-level key
with `host.gpu`, `host.vram_mb`, `host.compute_cap`, `host.driver_at_baseline`.

A preset's baseline is the most recent ACCEPTED bench at that preset.
The `_history` array on each preset preserves the prior accepted runs,
so we can audit how much each commit moved the needle.

## Honesty rule

If a commit message says "+10% tok/s" without a paired entry in
`PERF_LOG.md` showing measured baseline-vs-test numbers, it lied.
Treat such claims as untested until a TDPI run is logged.

## Usage

```powershell
# Build first
.\build-lora-train-cuda.bat

# Lock the kokonoe baseline (run on a free idle GPU)
.\bench\run-bench.ps1 q4km-7b-r8-qv-s128-gc

# Then run the autonomous loop, max 4 hours
python bench\tdpi-loop.py --max-hours 4

# Or just one hypothesis
python bench\tdpi-loop.py --hypothesis-only F-cechunk32-s128

# Plan-only, no bench execution
python bench\tdpi-loop.py --dry-run
```

## Constraints

- Each bench is ~20 min (3 warmup + 20 measured optimizer steps).
- 60s cooldown between runs lets thermals settle.
- Loop hard-stops at `--max-hours` regardless of queue state.
- Code-change hypotheses (`requires_code_change: true`) are LOGGED but
  NOT executed by the loop — they require human implementation.
- The loop does not touch git history. It only runs benches and updates
  `BENCH_BASELINE.json` + `PERF_LOG.md` on accept.
