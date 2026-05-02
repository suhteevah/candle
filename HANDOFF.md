# Candle Fork Handoff

## Last Updated
2026-05-01

## Project Status
🟢 **7B QLoRA on 8GB Ampere remains rock-solid; TDPI loop has now stress-tested four perf hypotheses and rejected three of them. 5.5 tok/s baseline is the local optimum on this hardware.**

## What Was Done This Session

TDPI (Test-Driven Performance Iteration) loop built and run end-to-end against the locked baseline. Four perf hypotheses tested with measured A/B; honesty discipline restored to the wardoc and commit messages.

### Perf hypotheses tested

Baseline locked at `q4km-7b-r8-qv-s128-gc`: **5.5 tok/s, 7919 MB peak, p95 55963 ms** (commit `d4d24c7`, kokonoe RTX 3070 Ti).

| ID | Win | Verdict | Delta |
|---|---|---|---|
| F-cechunk32-s128 | `--ce-chunk-size 32` (chunked CE w/ retention fix) | ❌ REJECT | OOM ×2 |
| G-tf32-s128 | `--tf32` (CUBLAS_COMPUTE_32F_FAST_TF32) | ❌ REJECT | -25.5% tok/s, +108% p95 |
| H-loraf16-s128 | f16 LoRA forward (cast adapter A/B at use-time) | ❌ REJECT | -12.7% tok/s, +25% p95 |
| (no test) A-prequant-s128 | `--prequantize-base` | ⏸ defer | Adds ~10GB; won't fit on 8GB |

All three rejected hypotheses had the same shape: **theoretical claim from optimization literature, never actually measured against hardware**. The TDPI loop caught them before they shipped to a real training run.

### Code changes (commit `279c210` on `matt-voice-lora`)

- `bench/HONESTY.md` — **new**, canonical claim status table (✅ verified, ❌ rejected, ⏸ untested)
- `bench/PERF_LOG.md` — append G + H REJECT verdicts with full evidence + analysis
- `bench/HYPOTHESES.json` — add G + H, mark F `skip_on_kokonoe: true`
- `bench/run-bench.ps1` — add `q4km-7b-r8-qv-s128-gc-tf32` and `q4km-7b-r8-qv-s128-gc-loraf16` presets
- `qwen-lora-train/lora.rs` — Win H code: dtype-gated adapter cast in `forward_delta` (no-op on default BF16 path; only activates on dtype mismatch)
- `qwen-lora-train/main.rs` — `--tf32` and `--reduced-precision-f16` CLI flags + `set_gemm_reduced_precision_f32(true)` wiring

### Wardoc discipline

`J:/claudeai/scratch/candle-killshot-wardoc.md` got a "Calibration on speed claims" disclaimer at the top. Most "+X% speedup" lines in that doc are NOT verified — anything not paired with a `PERF_LOG.md` entry is now treated as marketing prose until measured.

## Current State

### Working (verified)
- 7B QLoRA training on 8GB at 5.5 tok/s — no changes needed, do not touch
- `Tensor::backward_into` API — used by matt-voice training
- `QMatMul` backward — used by 7B QLoRA path
- TDPI loop infrastructure — `bench/run-bench.ps1`, `bench/tdpi-loop.py`, baseline locked

### Working but DON'T enable
- `--ce-chunk-size N` flag — present, OOMs at any chunk size on 8GB
- `--tf32` flag — present, regresses 25%
- f16 LoRA cast path in `lora.rs` — present, gated to no-op on default BF16; do not force-enable

### Stubbed / scaffolded
- `xentropy.rs::tiled_cross_entropy_with_backward` — wired but OOMs; needs ≥16GB to test
- `prefetch.rs` — caused stalls, disabled

## Blocking Issues

- **kokonoe 8GB ceiling** — 99.4% of card on baseline; no headroom for prequant / fuseqkv / chunked CE. Real perf wins blocked on >=16GB hardware.
- **cnc P100s blocked on power cables + airflow** — proper EPS 8-pin + directed blower fan still pending. Pulled cards from tower 2026-04-24 after thermal failure.
- **`origin` remote (`suhteevah/candle-src.git`) is broken** — push fails with "did not receive expected object". Use `matt` remote (`suhteevah/candle.git`) for the real fork. Branches live there.
- **Local branch `matt-voice-lora` is buried under github-uploader-buildout commits** — `git log` shows 6× "Initial commit" piled on top of `279c210`. The real work is preserved in those commits and pushed to `matt/matt-voice-lora`, but local HEAD does not match remote. Fix: `git reset --hard matt/matt-voice-lora` next session if working locally.

## What's Next

In priority order:

1. **Resume on cnc P100 once blowers land** — that unblocks A-prequant, H-fuseqkv, F-cechunk32 verification (all three were the "wins" tonight rejected for being VRAM-bound on 8GB).
2. **Open the 3 upstream PRs** (already pushed, need URLs hit):
   - `pascal-sm60-compat`
   - `backward-into`
   - `qmatmul-backward`
3. **matt-voice 7B QLoRA overnight on kokonoe** — script ready, Matt launches when ready. Do NOT add `--ce-chunk-size`, `--tf32`, or `--prequantize-base`. Default BF16 path with `--gradient-checkpoint` only.
4. **Win R: tiled QMatMul backward dequant** — would lift seq=512 ceiling on 8GB. Code change required, defer until P100 path validates the value.
5. **Reset local `matt-voice-lora` to match remote** — `git reset --hard matt/matt-voice-lora` to bury the github-uploader noise.

## Notes for Next Session

- **Trust the HONESTY.md table.** If a row says ❌ on this hardware, do not retry it without new evidence. Three perf claims tonight were rejected by measurement — every one of them sounded plausible on paper.
- **No perf claim ships in a commit message without a PERF_LOG.md entry.** Established this session after the F/G/H trifecta. Don't break it.
- **`run-bench.ps1` quirks** — PowerShell parser hates `${var}` and `{0} MB` patterns inside double-quoted strings. Use single quotes + `+` concatenation. Wrap native exe calls in `cmd /c` to avoid `$ErrorActionPreference='Stop'` treating stderr as fatal.
- **Local optimum hypothesis** — 8GB Ampere QLoRA at this config (rank-8, q/v, seq=128, GC) is at hard memory ceiling AND small-matmul throughput floor. Generic optimizations don't apply. Real headroom is dtype/precision changes on the base path, which means a candle-core PR, not a CLI flag.
- **`origin` remote broken** — always `git push matt matt-voice-lora`, never `origin`.
- **Ridge Cell Repair / Mauker rack** — handoff zip ready at `J:/claudeai/scratch/fcp-rack-handoff.zip` for FCP. Blocking question: Snapmaker U1 (270mm bed, 15U) vs Centauri Carbon only (260mm bed, 16U).
