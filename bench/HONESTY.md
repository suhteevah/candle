# HONESTY.md — claim status table

Every measurable claim in the `matt-voice-lora` branch's commits, with
its current verification status. This is the canonical "what's actually
been measured" inventory. Update whenever PERF_LOG.md gets a new
ACCEPT or REJECT verdict.

## Why this file exists

Tonight (2026-04-25) the TDPI loop caught **three** false speed/memory
claims from earlier commits:
1. Chunked CE "frees ~155 MB" — actually OOM'd (twice).
2. `--prequantize-base` "2-3× backward" — never measured (also won't
   fit on 8GB so untestable here).
3. `--tf32` "+5-15% tok/s" — actual measurement was -25.5% tok/s and
   +108% p95 step time.

Pattern: optimization theory not matching real hardware behavior. The
rule from now on: **no perf claim ships in a commit message without
a paired PERF_LOG.md entry showing measured baseline-vs-test deltas.**

## Status table

| ID | Component / claim | Bucket | Status | Evidence |
|---|---|---|---|---|
| PR1 | Pascal sm_60 build | build fix | ✅ build verified, runtime under load NOT verified | cnc cargo build 2026-04-24, no sustained train |
| PR2 | `Tensor::backward_into` API | API addition | ✅ verified | matt-voice training uses it |
| PR3 | `QMatMul` backward (QLoRA unlock) | API addition | ✅ verified | 7B QLoRA loss 9.23 → 3.21 in 7 steps |
| F | `--ce-chunk-size 32` "frees 155 MB" | perf | ❌ REJECT (OOM, both designs) | PERF_LOG.md F-cechunk32-s128 (×2) |
| A | `--prequantize-base` "2-3× backward" | perf | ⏸ untestable on 8GB; will fit on 16GB+ | none |
| H-fuseqkv | `--fuse-qkv` "5-10% attn" | perf | ⏸ requires `--prequantize-base`; untested | none |
| Q | `--merge-adapters` "10-20% decode" | inference perf | ⏸ harness ready, never run | bench/run-bench-infer.ps1 ready |
| K | SSE streaming `/chat/stream` | UX (no perf claim) | ✅ functional | CPU smoke works |
| W | Jinja chat template | UX (no perf claim) | ✅ functional | CPU smoke loads 2507-char template |
| G | `--tf32` "+5-15% tok/s" | perf | ❌ REJECT (-25.5% tok/s, +108% p95) | PERF_LOG.md G-tf32-s128 |
| H | f16 LoRA forward (cast at use) | perf | ❌ REJECT (-12.7% tok/s, +25% p95) | PERF_LOG.md H-loraf16-s128 |
| (D) | per-chunk backward into GradStore | architectural fix (not a perf claim) | ✅ ships (cleaner code) regardless of OOM rejection | code review |

## How to use this table

- **Untested claims** (`⏸`) should be discounted as marketing prose.
- **Rejected claims** (`❌`) should be treated as anti-recommendations
  for THIS hardware. They may hold on different hardware.
- **Verified claims** (`✅`) have a linked artifact (commit, log, run).
- A claim moves from `⏸` to `✅` or `❌` only when a TDPI run logs
  the verdict. Do not promote claims based on "looks faster" or
  "should be faster" — those are how the OOMs and regressions
  ended up in commit messages in the first place.

## Hardware caveat

Tonight's TDPI runs are **all on kokonoe (RTX 3070 Ti 8GB Ampere)**.
Some claims that fail here may pass on other hardware:

- `--prequantize-base` needs >=16GB VRAM (P100 16GB, V100, A4000+,
  etc.). Untestable here.
- `--fuse-qkv` requires `--prequantize-base`, so same constraint.
- Chunked CE may work at >=16GB where the +172 MB overhead has
  headroom.
- TF32 may help on newer Ampere SKUs (A100/4090) where the cuBLAS
  TF32 kernel maturity is better.

The 8GB-Ampere local-optimum hypothesis: tonight's data suggests this
specific configuration is near pareto-optimal already at 5.5 tok/s
baseline, and most "free win" levers run into hard constraints
(memory ceiling, small-matmul TF32 overhead) on this card.

## Bench-window queue

Hypotheses waiting for a clean GPU window:

1. ~~H-loraf16-s128~~ — REJECTED 2026-04-25T18:10 (-12.7% tok/s)
2. Q-merge-inference — 7B inference bench on kokonoe
3. A-prequant-inference — same, with adapter loaded
4. Q-merge-fq-inference — ditto + fused QKV

When cnc 16GB lands (post-blower):
5. A-prequant on 16GB — should actually verify the 2-3× claim
6. H-fuseqkv on 16GB — should verify the 5-10% claim
7. F-cechunk32 on 16GB — may pass with headroom
